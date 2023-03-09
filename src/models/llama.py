from typing import Any, Optional
import jax
from models.base import HuggingfacePjitModelDescription, get_dtype, handle_checkpoint
from transformers_patch.llama_config_remat import LLaMAConfig
from transformers_patch.llama_remat import FlaxLLaMAForCausalLM
from dataclasses import dataclass
from flax.core.frozen_dict import freeze
from jax.experimental import PartitionSpec as P
import jax.numpy as jnp
from transformers.tokenization_utils import PreTrainedTokenizer
from pathlib import Path
import torch
import json
import numpy as np
from jaxtyping import PyTree
from typing import Tuple
from dataclasses import dataclass

# PartitionSpec for LLaMA
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_llama():
    return [
        # embeddings
        (("transformer", "wte", "embedding"), P(None, "mp")), 
        # atention
        (("attention", "(wq|wk|wv)", "kernel"), P(None, "mp")), 
        (("attention", "wo", "kernel"), P("mp", None)), 
        # mlp
        (("feed_forward", "w1", "kernel"), P(None, "mp")), 
        (("feed_forward", "w2", "kernel"), P("mp", None)), 
        (("feed_forward", "w3", "kernel"), P(None, "mp")), 
        # layer norms
        (("attention_norm", "kernel"), P(None)),
        (("ffn_norm", "kernel"), P(None)),
        # output head
        (("transformer", "ln_f", "kernel"), P(None)), 
        (("lm_head", "kernel"), P(None, "mp")), 
    ]

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

def config_from_params(args: ModelArgs, **kwargs) -> LLaMAConfig:
    intermediate_size = int(2 * (args.dim * 4) / 3)
    intermediate_size = args.multiple_of * ((intermediate_size + args.multiple_of - 1) // args.multiple_of)
    vocab_size = args.vocab_size
    if 'vocab_size' in kwargs:
        vocab_size = kwargs.pop('vocab_size')
    return LLaMAConfig(
        vocab_size=vocab_size, 
        hidden_size=args.dim, 
        intermediate_size=intermediate_size, 
        num_hidden_layers=args.n_layers, 
        num_attention_heads=args.n_heads, 
        max_sequence_length=args.max_seq_len, 
        rms_norm_eps=args.norm_eps, 
        **kwargs, 
    )

def convert_llama_weights(ckpt_dir: str, n_tokens: int, max_seq_len: int=2048, 
                          verbose: bool=False, **config_kwargs) -> Tuple[PyTree[np.ndarray], LLaMAConfig]:
    ckpt_paths = sorted(Path(ckpt_dir).glob("*.pth"))
    ckpts = {}
    for i, ckpt_path in enumerate(ckpt_paths):
        if verbose:
            print(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if verbose:
            print('Loaded.')
        ckpts[int(ckpt_path.name.split('.', maxsplit=2)[1])] = checkpoint
    ckpts = [ckpts[i] for i in sorted(list(ckpts.keys()))]
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    
    jax_weights = {
        'transformer': {
            'wte': {'embedding': np.concatenate([ckpt['tok_embeddings.weight'].numpy() for ckpt in ckpts], axis=1)}, 
            'ln_f': {'kernel': ckpts[0]['norm.weight'].numpy()}, 
            'h': {
                '%d' % (layer): {
                    'attention': {
                        'wq': {'kernel': np.concatenate([ckpt['layers.%d.attention.wq.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()}, 
                        'wk': {'kernel': np.concatenate([ckpt['layers.%d.attention.wk.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()}, 
                        'wv': {'kernel': np.concatenate([ckpt['layers.%d.attention.wv.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()}, 
                        'wo': {'kernel': np.concatenate([ckpt['layers.%d.attention.wo.weight' % (layer)].numpy() for ckpt in ckpts], axis=1).transpose()}, 
                    }, 
                    'feed_forward': {
                        'w1': {'kernel': np.concatenate([ckpt['layers.%d.feed_forward.w1.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()}, 
                        'w2': {'kernel': np.concatenate([ckpt['layers.%d.feed_forward.w2.weight' % (layer)].numpy() for ckpt in ckpts], axis=1).transpose()}, 
                        'w3': {'kernel': np.concatenate([ckpt['layers.%d.feed_forward.w3.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()}, 
                    }, 
                    'attention_norm': {'kernel': ckpts[0]['layers.%d.attention_norm.weight' % (layer)].numpy()}, 
                    'ffn_norm': {'kernel': ckpts[0]['layers.%d.ffn_norm.weight' % (layer)].numpy()}, 
                }
            for layer in range(params['n_layers'])}, 
        }, 
        'lm_head': {'kernel': np.concatenate([ckpt['output.weight'].numpy() for ckpt in ckpts], axis=0).transpose()}, 
    }
    params.update({'vocab_size': n_tokens, 'max_seq_len': max_seq_len})
    llama_config = config_from_params(ModelArgs(**params), **config_kwargs)
    return jax_weights, llama_config

configs = {
    "llama_7B": ModelArgs(**{
        "dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": -1, 
    }), 
    "llama_13B": ModelArgs(**{
        "dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-06, "vocab_size": -1, 
    }), 
    "llama_30B": ModelArgs(**{
        "dim": 6656, "multiple_of": 256, "n_heads": 52, "n_layers": 60, "norm_eps": 1e-06, "vocab_size": -1, 
    }), 
    "llama_65B": ModelArgs(**{
        "dim": 8192, "multiple_of": 256, "n_heads": 64, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": -1, 
    }), 
}

# model_str is path to weights directory (e.g. /some/path/7B/)
def load_llama_from_pretrained(model_str, dtype, pad_token_id, n_tokens, n_real_tokens, gradient_checkpoint):
    params, config = convert_llama_weights(model_str, n_tokens, 
                                           n_real_tokens=n_real_tokens, 
                                           gradient_checkpoint=gradient_checkpoint, 
                                           vocab_size=n_tokens, dtype=dtype, 
                                           pad_token_id=pad_token_id)

    model = FlaxLLaMAForCausalLM(config, _do_init=False, dtype=dtype)
    params = jax.tree_map(lambda x: jnp.asarray(x), params)
    
    return model, freeze(params)

# model str is one of the following: {"llama_7B", "llama_13B", "llama_30B", "llama_65B"}
def load_llama_from_local_path(params, model_str, dtype, pad_token_id, n_tokens, n_real_tokens, gradient_checkpoint):
    config = config_from_params(configs[model_str], vocab_size=n_tokens, 
                                dtype=dtype, pad_token_id=pad_token_id, 
                                gradient_checkpoint=gradient_checkpoint, 
                                n_real_tokens=n_real_tokens)
    
    model = FlaxLLaMAForCausalLM(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

# model str is one of the following: {"llama_7B", "llama_13B", "llama_30B", "llama_65B"}
def load_llama_from_random(model_str, dtype, pad_token_id, n_tokens, n_real_tokens, gradient_checkpoint, seed):
    config = config_from_params(configs[model_str], vocab_size=n_tokens, 
                                dtype=dtype, pad_token_id=pad_token_id, 
                                gradient_checkpoint=gradient_checkpoint, 
                                n_real_tokens=n_real_tokens)
    
    model = FlaxLLaMAForCausalLM(config, _do_init=True, dtype=dtype, seed=seed)
    params = model.params
    model = FlaxLLaMAForCausalLM(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_llama_model(model_str: str, from_pretrained: bool, checkpoint_path: Optional[str], 
                    use_fp16: bool, tokenizer: PreTrainedTokenizer, gradient_checkpoint: bool, 
                    seed: int, gcloud_project: Optional[str]=None, gcloud_token: Optional[Any]=None):
    n_tokens = len(tokenizer)

    with jax.default_device(jax.devices('cpu')[0]):
        dtype = get_dtype(use_fp16)
        if checkpoint_path is not None:
            params = handle_checkpoint(
                checkpoint_path, 
                gcloud_project=gcloud_project, 
                gcloud_token=gcloud_token
            )
            model, params = load_llama_from_local_path(params, model_str, dtype, 
                                                      tokenizer.pad_token_id, 
                                                      n_tokens, len(tokenizer), 
                                                      gradient_checkpoint)
        elif from_pretrained:
            model, params = load_llama_from_pretrained(model_str, dtype, 
                                                      tokenizer.pad_token_id, 
                                                      n_tokens, len(tokenizer), 
                                                      gradient_checkpoint)
        else:
            model, params = load_llama_from_random(model_str, dtype, 
                                                  tokenizer.pad_token_id, 
                                                  n_tokens, len(tokenizer), 
                                                  gradient_checkpoint, seed)
    shard_rules = _get_partition_rules_llama()
    return HuggingfacePjitModelDescription(model, params, shard_rules)
