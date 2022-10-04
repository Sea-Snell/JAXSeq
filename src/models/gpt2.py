from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Optional
import jax
import jax.numpy as jnp
from models.base import HuggingfacePjitModelDescription, get_dtype, handle_checkpoint_path
from transformers_patch.gpt2_config_remat import GPT2Config
from transformers_patch.gpt2_remat import FlaxGPT2LMHeadModel
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import unfreeze, freeze
from jax.experimental import PartitionSpec as P
from transformers_patch.load_sharded import from_path
import math
from transformers.tokenization_utils import PreTrainedTokenizer

# PartitionSpec for GPT2
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_gpt2():
    return [
        # embeddings
        (("transformer", "wpe", "embedding"), P("mp", None)),
        (("transformer", "wte", "embedding"), P("mp", None)),
        # atention
        (("attn", "(q_attn|c_attn)", "kernel"), P(None, "mp")),
        (("attn", "(q_attn|c_attn)", "bias"), P("mp")),
        (("attn", "c_proj", "kernel"), P("mp", None)),
        (("attn", "c_proj", "bias"), None),
        # mlp
        (("mlp", "c_fc", "kernel"), P(None, "mp")),
        (("mlp", "c_fc", "bias"), P("mp")),
        (("mlp", "c_proj", "kernel"), P("mp", None)),
        (("mlp", "c_proj", "bias"), None),
        # layer norms
        ((r"ln_\d+", "bias"), None),
        ((r"\d+", r"ln_\d+", "scale"), None),
        (("ln_f", "bias"), None),
        (("ln_f", "scale"), None),
    ]

# Source: https://github.com/huggingface/transformers/tree/main/examples/research_projects/jax-projects/model_parallel
def load_gpt2_from_pretrained(model_str, dtype, pad_token_id, n_tokens, gradient_checkpoint):
    model, params = FlaxGPT2LMHeadModel.from_pretrained(model_str, _do_init=False, dtype=dtype, pad_token_id=pad_token_id)
    
    # pad embeddings
    emb = jnp.zeros((n_tokens, model.config.hidden_size))
    emb = emb.at[:50257, :].set(params["transformer"]["wte"]["embedding"])
    params["transformer"]["wte"]["embedding"] = emb
    
    config = GPT2Config.from_pretrained(model_str, vocab_size=n_tokens, dtype=dtype, 
                                        pad_token_id=pad_token_id, gradient_checkpoint=gradient_checkpoint)
    model = FlaxGPT2LMHeadModel(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_gpt2_from_local_path(model_path, dtype, pad_token_id, n_tokens, gradient_checkpoint):
    params = from_path(FlaxGPT2LMHeadModel, model_path)
    config = GPT2Config.from_pretrained(model_path, vocab_size=n_tokens, dtype=dtype, 
                                        pad_token_id=pad_token_id, gradient_checkpoint=gradient_checkpoint)
    model = FlaxGPT2LMHeadModel(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_gpt2_from_random(model_str, dtype, pad_token_id, n_tokens, gradient_checkpoint, seed):
    config = GPT2Config.from_pretrained(model_str, vocab_size=n_tokens, dtype=dtype, 
                                        pad_token_id=pad_token_id, gradient_checkpoint=gradient_checkpoint)
    model = FlaxGPT2LMHeadModel(config, _do_init=True, dtype=dtype, seed=seed)
    params = model.params
    model = FlaxGPT2LMHeadModel(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_gpt2_model(model_str: str, from_pretrained: bool, checkpoint_path: Optional[str], 
                    use_fp16: bool, tokenizer: PreTrainedTokenizer, gradient_checkpoint: bool, 
                    seed: int, gcloud_project: Optional[str]=None, gcloud_token: Optional[Any]=None):
    # make n_tokens a power of 2, so parameters can be shareded evanely across devices
    n_tokens=int(2**math.ceil(math.log2(len(tokenizer))))

    with jax.default_device(jax.devices('cpu')[0]):
        dtype = get_dtype(use_fp16)
        if checkpoint_path is not None:
            checkpoint_path, tmp_dir = handle_checkpoint_path(
                checkpoint_path, 
                gcloud_project=gcloud_project, 
                gcloud_token=gcloud_token
            )
            model, params = load_gpt2_from_local_path(checkpoint_path, dtype, 
                                                      tokenizer.pad_token_id, 
                                                      n_tokens, gradient_checkpoint)
            if tmp_dir is not None:
                tmp_dir.cleanup()
        elif from_pretrained:
            model, params = load_gpt2_from_pretrained(model_str, dtype, 
                                                      tokenizer.pad_token_id, 
                                                      n_tokens, gradient_checkpoint)
        else:
            model, params = load_gpt2_from_random(model_str, dtype, 
                                                  tokenizer.pad_token_id, 
                                                  n_tokens, gradient_checkpoint, 
                                                  seed)
    shard_rules = _get_partition_rules_gpt2()
    return HuggingfacePjitModelDescription(model, params, shard_rules)
