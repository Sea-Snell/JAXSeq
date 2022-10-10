from typing import Any, Optional
import jax
from models.base import HuggingfacePjitModelDescription, get_dtype, handle_checkpoint_path
from transformers_patch.gptj_config_remat import GPTJConfig
from transformers_patch.gptj_remat import FlaxGPTJForCausalLM
from dataclasses import dataclass
from flax.core.frozen_dict import unfreeze, freeze
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec as P
from transformers_patch.load_sharded import from_path
import math
import jax.numpy as jnp
from transformers.tokenization_utils import PreTrainedTokenizer

# PartitionSpec for GPTJ
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_gptj():
    return [
        # embeddings
        (("transformer", "wte", "embedding"), P("mp", None)),
        # atention
        (("attn", "(k_proj|q_proj|v_proj)", "kernel"), P(None, "mp")),
        (("attn", "out_proj", "kernel"), P("mp", None)),
        # mlp
        (("mlp", "fc_in", "kernel"), P(None, "mp")),
        (("mlp", "fc_in", "bias"), P("mp")),
        (("mlp", "fc_out", "kernel"), P("mp", None)),
        (("mlp", "fc_out", "bias"), None),
        # layer norms
        ((r"ln_\d+", "bias"), None),
        ((r"\d+", r"ln_\d+", "scale"), None),
        (("ln_f", "bias"), None),
        (("ln_f", "scale"), None),
        # output head
        (("lm_head", "kernel"), P(None, "mp")), 
        (("lm_head", "bias"), P("mp")), 
    ]

def load_gptj_from_pretrained(model_str, dtype, pad_token_id, n_tokens, n_real_tokens, gradient_checkpoint):
    model, params = FlaxGPTJForCausalLM.from_pretrained(model_str, _do_init=False, dtype=dtype, pad_token_id=pad_token_id)
    
    # pad embeddings
    emb = jnp.zeros((n_tokens, model.config.hidden_size))
    emb = emb.at[:model.config.vocab_size, :].set(params["transformer"]["wte"]["embedding"])
    params["transformer"]["wte"]["embedding"] = emb
    lm_head_kernel = jnp.zeros((model.config.hidden_size, n_tokens))
    lm_head_kernel = lm_head_kernel.at[:, :model.config.vocab_size].set(params["lm_head"]["kernel"])
    params["lm_head"]["kernel"] = lm_head_kernel
    lm_head_bias = jnp.zeros((n_tokens,))
    lm_head_bias = lm_head_bias.at[:model.config.vocab_size].set(params["lm_head"]["bias"])
    params["lm_head"]["bias"] = lm_head_bias
    
    config = GPTJConfig.from_pretrained(model_str, vocab_size=n_tokens, dtype=dtype, 
                                        pad_token_id=pad_token_id, gradient_checkpoint=gradient_checkpoint, 
                                        n_real_tokens=n_real_tokens)
    model = FlaxGPTJForCausalLM(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_gptj_from_local_path(model_path, dtype, pad_token_id, n_tokens, n_real_tokens, gradient_checkpoint):
    params = from_path(FlaxGPTJForCausalLM, model_path)
    config = GPTJConfig.from_pretrained(model_path, vocab_size=n_tokens, dtype=dtype, 
                                        pad_token_id=pad_token_id, gradient_checkpoint=gradient_checkpoint, 
                                        n_real_tokens=n_real_tokens)
    model = FlaxGPTJForCausalLM(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_gptj_from_random(model_str, dtype, pad_token_id, n_tokens, n_real_tokens, gradient_checkpoint, seed):
    config = GPTJConfig.from_pretrained(model_str, vocab_size=n_tokens, dtype=dtype, 
                                        pad_token_id=pad_token_id, gradient_checkpoint=gradient_checkpoint, 
                                        n_real_tokens=n_real_tokens)
    model = FlaxGPTJForCausalLM(config, _do_init=True, dtype=dtype, seed=seed)
    params = model.params
    model = FlaxGPTJForCausalLM(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_gptj_model(model_str: str, from_pretrained: bool, checkpoint_path: Optional[str], 
                    use_fp16: bool, tokenizer: PreTrainedTokenizer, gradient_checkpoint: bool, 
                    seed: int, gcloud_project: Optional[str]=None, gcloud_token: Optional[Any]=None):
    # pad token should be last token
    assert tokenizer.pad_token_id == (len(tokenizer)-1)
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
            model, params = load_gptj_from_local_path(checkpoint_path, dtype, 
                                                      tokenizer.pad_token_id, 
                                                      n_tokens, len(tokenizer)-1, 
                                                      gradient_checkpoint)
            if tmp_dir is not None:
                tmp_dir.cleanup()
        elif from_pretrained:
            model, params = load_gptj_from_pretrained(model_str, dtype, 
                                                      tokenizer.pad_token_id, 
                                                      n_tokens, len(tokenizer)-1, 
                                                      gradient_checkpoint)
        else:
            model, params = load_gptj_from_random(model_str, dtype, 
                                                  tokenizer.pad_token_id, 
                                                  n_tokens, len(tokenizer)-1, 
                                                  gradient_checkpoint, seed)
    shard_rules = _get_partition_rules_gptj()
    return HuggingfacePjitModelDescription(model, params, shard_rules)
