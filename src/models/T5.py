from lib2to3.pgen2.tokenize import tokenize
from typing import Any, Optional
import jax.numpy as jnp
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers_patch.t5_remat import FlaxT5ForConditionalGeneration
from transformers_patch.t5_config_remat import T5Config
from flax.core.frozen_dict import freeze
from jax.experimental import PartitionSpec as P
from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
from transformers_patch.load_sharded import from_path
from transformers.tokenization_utils import PreTrainedTokenizer
import math
from models.base import HuggingfacePjitModelDescription, get_dtype, handle_checkpoint_path
import jax
import tempfile

# PartitionSpec for T5v1.1
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_t5_v1_1():
    return [
        # embeddings
        (("shared", "embedding"), P("mp", None)),
        (("relative_attention_bias", "embedding"), None), 
        # self atention
        (("SelfAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("SelfAttention", "o", "kernel"), P("mp", None)),
        # cross atention
        (("EncDecAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("EncDecAttention", "o", "kernel"), P("mp", None)),
        # mlp
        (("DenseReluDense", "wi_0", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wi_1", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wo", "kernel"), P("mp", None)),
        # layer norms
        (("layer_norm", "weight"), None), 
        (("final_layer_norm", "weight"), None), 
        # output head
        (("lm_head", "kernel"), P(None, "mp")), 
    ]

# PartitionSpec for T5
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_t5():
    return [
        # embeddings
        (("shared", "embedding"), P("mp", None)),
        (("relative_attention_bias", "embedding"), None), 
        # self atention
        (("SelfAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("SelfAttention", "o", "kernel"), P("mp", None)),
        # cross atention
        (("EncDecAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("EncDecAttention", "o", "kernel"), P("mp", None)),
        # mlp
        (("DenseReluDense", "wi", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wo", "kernel"), P("mp", None)),
        # layer norms
        (("layer_norm", "weight"), None), 
        (("final_layer_norm", "weight"), None), 
        # output head
        (("lm_head", "kernel"), P(None, "mp")), 
    ]

def load_t5_from_pretrained(model_str, dtype, pad_token_id, n_tokens, n_real_tokens, gradient_checkpoint):
    try:
        model, params = FlaxT5ForConditionalGeneration.from_pretrained(model_str, _do_init=False, dtype=dtype)
    except:
        model = FlaxT5ForConditionalGeneration.from_pretrained(model_str, _do_init=True, from_pt=True, dtype=dtype)
        params = model.params
    
    # pad embeddings
    if "shared" in params and "embedding" in params["shared"]:
        emb = jnp.zeros((n_tokens, model.config.hidden_size))
        emb = emb.at[:model.config.vocab_size, :].set(params["shared"]["embedding"])
        params["shared"]["embedding"] = emb    
    if "lm_head" in params:
        lm_head_kernel = jnp.zeros((model.config.hidden_size, n_tokens))
        lm_head_kernel = lm_head_kernel.at[:, :model.config.vocab_size].set(params["lm_head"]["kernel"])
        params["lm_head"]["kernel"] = lm_head_kernel
        if "bias" in params["lm_head"]:
            lm_head_bias = jnp.zeros((n_tokens,))
            lm_head_bias = lm_head_bias.at[:model.config.vocab_size].set(params["lm_head"]["bias"])
            params["lm_head"]["bias"] = lm_head_bias

    config = T5Config.from_pretrained(model_str, dtype=dtype, gradient_checkpointing=gradient_checkpoint, 
                                      pad_token_id=pad_token_id, vocab_size=n_tokens, n_real_tokens=n_real_tokens)
    model = FlaxT5ForConditionalGeneration(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_t5_from_local_path(model_path, dtype, pad_token_id, n_tokens, n_real_tokens, gradient_checkpoint):
    params = from_path(FlaxT5ForConditionalGeneration, model_path)
    config = T5Config.from_pretrained(model_path, dtype=dtype, gradient_checkpointing=gradient_checkpoint, 
                                      pad_token_id=pad_token_id, vocab_size=n_tokens, n_real_tokens=n_real_tokens)
    model = FlaxT5ForConditionalGeneration(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_t5_from_random(model_str, dtype, pad_token_id, n_tokens, n_real_tokens, gradient_checkpoint, seed):
    config = T5Config.from_pretrained(model_str, dtype=dtype, gradient_checkpointing=gradient_checkpoint, 
                                      pad_token_id=pad_token_id, vocab_size=n_tokens, n_real_tokens=n_real_tokens)
    model = FlaxT5ForConditionalGeneration(config, _do_init=True, dtype=dtype, seed=seed)
    params = model.params
    model = FlaxT5ForConditionalGeneration(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_t5_model(model_str: str, from_pretrained: bool, checkpoint_path: Optional[str], 
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
            model, params = load_t5_from_local_path(checkpoint_path, dtype, 
                                                    tokenizer.pad_token_id, 
                                                    n_tokens, len(tokenizer), gradient_checkpoint)
            if tmp_dir is not None:
                tmp_dir.cleanup()
        elif from_pretrained:
            model, params = load_t5_from_pretrained(model_str, dtype, 
                                                    tokenizer.pad_token_id, 
                                                    n_tokens, len(tokenizer), gradient_checkpoint)
        else:
            model, params = load_t5_from_random(model_str, dtype, 
                                                tokenizer.pad_token_id, 
                                                n_tokens, len(tokenizer), gradient_checkpoint, 
                                                seed)
    if 'v1_1' in model_str or 'lm-adapt' in model_str:
        shard_rules = _get_partition_rules_t5_v1_1()
    else:
        shard_rules = _get_partition_rules_t5()
    return HuggingfacePjitModelDescription(model, params, shard_rules)


# data utility

def prepend_pad(output_str: str) -> str:
    return '<pad> ' + output_str if not output_str.startswith('<pad>') else output_str
