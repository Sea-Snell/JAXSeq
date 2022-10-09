from typing import Any, Optional
from transformers_patch.opt_config_remat import OPTConfig
from transformers_patch.opt_remat import FlaxOPTForCausalLM
from transformers import OPTForCausalLM
import jax.numpy as jnp
from jax.experimental import PartitionSpec as P
from flax.core.frozen_dict import freeze
from models.base import HuggingfacePjitModelDescription, get_dtype, handle_checkpoint_path
from transformers_patch.load_sharded import from_path
from transformers.tokenization_utils import PreTrainedTokenizer
import jax
import math
from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax

# PartitionSpec for OPT
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_opt():
    return [
        # embeddings
        (("model", "decoder", "embed_positions", "embedding"), P("mp", None)),
        (("model", "decoder", "embed_tokens", "embedding"), P("mp", None)),
        (("model", "decoder", "project_in", "kernel"), None), 
        (("model", "decoder", "project_out", "kernel"), None), 
        # atention
        (("self_attn", "(k_proj|q_proj|v_proj)", "kernel"), P(None, "mp")),
        (("self_attn", "(k_proj|q_proj|v_proj)", "bias"), P("mp")),
        (("self_attn", "out_proj", "kernel"), P("mp", None)),
        (("self_attn", "out_proj", "bias"), P(None)),
        # mlp
        (("fc1", "kernel"), P(None, "mp")),
        (("fc1", "bias"), P("mp")),
        (("fc2", "kernel"), P("mp", None)),
        (("fc2", "bias"), None),
        # layer norms
        (("final_layer_norm", "bias"), None),
        (("final_layer_norm", "scale"), None),
        (("self_attn_layer_norm", "bias"), None),
        (("self_attn_layer_norm", "scale"), None),
        # output head
        (("model", "lm_head", "kernel"), P(None, "mp")), 
    ]

def load_opt_from_pretrained(model_str, dtype, pad_token_id, n_tokens, gradient_checkpoint):
    partitioned_models = ['facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b']
    if model_str in partitioned_models:
        # have to load through pytorch and convert weights manually due to bug with transformers for partitioned weights
        # see: https://github.com/huggingface/transformers/pull/18170
        pytorch_model = OPTForCausalLM.from_pretrained(model_str)
        config = OPTConfig.from_pretrained(model_str, vocab_size=n_tokens, dtype=dtype, 
                                           pad_token_id=pad_token_id, gradient_checkpoint=gradient_checkpoint)
        model = FlaxOPTForCausalLM(config, _do_init=False, dtype=dtype, pad_token_id=pad_token_id)
        params = convert_pytorch_state_dict_to_flax(pytorch_model.state_dict(), model)
    else:
        model, params = FlaxOPTForCausalLM.from_pretrained(model_str, _do_init=False, dtype=dtype, pad_token_id=pad_token_id)
    
    # pad embeddings
    pos_emb = jnp.zeros((4096, model.config.hidden_size))
    pos_emb = pos_emb.at[:2050, :].set(params['model']['decoder']['embed_positions']['embedding'])
    params['model']['decoder']['embed_positions']['embedding'] = pos_emb
    emb = jnp.zeros((n_tokens, model.config.hidden_size))
    emb = emb.at[:model.config.vocab_size, :].set(params["model"]["decoder"]["embed_tokens"]['embedding'])
    params["model"]["decoder"]["embed_tokens"]['embedding'] = emb
    if 'lm_head' in params['model']:
        lm_head_kernel = jnp.zeros((model.config.hidden_size, n_tokens))
        lm_head_kernel = lm_head_kernel.at[:, :model.config.vocab_size].set(params['model']["lm_head"]["kernel"])
        params['model']["lm_head"]["kernel"] = lm_head_kernel
    
    config = OPTConfig.from_pretrained(model_str, vocab_size=n_tokens, dtype=dtype, 
                                        pad_token_id=pad_token_id, gradient_checkpoint=gradient_checkpoint, 
                                        max_position_embeddings=4096-2)
    model = FlaxOPTForCausalLM(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_opt_from_local_path(model_path, dtype, pad_token_id, n_tokens, gradient_checkpoint):
    params = from_path(FlaxOPTForCausalLM, model_path)
    config = OPTConfig.from_pretrained(model_path, vocab_size=n_tokens, dtype=dtype, 
                                        pad_token_id=pad_token_id, gradient_checkpoint=gradient_checkpoint)
    model = FlaxOPTForCausalLM(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_opt_from_random(model_str, dtype, pad_token_id, n_tokens, gradient_checkpoint, seed):
    config = OPTConfig.from_pretrained(model_str, vocab_size=n_tokens, dtype=dtype, 
                                        pad_token_id=pad_token_id, gradient_checkpoint=gradient_checkpoint)
    model = FlaxOPTForCausalLM(config, _do_init=True, dtype=dtype, seed=seed)
    params = model.params
    model = FlaxOPTForCausalLM(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_opt_model(model_str: str, from_pretrained: bool, checkpoint_path: Optional[str], 
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
            model, params = load_opt_from_local_path(checkpoint_path, dtype, 
                                                     tokenizer.pad_token_id, 
                                                     n_tokens, gradient_checkpoint)
            if tmp_dir is not None:
                tmp_dir.cleanup()
        elif from_pretrained:
            model, params = load_opt_from_pretrained(model_str, dtype, 
                                                     tokenizer.pad_token_id, 
                                                     n_tokens, gradient_checkpoint)
        else:
            model, params = load_opt_from_random(model_str, dtype, 
                                                 tokenizer.pad_token_id, 
                                                 n_tokens, gradient_checkpoint, 
                                                 seed)
        params = model.to_fp32(params)
    shard_rules = _get_partition_rules_opt()
    return HuggingfacePjitModelDescription(model, params, shard_rules)
