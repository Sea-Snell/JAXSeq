from __future__ import annotations
from typing import Optional, Union, Callable, Tuple
from JaxSeq.bucket_manager import open_with_bucket as open
from jax.sharding import Mesh
from jax.sharding import NamedSharding
import json
import jax
import jax.numpy as jnp
from JaxSeq.models.llama.model import FlaxLLaMAForCausalLM
from JaxSeq.models.llama.config import LLaMAConfig
from JaxSeq.utils import match_partition_rules, inplace_float_to_dtype
import os
import optax
from flax.training.train_state import TrainState
from JaxSeq.shard_model import shard_train_state_from_checkpoint, shard_train_state_from_params, shard_params_from_params, shard_params_from_config, shard_params_from_checkpoint, get_sharding_from_model
import math
from flax.core import unfreeze, freeze
from enum import Enum
from jaxtyping import PyTree
from transformers.tokenization_utils import PreTrainedTokenizer
from pathlib import Path
import torch
import numpy as np
from dataclasses import dataclass
import tempfile
from JaxSeq.models.llama.tokenizer import LLaMATokenizer

class ModelLoadMode(Enum):
    HF = 'hf'
    OFFICIAL_WEIGHTS = 'official_weights'
    CONFIG = 'config'
    TRAIN_STATE = 'train_state'
    TRAIN_STATE_PARAMS = 'train_state_params'
    PARAMS = 'params'

    @staticmethod
    def match_load_mode(load_mode: Union[ModelLoadMode, str], target: ModelLoadMode):
        if isinstance(load_mode, str):
            return load_mode == target.value
        return load_mode == target

def pad_embeddings(
    params: PyTree, 
    model: FlaxLLaMAForCausalLM, 
    tokenizer: PreTrainedTokenizer, 
    dtype: jnp.dtype=jnp.float32, 
) -> PyTree:
    old_size = model.config.vocab_size
    model.config.vocab_size = int(2**math.ceil(math.log2(len(tokenizer))))
    print(f'Padding embeddings from size {old_size} to size {model.config.vocab_size}. Tokenizer vocab size {len(tokenizer)}.')
    # pad embeddings
    sharding = get_sharding_from_model(model, params)
    return model.pad_embeddings(params, param_sharding=sharding, dtype=dtype)

@dataclass
class OfficialModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

def config_from_llama_params_json(args: OfficialModelArgs, **kwargs) -> LLaMAConfig:
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

def convert_llama_weights(
    ckpt_dir: str, 
    n_tokens: int, 
    max_seq_len: int=2048, 
    verbose: bool=False, 
    **config_kwargs, 
) -> Tuple[PyTree[np.ndarray], LLaMAConfig]:
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
    with open(str(Path(ckpt_dir) / "params.json"), "r") as f:
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
    llama_config = config_from_llama_params_json(OfficialModelArgs(**params), **config_kwargs)
    return jax_weights, llama_config

def load_train_state_from_config(
    model_config: LLaMAConfig, 
    model_dtype: Union[str, jnp.dtype], 
    optim_getter: Callable[[PyTree], optax.GradientTransformation], 
    tokenizer: PreTrainedTokenizer, 
    mesh: Mesh, # should be shape (dp, fsdp, mp)
    prng_key: jax.random.PRNGKeyArray, 
    force_pad_embeddings: bool=False, 
    params_dtype: Optional[Union[str, jnp.dtype]]=jnp.float32, 
) -> Tuple[TrainState, FlaxLLaMAForCausalLM]:
    model = FlaxLLaMAForCausalLM(model_config, dtype=model_dtype, _do_init=False)
    model.config.mesh = mesh
    # shard params
    params = freeze(shard_params_from_config(model, prng_key, params_dtype=params_dtype))
    # pad embeddings
    should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
    if should_pad:
        params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
    # shard train_state
    train_state = shard_train_state_from_params(model, params, optim_getter(params))

    return train_state, model

def load_train_state(
    model_load_mode: Union[ModelLoadMode, str], 
    model_load_path: str, 
    model_dtype: Union[str, jnp.dtype], 
    optim_getter: Callable[[PyTree], optax.GradientTransformation], 
    tokenizer: PreTrainedTokenizer, 
    mesh: Mesh, # should be shape (dp, fsdp, mp)
    prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    force_pad_embeddings: bool=False, 
    params_dtype: Optional[Union[str, jnp.dtype]]=jnp.float32, 
) -> Tuple[TrainState, FlaxLLaMAForCausalLM]:
    
    if ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.HF):
        # load model
        with jax.default_device(jax.devices('cpu')[0]):
            model, params = FlaxLLaMAForCausalLM.from_pretrained(model_load_path, _do_init=False, dtype=model_dtype)
        model.config.mesh = None # None so that padding is not sharded
        # set dtype
        params = unfreeze(params)
        inplace_float_to_dtype(params, params_dtype)
        params = freeze(params)
        # pad embeddings
        should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
        if should_pad:
            with jax.default_device(jax.devices('cpu')[0]):
                params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
        # shard
        model.config.mesh = mesh # back to mesh for final sharding
        params = shard_params_from_params(model, params)
        train_state = shard_train_state_from_params(model, params, optim_getter(params))
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.OFFICIAL_WEIGHTS):
        with jax.default_device(jax.devices('cpu')[0]):
            params, model_config = convert_llama_weights(model_load_path, len(tokenizer))
            params = freeze(jax.tree_map(lambda x: jnp.asarray(x, dtype=params_dtype), params))
        model = FlaxLLaMAForCausalLM(model_config, dtype=model_dtype, _do_init=False)
        model.config.mesh = mesh
        # shard params
        params = shard_params_from_params(model, params)
        # pad embeddings
        should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
        if should_pad:
            params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
        # shard train_state
        train_state = shard_train_state_from_params(model, params, optim_getter(params))
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.CONFIG):
        # load config
        assert prng_key is not None, 'Must provide prng_key when loading from config.'
        with open(model_load_path, 'r') as f:
            model_config = LLaMAConfig.from_dict(json.load(f))
        train_state, model = load_train_state_from_config(
            model_config=model_config, 
            model_dtype=model_dtype, 
            optim_getter=optim_getter, 
            tokenizer=tokenizer, 
            mesh=mesh, 
            prng_key=prng_key, 
            force_pad_embeddings=force_pad_embeddings, 
            params_dtype=params_dtype, 
        )
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.TRAIN_STATE):
        # load model
        with open(os.path.join(model_load_path, 'config.json'), 'r') as f:
            model_config = LLaMAConfig.from_dict(json.load(f))
        model = FlaxLLaMAForCausalLM(model_config, dtype=model_dtype, _do_init=False)
        model.config.mesh = mesh
        # shard and pad embeddings
        should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
        if not should_pad:
            # if no padding, just load train_state, shard as well
            train_state = shard_train_state_from_checkpoint(model, os.path.join(model_load_path, 'train_state.msgpack'), optim_getter, just_params=False, train_state_dtype=params_dtype)
        else:
            # if padding, load params, pad, shard
            params = shard_train_state_from_checkpoint(model, os.path.join(model_load_path, 'train_state.msgpack'), optim_getter, just_params=True, train_state_dtype=params_dtype)
            params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
            train_state = shard_train_state_from_params(model, params, optim_getter(params))
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.TRAIN_STATE_PARAMS):
        # load model
        with open(os.path.join(model_load_path, 'config.json'), 'r') as f:
            model_config = LLaMAConfig.from_dict(json.load(f))
        model = FlaxLLaMAForCausalLM(model_config, dtype=model_dtype, _do_init=False)
        model.config.mesh = mesh
        # load params, shard params
        params = shard_train_state_from_checkpoint(model, os.path.join(model_load_path, 'train_state.msgpack'), optim_getter, just_params=True, train_state_dtype=params_dtype)
        # pad embeddings
        should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
        if should_pad:
            params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
        # shard train_state
        train_state = shard_train_state_from_params(model, params, optim_getter(params))
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.PARAMS):
        # load model
        with open(os.path.join(model_load_path, 'config.json'), 'r') as f:
            model_config = LLaMAConfig.from_dict(json.load(f))
        model = FlaxLLaMAForCausalLM(model_config, dtype=model_dtype, _do_init=False)
        model.config.mesh = mesh
        # load params, shard params
        params = shard_params_from_checkpoint(model, os.path.join(model_load_path, 'params.msgpack'), params_dtype=params_dtype)
        # pad embeddings
        should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
        if should_pad:
            params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
        # shard train_state
        train_state = shard_train_state_from_params(model, params, optim_getter(params))
    else:
        raise ValueError(f"Invalid model_load_mode: {model_load_mode}")
    
    return train_state, model

def load_params_from_config(
    model_config: LLaMAConfig, 
    model_dtype: Union[str, jnp.dtype], 
    tokenizer: PreTrainedTokenizer, 
    mesh: Mesh, # should be shape (dp, fsdp, mp)
    prng_key: jax.random.PRNGKey, 
    force_pad_embeddings: bool=False, 
    params_dtype: Optional[Union[str, jnp.dtype]]=jnp.float32, 
) -> Tuple[PyTree, FlaxLLaMAForCausalLM]:
    model = FlaxLLaMAForCausalLM(model_config, dtype=model_dtype, _do_init=False)
    model.config.mesh = mesh
    # shard params
    params = freeze(shard_params_from_config(model, prng_key, params_dtype=params_dtype))
    # pad embeddings
    should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
    if should_pad:
        params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
    
    return params, model

def load_params(
    model_load_mode: Union[ModelLoadMode, str], 
    model_load_path: str, 
    model_dtype: Union[str, jnp.dtype], 
    tokenizer: PreTrainedTokenizer, 
    mesh: Mesh, # should be shape (dp, fsdp, mp)
    prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    force_pad_embeddings: bool=False, 
    params_dtype: Optional[Union[str, jnp.dtype]]=jnp.float32, 
) -> Tuple[PyTree, FlaxLLaMAForCausalLM]:
    if ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.HF):
        # load model
        with jax.default_device(jax.devices('cpu')[0]):
            model, params = FlaxLLaMAForCausalLM.from_pretrained(model_load_path, _do_init=False, dtype=model_dtype)
        model.config.mesh = None # None so that padding is not sharded
        # set dtype
        params = unfreeze(params)
        inplace_float_to_dtype(params, params_dtype)
        params = freeze(params)
        # pad embeddings
        should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
        if should_pad:
            with jax.default_device(jax.devices('cpu')[0]):
                params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
        # shard
        model.config.mesh = mesh # back to mesh for final sharding
        params = shard_params_from_params(model, params)
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.OFFICIAL_WEIGHTS):
        with jax.default_device(jax.devices('cpu')[0]):
            params, model_config = convert_llama_weights(model_load_path, len(tokenizer))
            params = freeze(jax.tree_map(lambda x: jnp.asarray(x, dtype=params_dtype), params))
        model = FlaxLLaMAForCausalLM(model_config, dtype=model_dtype, _do_init=False)
        model.config.mesh = mesh
        # shard params
        params = shard_params_from_params(model, params)
        # pad embeddings
        should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
        if should_pad:
            params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.CONFIG):
        # load config
        assert prng_key is not None, 'Must provide prng_key when loading from config.'
        with open(model_load_path, 'r') as f:
            model_config = LLaMAConfig.from_dict(json.load(f))
        params, model = load_params_from_config(
            model_config=model_config, 
            model_dtype=model_dtype, 
            tokenizer=tokenizer, 
            mesh=mesh, 
            prng_key=prng_key, 
            force_pad_embeddings=force_pad_embeddings, 
            params_dtype=params_dtype, 
        )
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.PARAMS):
        # load model
        with open(os.path.join(model_load_path, 'config.json'), 'r') as f:
            model_config = LLaMAConfig.from_dict(json.load(f))
        model = FlaxLLaMAForCausalLM(model_config, dtype=model_dtype, _do_init=False)
        model.config.mesh = mesh
        # load params, shard params
        params = shard_params_from_checkpoint(model, os.path.join(model_load_path, 'params.msgpack'), params_dtype=params_dtype)
        # pad embeddings
        should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
        if should_pad:
            params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
    else:
        raise ValueError(f"Invalid model_load_mode: {model_load_mode}")
    
    return params, model

def load_tokenizer(tokenizer_path: str, **kwargs) -> LLaMATokenizer:
    # load to temp file first since tokenizer doesn't support gcs loading
    with open(tokenizer_path, 'rb') as f:
        tokenizer_fp = tempfile.NamedTemporaryFile('wb')
        tokenizer_fp.write(f.read())
        tokenizer_path = tokenizer_fp.name

    tokenizer = LLaMATokenizer(
        tokenizer_path, 
        **kwargs, 
    )
    
    tokenizer_fp.close()

    return tokenizer
