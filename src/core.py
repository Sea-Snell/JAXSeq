from __future__ import annotations
from collections import namedtuple
from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Protocol
import jax
from utils.shard_utils import set_partitions, _id_fn
from flax.core.frozen_dict import freeze
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental.maps import Mesh
import numpy as np
from utils.multihost_shard_utils import host_param_shard
from jax.random import KeyArray
from optax import softmax_cross_entropy_with_integer_labels
from flax.core.frozen_dict import FrozenDict
import optax
from jaxtyping import PyTree
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from data import block_sequences
from transformers.tokenization_utils import PreTrainedTokenizer
from jax.experimental.pjit import pjit, with_sharding_constraint
from flax import struct
from jax.experimental import PartitionSpec

# utilities

class LogProbsOutput(NamedTuple):
    log_probs: jnp.ndarray
    logits: jnp.ndarray

class TrainStepOutput(NamedTuple):
    loss: jnp.ndarray
    info: Dict[str, Any]
    params: PyTree
    optim_state: PyTree

# general trainer

class _TrainFnType(Protocol):
    def __call__(
        self, 
        params: PyTree, 
        optim_state: PyTree, 
        rng: KeyArray, 
        batch: PyTree, 
    ) -> TrainStepOutput: ...

class Trainer(struct.PyTreeNode):
    params: PyTree
    optim_state: PyTree
    tokenizer: PreTrainedTokenizer = struct.field(pytree_node=False)
    train_fn: _TrainFnType = struct.field(pytree_node=False)
    
    def set_params(self, params: PyTree) -> Inference:
        return self.replace(params=params)
    
    def train_step(self, batch: PyTree, rng_key: KeyArray) -> Tuple[jnp.ndarray, Dict[str, Any], Trainer]:
        
        loss, info, new_params, new_optim_state = self.train_fn(self.params, self.optim_state, rng_key, batch)

        return loss, info, self.replace(params=new_params, optim_state=new_optim_state)

# general inference

class _GenerateFnType(Protocol):
    def __call__(
        self, 
        params: PyTree, 
        rng_key: KeyArray, 
        in_tokens: jnp.ndarray, 
        generation_kwargs: FrozenDict[str, Any], 
    ) -> jnp.ndarray: ...

class _LogProbFnType(Protocol):
    def __call__(
        self, 
        params: PyTree, 
        in_tokens: jnp.ndarray, 
        out_tokens: jnp.ndarray, 
    ) -> LogProbsOutput: ...

class _LossFnType(Protocol):
    def __call__(
        self, 
        params: PyTree, 
        batch: PyTree, 
    ) -> LogProbsOutput: ...

class Inference(struct.PyTreeNode):
    params: PyTree
    tokenizer: PreTrainedTokenizer = struct.field(pytree_node=False)
    generate_fn: _GenerateFnType = struct.field(pytree_node=False)
    logprob_fn: _LogProbFnType = struct.field(pytree_node=False)
    loss_fn: _LossFnType = struct.field(pytree_node=False)

    def set_params(self, params: PyTree) -> Inference:
        return self.replace(params=params)
    
    def generate(self, in_tokens: jnp.ndarray, rng_key: KeyArray, 
                 **generation_kwargs: Dict[str, Any]) -> jnp.ndarray:
        
        outputs = self.generate_fn(self.params, rng_key, in_tokens, freeze(generation_kwargs))
        
        return outputs
    
    def log_probs(self, in_tokens: jnp.ndarray, out_tokens: jnp.ndarray) -> LogProbsOutput:
        
        log_prob_output = self.logprob_fn(self.params, in_tokens, out_tokens)

        return log_prob_output
    
    def eval_loss(self, batch: PyTree) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        
        loss, info = self.loss_fn(self.params, batch)

        return loss, info
