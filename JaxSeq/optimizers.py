from dataclasses import dataclass
from typing import Tuple, NamedTuple, Dict, Any
import optax
import jax.numpy as jnp
from jaxtyping import PyTree
import jax
from JaxSeq.utils import float_to_dtype
from functools import partial

# Adapted from: https://github.com/young-geng/EasyLM/blob/main/EasyLM/optimizers.py

@dataclass
class GPT3Optimizer:
    init_lr: float=0.0
    end_lr: float=3e-5
    lr: float=3e-4
    lr_warmup_steps: int=3000
    lr_decay_steps: int=300000
    b1: float=0.9
    b2: float=0.95
    clip_gradient: float=1.0
    weight_decay: float=0.1
    bf16_momentum: bool=False
    multiply_by_parameter_scale: bool=False

    def get_optim(self, weight_decay_mask: PyTree) -> Tuple[optax.GradientTransformation, Dict[str, Any]]:

        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=self.init_lr, 
            peak_value=self.lr, 
            warmup_steps=self.lr_warmup_steps, 
            decay_steps=self.lr_decay_steps, 
            end_value=self.end_lr, 
        )

        optimizer_info = dict(
            learning_rate_schedule=learning_rate_schedule, 
        )

        if self.multiply_by_parameter_scale:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.clip_gradient), 
                optax.adafactor(
                    learning_rate=learning_rate_schedule, 
                    multiply_by_parameter_scale=True, 
                    momentum=self.b1, 
                    decay_rate=self.b2, 
                    factored=False, 
                    clipping_threshold=None, 
                    dtype_momentum=jnp.bfloat16 if self.bf16_momentum else jnp.float32, 
                ), 
                optax_add_scheduled_weight_decay(
                    lambda step: -learning_rate_schedule(step) * self.weight_decay,
                    weight_decay_mask, 
                ), 
            )
        else:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.clip_gradient), 
                optax.adamw(
                    learning_rate=learning_rate_schedule, 
                    weight_decay=self.weight_decay, 
                    b1=self.b1, 
                    b2=self.b2, 
                    mask=weight_decay_mask, 
                    mu_dtype=jnp.bfloat16 if self.bf16_momentum else jnp.float32, 
                ), 
            )

        return optimizer, optimizer_info

@dataclass
class PALMOptimizer:
    lr: float=0.01
    lr_warmup_steps: int=10000
    b1: float=0.9
    b2: float=0.99
    clip_gradient: float=1.0
    weight_decay: float=1e-4
    bf16_momentum: bool=False
    
    def get_optim(self, weight_decay_mask: PyTree) -> Tuple[optax.GradientTransformation, Dict[str, Any]]:
        def learning_rate_schedule(step):
            multiplier = self.lr / 0.01
            return multiplier / jnp.sqrt(jnp.maximum(step, self.lr_warmup_steps))

        def weight_decay_schedule(step):
            multiplier = self.weight_decay / 1e-4
            return -multiplier * jnp.square(learning_rate_schedule(step))
        
        optimizer_info = dict(
            learning_rate_schedule=learning_rate_schedule, 
            weight_decay_schedule=weight_decay_schedule, 
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.clip_gradient), 
            optax.adafactor(
                learning_rate=learning_rate_schedule, 
                multiply_by_parameter_scale=True, 
                momentum=self.b1, 
                decay_rate=self.b2, 
                factored=False, 
                clipping_threshold=None, 
                dtype_momentum=jnp.bfloat16 if self.bf16_momentum else jnp.float32, 
            ), 
            optax_add_scheduled_weight_decay(
                weight_decay_schedule, weight_decay_mask, 
            ), 
        )

        return optimizer, optimizer_info

class OptaxScheduledWeightDecayState(NamedTuple):
    count: jnp.DeviceArray

def optax_add_scheduled_weight_decay(schedule_fn, mask=None):
    """ Apply weight decay with schedule. """

    def init_fn(params):
        del params
        return OptaxScheduledWeightDecayState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError('Params cannot be None for weight decay!')

        weight_decay = schedule_fn(state.count)
        updates = jax.tree_util.tree_map(
            lambda g, p: g + weight_decay * p, updates, params
        )
        return updates, OptaxScheduledWeightDecayState(
            count=optax.safe_int32_increment(state.count)
        )

    if mask is not None:
        return optax.masked(optax.GradientTransformation(init_fn, update_fn), mask)
    return optax.GradientTransformation(init_fn, update_fn)
