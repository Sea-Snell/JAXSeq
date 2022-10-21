from collections import namedtuple
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import jax
from utils.shard_utils import set_partitions, _id_fn
from flax.core.frozen_dict import freeze
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental.maps import Mesh
import numpy as np
from utils.multihost_shard_utils import host_param_shard, get_mesh_idxs, get_mesh_lens
from jax.random import KeyArray
from optax import softmax_cross_entropy_with_integer_labels
from flax.core.frozen_dict import FrozenDict
import optax
from jaxtyping import PyTree
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from jax.experimental.pjit import pjit
import itertools

class OptimType(Enum):
    AdamW = 1
    AdamWMultiStep = 2
    AdaFactor = 3
    AdaFactorMultiStep = 4

def shard_params(model_init_fn: Callable[[KeyArray], PyTree], params: PyTree, shard_rules: Any, mesh: Mesh, mp_axis: int) -> Tuple[PyTree, PyTree]:

    # dummy rng
    rng = jax.random.PRNGKey(0)

    # specifies how to split model parameters beteen devices
    param_spec = set_partitions(unfreeze(params), shard_rules)

    # initialization function for splitting parameters to devices
    p_get_initial_params = pjit(
        _id_fn, 
        in_axis_resources=(param_spec, None), 
        out_axis_resources=(param_spec, None), 
    )
    
    # initialize parameters from random, used to determining host-level param mapping
    p_model_init_fn = pjit(
        model_init_fn,
        in_axis_resources=(None,), 
        out_axis_resources=param_spec, 
    )
    
    # split the parameters per-host
    with mesh:
        rng, new_rng = jax.random.split(rng)
        host_param_shapes = jax.eval_shape(p_model_init_fn, new_rng)
    with jax.default_device(jax.devices('cpu')[0]):
        params = host_param_shard(host_param_shapes, params, mesh.devices, mp_axis)

    # split the params between all devices
    with mesh:
        params, _ = p_get_initial_params(freeze(params), jnp.ones((), dtype=jnp.uint32))
    
    return params, param_spec

def shard_optim_and_params(model_init_fn: Callable[[KeyArray], PyTree], params: PyTree, shard_rules: Any, mesh: Mesh, mp_axis: int, 
                           optim: optax.GradientTransformation, optim_type: OptimType) -> Tuple[Tuple[PyTree, PyTree], Tuple[PyTree, PyTree]]:
    
    # dummy rng
    rng = jax.random.PRNGKey(0)
    
    # Shard params and optimizer state onto devices
    # Source: https://github.com/huggingface/transformers/blob/main/examples/research_projects/jax-projects/model_parallel/run_clm_mp.py
    def get_initial_state(params):
        opt_state = optim.init(params)
        return opt_state, params
    
    # specifies how to split model parameters beteen devices
    param_spec = set_partitions(unfreeze(params), shard_rules)

    # Get the PyTree for opt_state, we don't actually initialize the opt_state yet.
    class ShapeDtype(object):
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
    params_shapes = jax.tree_util.tree_map(lambda x: ShapeDtype(x.shape, x.dtype), params)
    state_shapes = jax.eval_shape(get_initial_state, params_shapes)

    # get PartitionSpec for opt_state, this is very specific to adamw
    # TODO: optax returns different state for different optimizers, how can we handle this generically ?
    # or maybe we don't since in our examples we just use adamw or adafactor
    def get_opt_spec(x):
        if isinstance(x, (dict, FrozenDict,)):
            return param_spec
        return None
    if optim_type is OptimType.AdamW or optim_type is OptimType.AdamWMultiStep:
        opt_state_spec, param_spec = jax.tree_util.tree_map(
            get_opt_spec, state_shapes, is_leaf=lambda x: isinstance(x, (dict, FrozenDict, optax.EmptyState,))
        )
    elif optim_type is OptimType.AdaFactorMultiStep:
        opt_state_spec, param_spec = jax.tree_util.tree_map(
            get_opt_spec, state_shapes, is_leaf=lambda x: isinstance(x, (dict, FrozenDict, optax.EmptyState,))
        )
        opt_state_spec = opt_state_spec._replace(inner_opt_state=None)
    elif optim_type is OptimType.AdaFactor:
        opt_state_spec = None
    else:
        raise NotImplementedError
    
    # pjit the get_initial_state function to shard params and init
    # optimizer state in sharded way
    p_get_initial_state = pjit(
        get_initial_state, 
        in_axis_resources=(param_spec,), 
        out_axis_resources=(opt_state_spec, param_spec),
    )
    
    # initialize parameters from random, used to determining host-level param mapping
    p_model_init_fn = pjit(
        model_init_fn,
        in_axis_resources=(None,), 
        out_axis_resources=param_spec, 
    )
    
    # split the parameters per-host
    with mesh:
        rng, new_rng = jax.random.split(rng)
        host_param_shapes = jax.eval_shape(p_model_init_fn, new_rng)
    with jax.default_device(jax.devices('cpu')[0]):
        params = host_param_shard(host_param_shapes, params, mesh.devices, mp_axis)

    # split the opt_state and params between all devices
    with mesh:
        opt_state, params = p_get_initial_state(params)
    
    return (params, param_spec), (opt_state, opt_state_spec)

def shard_data_list(data: List[Any], mesh: Mesh, dp_axis: int):
    dp_size = get_mesh_lens(mesh.devices)[dp_axis]
    dp_idx = get_mesh_idxs(jax.process_index(), mesh.devices)[dp_axis]
    return data[dp_idx::dp_size]

def shard_data_iterable(data: Iterable[Any], mesh: Mesh, dp_axis: int):
    dp_size = get_mesh_lens(mesh.devices)[dp_axis]
    dp_idx = get_mesh_idxs(jax.process_index(), mesh.devices)[dp_axis]
    return itertools.islice(data, dp_idx, None, dp_size)
