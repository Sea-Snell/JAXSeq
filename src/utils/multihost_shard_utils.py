from typing import Any, Callable, List
import jax
import jax.numpy as jnp
import numpy as np
import tree
from jaxtyping import PyTree
from jax.experimental.pjit import pjit
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec

def get_mesh_idxs(process_index: int, mesh_devices: np.ndarray) -> List[int]:
    match_devices = (process_index == np.asarray(tree.map_structure(lambda x: x.process_index, mesh_devices.tolist())))
    match_idxs = np.where(match_devices)
    
    assert len(match_idxs[0]) == (np.prod(mesh_devices.shape) // jax.process_count()), "number devices on host must be the same for all hosts"
    assert all([sorted(list(set(idxs))) == list(range(min(idxs), min(idxs)+len(set(idxs)))) for idxs in match_idxs]), "host devices must form a contiguous chunk"
    
    mesh_idxs = [(min(idxs) // len(set(idxs))) for idxs in match_idxs]
    
    return mesh_idxs

def get_mesh_lens(mesh_devices: np.ndarray) -> List[int]:
    mesh_lens = [0 for _ in range(len(mesh_devices.shape))]
    for process_index in range(jax.process_count()):
        mesh_idxs = get_mesh_idxs(process_index, mesh_devices)
        mesh_lens = [max(mesh_idx+1, mesh_len) for mesh_idx, mesh_len in zip(mesh_idxs, mesh_lens)]
    return mesh_lens

# utils for splitting/re-combining params across hosts

def host_param_shard(host_param_shapes: PyTree, params: PyTree, mesh_devices: np.ndarray, mp_axis: int) -> PyTree:
    mesh_idxs = get_mesh_idxs(jax.process_index(), mesh_devices)
    param_shard_idx = mesh_idxs[mp_axis]

    def split_param(host_shape: Any, param: jnp.ndarray):
        param_shape_arr = jnp.array(param.shape, dtype=jnp.int32)
        host_shape_arr = jnp.array(host_shape.shape, dtype=jnp.int32)
        mask = (param_shape_arr != host_shape_arr).astype(jnp.int32)
        return jax.lax.dynamic_slice(param, mask * host_shape_arr * param_shard_idx, host_shape_arr)
    
    return jax.tree_util.tree_map(split_param, host_param_shapes, params)

def get_host_param_combine_function(param_spec: Any) -> Callable[[PyTree, Mesh, int], PyTree]:
    
    def _get_full_param_at_idx(param: jnp.ndarray) -> jnp.ndarray:
        return param
    
    def _get_full_param_at_idx_p_function(individual_param_spec: Any) -> Callable:
        _p_get_full_param_at_idx= pjit(
            _get_full_param_at_idx, 
            in_axis_resources=individual_param_spec, 
            out_axis_resources=None, 
        )
        return _p_get_full_param_at_idx
    
    _p_get_param_at_idx_tree = jax.tree_util.tree_map(lambda x: _get_full_param_at_idx_p_function(x), param_spec, is_leaf=lambda x: isinstance(x, PartitionSpec) or (x is None))

    def _host_param_combine(host_params: PyTree, mesh: Mesh) -> PyTree:
        with mesh:
            with jax.default_device(jax.devices('cpu')[0]):
                full_params = jax.tree_util.tree_map(lambda f, x: jax.device_get(f(x)), _p_get_param_at_idx_tree, host_params)
        return full_params

    return _host_param_combine

def convert_bsize(bsize: int, mesh_devices: np.ndarray, dp_axis: int) -> int:
    dp_size = get_mesh_lens(mesh_devices)[dp_axis]
    assert (bsize % dp_size) == 0, "batch size must be divisible by the number of data parallel hosts"
    return bsize // dp_size
