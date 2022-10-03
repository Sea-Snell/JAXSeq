from typing import Dict, List, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import tree
from jaxtyping import PyTree

def index_under_mesh(real_process_index, mesh_devices, mp_axis):
    match_points = []
    for i in range(mesh_devices.shape[mp_axis]):
        process_id_match = (real_process_index == np.asarray(tree.map_structure(lambda x: x.process_index, np.take(mesh_devices, i, axis=mp_axis).tolist())))
        is_match = np.all(process_id_match)
        some_match = np.any(process_id_match)
        assert is_match or (not some_match), "host devices must form a contiguous chunk"
        if is_match:
            match_points.append(i)
    assert len(match_points) == (mesh_devices.shape[mp_axis] // jax.process_count()), "number param devices on host must be the same for all hosts"
    assert sorted(match_points) == list(range(min(match_points), min(match_points)+len(match_points))), "host devices must form a contiguous chunk"
    process_idx = min(match_points) // len(match_points)
    return process_idx

# utils for splitting params per-host
def host_param_shard(host_param_shapes, params, mesh_devices, mp_axis):
    shard_idx = index_under_mesh(jax.process_index(), mesh_devices, mp_axis)
    def split_param(host_shape, param):
        param_shape_arr = jnp.array(param.shape, dtype=jnp.int32)
        host_shape_arr = jnp.array(host_shape.shape, dtype=jnp.int32)
        mask = (param_shape_arr != host_shape_arr).astype(jnp.int32)
        return jax.lax.dynamic_slice(param, mask * host_shape_arr * shard_idx, host_shape_arr)
    return jax.tree_util.tree_map(split_param, host_param_shapes, params)

# assumes mesh devices are the same as when the parameters were saved
# TODO: Test this.
def combine_host_param_shards(full_param_shapes, p_idx_to_shard_params: Dict[int, PyTree], mesh_devices, mp_axis):
    p_idx_2_shard_idx = {}
    for p_idx in p_idx_to_shard_params.keys():
        p_idx_2_shard_idx[p_idx] = index_under_mesh(p_idx, mesh_devices, mp_axis)
    
    def empty_params(full_shape, shard_param):
        return jnp.empty(full_shape, dtype=shard_param.dtype)
    
    full_params = jax.tree_util.tree_map(empty_params, full_param_shapes, next(p_idx_to_shard_params.values()))

    def combine_param(empty_full_param, param_shard, shard_idx):
        full_param_shape_arr = jnp.array(empty_full_param.shape, dtype=jnp.int32)
        shard_shape_arr = jnp.array(param_shard.shape, dtype=jnp.int32)
        mask = (full_param_shape_arr != shard_shape_arr).astype(jnp.int32)
        return empty_full_param.at[(mask*shard_shape_arr*shard_idx):(mask*shard_shape_arr*(shard_idx+1))].set(param_shard)
    
    for real_p_idx, shard_idx in p_idx_2_shard_idx.items():
        full_params = jax.tree_util.tree_map(combine_param, full_params, p_idx_to_shard_params[real_p_idx], shard_idx)
    
    return full_params
