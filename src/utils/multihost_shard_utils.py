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
