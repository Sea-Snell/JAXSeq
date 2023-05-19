import msgpack
from jaxtyping import PyTree
import flax
from JaxSeq.bucket_manager import open_with_bucket as open
from flax.serialization import to_bytes, from_bytes
from JaxSeq.utils import inplace_float_to_dtype, float_to_dtype
import jax.numpy as jnp
import jax
from typing import Optional
from JaxSeq.utils import multihost_device_put, multihost_device_get

# Adapted from: https://github.com/young-geng/EasyLM/blob/main/EasyLM/checkpoint.py

""" 
Custom msgpack checkpointer that saves large train states by serializing
    and saving tensors one by one in a streaming fashion. Avoids running
    out of memory or local TPU disk with default flax checkpointer. The
        checkpointer saves the train state in an asynchronous manner to avoid
    timing out on JAX barriers in multi-host training.
"""

def save_pytree(
    tree: PyTree, 
    path: str, 
    dtype: jnp.dtype=None, 
    sharding: Optional[PyTree]=None, 
) -> None:
    if sharding is not None:
        sharding = flax.traverse_util.flatten_dict(flax.serialization.to_state_dict(sharding), keep_empty_nodes=True)

    tree = flax.serialization.to_state_dict(tree)
    packer = msgpack.Packer()
    flattend_tree = flax.traverse_util.flatten_dict(tree, keep_empty_nodes=True)
    with open(path, 'wb') as f:
        for key, value in flattend_tree.items():
            curr_sharding = None
            if sharding is not None and value is not flax.traverse_util.empty_node and len(jax.tree_util.tree_leaves(value)) > 0:
                curr_sharding = sharding[tuple(key)]
            tensor = multihost_device_get(value, curr_sharding)
            if dtype is not None:
                tensor = float_to_dtype(tensor, dtype)
            f.write(packer.pack((key, to_bytes(tensor))))

def load_pytree(
    path: str, 
    target: PyTree=None, 
    dtype: jnp.dtype=None, 
    sharding: Optional[PyTree]=None, 
) -> PyTree:
    if sharding is not None:
        sharding = flax.traverse_util.flatten_dict(flax.serialization.to_state_dict(sharding), keep_empty_nodes=True)
    
    flattened_tree = {}
    with open(path, 'rb') as f:
        # 83886080 bytes = 80 MB, which is 16 blocks on GCS
        unpacker = msgpack.Unpacker(f, read_size=83886080, max_buffer_size=0)
        for key, value in unpacker:
            tensor = from_bytes(None, value)
            if dtype is not None:
                tensor = float_to_dtype(tensor, dtype)
            if sharding is not None and tensor is not flax.traverse_util.empty_node and len(jax.tree_util.tree_leaves(tensor)) > 0:
                tensor = multihost_device_put(tensor, sharding[tuple(key)])
            flattened_tree[tuple(key)] = tensor

    tree = flax.traverse_util.unflatten_dict(flattened_tree)
    if target is None:
        return tree
    return flax.serialization.from_state_dict(target, tree)
