from typing import Union, Any, Dict, List, Tuple, Set, FrozenSet, Optional, Callable
import jax
from jaxtyping import PyTree
import flax.linen as nn
from jax.random import KeyArray
from flax.serialization import from_bytes
import pickle as pkl

def rngs_from_keys(rng: KeyArray, keys: Union[List[str], Set[str], Tuple[str], FrozenSet[str]]) -> Dict[str, KeyArray]:
    rngs = {}
    for k in keys:
        rng, new_rng = jax.random.split(rng)
        rngs[k] = new_rng
    return rngs

def split_rng_pytree(rng_pytree: PyTree[KeyArray], splits: int=2) -> PyTree[KeyArray]:
    if len(jax.tree_util.tree_leaves(rng_pytree)) == 0:
        return tuple([rng_pytree for _ in range(splits)])
    outer_tree_def = jax.tree_util.tree_structure(rng_pytree)
    split_rngs = jax.tree_util.tree_map(lambda x: tuple(jax.random.split(x, splits)), rng_pytree)
    return jax.tree_util.tree_transpose(outer_tree_def, jax.tree_util.tree_structure(tuple([0 for _ in range(splits)])), split_rngs)

def load_flax_params(model: nn.Module, rngs: Dict[str, KeyArray], checkpoint_path: Optional[str], 
                     *input_args: List[Any], **input_kwargs: Dict[str, Any]) -> PyTree:
    params = model.init(rngs, *input_args, **input_kwargs)['params']
    if checkpoint_path is not None:
        with open(checkpoint_path, 'rb') as f:
            params = from_bytes(params, pkl.load(f))
    return params
