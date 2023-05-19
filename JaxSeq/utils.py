import jax
from jax.sharding import PartitionSpec as PS
from jax._src.tree_util import KeyPath as KeyPath
import re
import numpy as np
import jax.numpy as jnp
from typing import List, Tuple
from jax.sharding import Mesh, NamedSharding
from jaxtyping import PyTree
from typing import NamedTuple, Any, Union, Optional, Generator, Dict, Iterator, Callable, Iterable
from enum import Enum
import os
import json
from jax.experimental import mesh_utils, multihost_utils
from flax.core import FrozenDict
from abc import ABC, abstractmethod
import collections
import itertools
import uuid
from datetime import datetime
from JaxSeq.bucket_manager import open_with_bucket as open
import pickle as pkl
from jax.experimental.pjit import pjit
from functools import reduce
from difflib import SequenceMatcher

ASSERT_EQUAL_PER_HOST = os.getenv('MULTIHOST_DEVICE_PUT_ASSERT_EQUAL_PER_HOST', '0') == '1'

# define convertpath as a class so that it is easy to override the default project root
class ConvertPath:
    DEFAULT_PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    @classmethod
    def convert(cls, path: Optional[str], project_root: Optional[str]=None) -> Optional[str]:
        """convert relative paths to be absolute paths from project root"""
        if path is None:
            return None
        if project_root is None:
            project_root = cls.DEFAULT_PROJECT_ROOT
        if path.startswith('~/'):
            return os.path.expanduser(path)
        if path.startswith('/') or path.startswith('gcs://'):
            return path
        return os.path.join(project_root, path)
convert_path = ConvertPath().convert # set convert_path function

def create_path(path: Optional[str]) -> None:
    if path is None:
        return None
    if (not path.startswith('gcs://')) and (not os.path.exists(path)):
        os.makedirs(path)

def get_enabled_save_path(path: str, enabled: bool=True) -> str:
    if enabled:
        return path
    return '/dev/null'

def match_partition_rules(rules: List[Tuple[str, PS]], params: PyTree) -> PyTree:
    """ Returns a pytree of PartitionSpec according to rules.
    """
    def get_partition_spec(path, leaf):
        if isinstance(leaf, int) or isinstance(leaf, float) or len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
            """ Don't partition scalar values. """
            return PS()
        for rule, ps in rules:
            if re.search(rule, jax.tree_util.keystr(path)) is not None:
                return ps
        raise ValueError(f'Partition rule not found for param: {path}')
    return jax.tree_util.tree_map_with_path(get_partition_spec, params)

def get_weight_decay_mask(exclusions):
    """ Return a weight decay mask function that computes the pytree masks
        according to the given exclusion rules.
    """
    def decay(name, _):
        for rule in exclusions:
            if re.search(rule, jax.tree_util.keystr(name)) is not None:
                return False
        return True

    def weight_decay_mask(params):
        return jax.tree_util.tree_map_with_path(decay, params)

    return weight_decay_mask

def get_dtype(use_fp16: bool) -> jnp.dtype:
    """util for getting the correct dtype for the backend."""
    if use_fp16:
        return jnp.bfloat16
    return jnp.float32


def global_mesh_defined() -> bool:
    """Checks if global xmap/pjit mesh resource environment is defined."""
    maps_env = jax.experimental.maps.thread_resources.env
    return maps_env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison

def with_sharding_constraint(x: jax.Array, axis_resources) -> jax.Array:
    """Wrapper for with_sharding_constraint, no-op on cpu or outside pjit."""
    if jax.devices()[0].platform == 'cpu' or not global_mesh_defined():
        return x
    else:
        return jax.lax.with_sharding_constraint(x, axis_resources)

def with_named_sharding_constraint(x: jax.Array, mesh: Mesh, partition_spec: PS) -> jax.Array:
    """since we often use named sharding, this is a helper function to make it easier to use"""
    if mesh is not None:
        return with_sharding_constraint(x, NamedSharding(mesh, partition_spec))
    return x

def multihost_device_put(
    x: jax.Array, 
    sharding: Optional[Union[PS, jax.sharding.NamedSharding]]=None, 
    assert_equal_per_host: Optional[bool]=None, 
) -> jax.Array:
    """Similar to device_put, but works in multi-host environments."""
    if assert_equal_per_host is None:
        assert_equal_per_host = ASSERT_EQUAL_PER_HOST
    if assert_equal_per_host:
        multihost_utils.assert_equal(
            x, fail_message='multihost_device_put: input array differs between hosts', 
        )
    
    if sharding is None:
        p_shard_fn = lambda x: x
    elif isinstance(sharding, PS):
        p_shard_fn = pjit(
            multihost_utils._identity_fn, 
            in_shardings=PS(), 
            out_shardings=sharding, 
        )
    else:
        p_shard_fn = pjit(
            multihost_utils._identity_fn, 
            in_shardings=NamedSharding(sharding.mesh, PS()), 
            out_shardings=sharding, 
        )
    
    # if on cpu convert to numpy to avoid errors
    if isinstance(x, jax.Array) and any([device in jax.devices('cpu') for device in x.devices()]) and jax.default_backend() != 'cpu':
        x = np.asarray(x)
    x = p_shard_fn(x)
    return x

def multihost_device_get(
    x: jax.Array, 
    sharding: Optional[Union[PS, NamedSharding]]=None, 
    mesh: Optional[Mesh]=None, 
    assert_equal_per_host: Optional[bool]=None, 
) -> jax.Array:
    """Similar to device_get, but works in multi-host environments."""
    # takes in shardings for x to enforce input shardings, or just mesh to not enforce input shardings
    if assert_equal_per_host is None:
        assert_equal_per_host = ASSERT_EQUAL_PER_HOST
    if assert_equal_per_host:
        multihost_utils.assert_equal(
            x, fail_message='multihost_device_get: input array differs between hosts', 
        )
    
    if sharding is None and mesh is None:
        p_gather_fn = lambda x: x
    elif sharding is None:
        p_gather_fn = pjit(
            multihost_utils._identity_fn, 
            out_shardings=NamedSharding(mesh, PS()), 
        )
    elif isinstance(sharding, PS):
        p_gather_fn = pjit(
            multihost_utils._identity_fn, 
            in_shardings=sharding, 
            out_shardings=PS(), 
        )
    elif isinstance(sharding, NamedSharding):
        p_gather_fn = pjit(
            multihost_utils._identity_fn, 
            in_shardings=sharding, 
            out_shardings=NamedSharding(sharding.mesh, PS()), 
        )
    else:
        raise NotImplementedError

    x = jax.device_get(p_gather_fn(x))
    return x

class Padding(Enum):
    """Enum for padding type."""
    LEFT = 0
    RIGHT = 1

class Truncation(Enum):
    """Enum for truncation type."""
    LEFT = 0
    RIGHT = 1

class BlockingStrategy(NamedTuple):
    """Tokenization strategy for querying models."""
    padding: Padding
    truncation: Truncation
    max_length: Optional[int]


def pad_sequence(
    sequence: np.ndarray, 
    max_len: int, 
    pad_value: Any, 
    padding: Padding=Padding.RIGHT, 
) -> np.ndarray:
    """pad an individual sequence"""
    assert sequence.shape[0] <= max_len, 'sequence has size larger than max_len'
    
    if isinstance(pad_value, np.ndarray):
        pad_tokens = np.full((max_len-sequence.shape[0],)+pad_value.shape, pad_value)
    else:
        pad_tokens = np.full((max_len-sequence.shape[0],), pad_value)
    
    if padding == Padding.RIGHT:
        return np.concatenate((sequence, pad_tokens), axis=0)
    return np.concatenate((pad_tokens, sequence), axis=0)


def block_sequences(
    sequences: Union[List[List[Any]], np.ndarray, List[np.ndarray]], 
    pad_value: Any, 
    dtype: Any, 
    blocking_strategy: BlockingStrategy, 
) -> np.ndarray:
    """convert a list of sequences into a padded and truncated numpy array"""
    max_len = blocking_strategy.max_length
    if max_len is None:
        max_len = max(map(lambda x: len(x), sequences))
    
    full_sequences = []
    for i in range(len(sequences)):
        if blocking_strategy.truncation == Truncation.RIGHT or max_len == 0:
            new_toks = sequences[i][:max_len]
        elif blocking_strategy.truncation == Truncation.LEFT:
            new_toks = sequences[i][-max_len:]
        full_sequences.append(pad_sequence(np.asarray(new_toks), max_len, pad_value, padding=blocking_strategy.padding))
    return np.asarray(full_sequences, dtype=dtype)


def pack_sequences(
    sequences: Union[List[List[Any]], np.ndarray, List[np.ndarray]], 
    dtype: Any, 
    max_len: Optional[int]=None, 
    pad_value: Optional[Any]=None, 
    initial_buffer: Optional[List[Any]]=None, 
) -> np.ndarray:
    """convert a list of sequences into a padded and truncated numpy array with sequences chunked together to fit max_length"""
    if max_len is None:
        max_len = max(map(lambda x: len(x), sequences))
    if initial_buffer is None:
        initial_buffer = []
    
    full_sequences = []
    buffer = list(initial_buffer)
    for sequence in sequences:
        buffer.extend(sequence)
        while len(buffer) > max_len:
            full_sequences.append(buffer[:max_len])
            buffer = list(initial_buffer)+buffer[max_len:]
    if len(buffer) > 0 and pad_value is not None:
        full_sequences.append(pad_sequence(np.asarray(buffer), max_len, pad_value, padding=Padding.RIGHT))
    return np.asarray(full_sequences, dtype=dtype)

def pack_sequences_stream(
    sequences: Union[Iterable[List[Any]], Iterable[np.ndarray]], 
    dtype: Any, 
    max_len: int, 
    pad_value: Optional[Any]=None, 
    initial_buffer: Optional[List[Any]]=None, 
) -> Generator[np.ndarray, None, None]:
    """convert a iterable of sequences into a padded and truncated numpy array with sequences chunked together to fit max_length"""
    
    buffer = list(initial_buffer) if initial_buffer is not None else []
    for sequence in sequences:
        buffer.extend(sequence)
        while len(buffer) > max_len:
            yield np.asarray(buffer[:max_len], dtype=dtype)
            buffer = (list(initial_buffer) if initial_buffer is not None else [])+buffer[max_len:]
    if len(buffer) > 0 and pad_value is not None:
        yield np.asarray(pad_sequence(np.asarray(buffer), max_len, pad_value, padding=Padding.RIGHT), dtype=dtype)


def patch_call(instance, func):
    """Overrides the __call__ function in a class. Useful for changing tokenizer defaults."""
    class _(type(instance)):
        def __call__(self, *arg, **kwarg):
           return func(*arg, **kwarg)
    instance.__class__ = _

class FileOpenIterable:
    """create file iterator object. pipe will be applied to the file object before iteration."""
    def __init__(self, path, *open_args, pipe: Optional[Callable[[Iterator], Iterator]]=None, **open_kwargs):
        self.path = path
        self.pipe = pipe
        self.open_args = open_args
        self.open_kwargs = open_kwargs
    
    def __iter__(self):
        f = open(self.path, *self.open_args, **self.open_kwargs)
        if self.pipe is not None:
            f = self.pipe(f)
        return f

class MapIterable:
    def __init__(self, map_fn: Callable, *iterables):
        self.map_fn = map_fn
        self.iterables = iterables
    
    def __iter__(self):
        return map(self.map_fn, *self.iterables)

def jsonl_stream(fp: Iterator) -> Generator[Dict, None, None]:
    """Generator for reading jsonl files."""
    for line in fp:
        line = line.strip()
        if line == '':
            continue
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            continue
        yield data

def jsonl_load(fp: Iterator) -> List[Dict]:
    """Loads a jsonl file into a list of dicts."""
    return list(jsonl_stream(fp))


def load_mesh(shape: Tuple[int], axis_names: Tuple[str]) -> Mesh:
    """load device mesh with data parallel on the first axis and model parallel on the second axis"""
    assert sum(map(lambda x: int(x == -1), shape)) <= 1, "only one of the mesh dimensions can be -1"
    if -1 in shape:
        shape = list(shape)
        prod = reduce(lambda a, b: a*b, filter(lambda x: x != -1, shape), 1)
        shape[shape.index(-1)] = int(len(jax.devices()) / prod)
        shape = tuple(shape)
    devices = mesh_utils.create_device_mesh(shape)
    return Mesh(devices, axis_names=axis_names)

def float_to_dtype(tree, dtype=jnp.float32):
    """ Convert all float(fp16, bf16, fp32, fp64) arrays in a pytree to a given
        dtype.
    """
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)
    assert dtype in float_dtypes, 'dtype must be a float type!'
    def to_dtype(x):
        if isinstance(x, (np.ndarray, jnp.ndarray)):
            if x.dtype in float_dtypes and x.dtype != dtype:
                x = x.astype(dtype)
        return x
    return jax.tree_util.tree_map(to_dtype, tree)


def inplace_float_to_dtype(tree, dtype=jnp.float32):
    """ Convert all float(fp16, bf16, fp32, fp64) arrays in a pytree to a given
        dtype inplace. Only supports nested dicts.
    """
    if isinstance(tree, FrozenDict):
        raise ValueError('Only supports nested dicts!')
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)
    assert dtype in float_dtypes, 'dtype must be a float type!'
    for key, val in tree.items():
        if isinstance(val, (np.ndarray, jnp.ndarray)) and val.dtype in float_dtypes:
            if val.dtype != dtype:
                tree[key] = val.astype(dtype)
        elif isinstance(val, dict):
            inplace_float_to_dtype(val, dtype)
        elif isinstance(val, FrozenDict):
            raise ValueError('Only supports nested dicts!')


# basic dataset types

class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, index: Union[int, np.ndarray, jnp.ndarray]) -> Tuple[PyTree[jnp.ndarray], Union[List[Any], Any]]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

class IterableDataset(ABC):
    @abstractmethod
    def __iter__(self) -> Iterable[Tuple[PyTree[jnp.ndarray], Any]]:
        pass

# batch list / iterable

def _batch_idxs(prng_key: Optional[jax.random.KeyArray], data_size: int, bsize: int) -> np.ndarray:
    steps_per_epoch = data_size // bsize
    
    if prng_key is not None:
        with jax.default_device(jax.devices('cpu')[0]):
            permutations = np.asarray(jax.random.permutation(prng_key, data_size))
    else:
        permutations = np.arange(data_size)
    
    trunc_batch = permutations[steps_per_epoch * bsize:]
    
    permutations = permutations[:steps_per_epoch * bsize]
    permutations = permutations.reshape(steps_per_epoch, bsize)
    
    return permutations, trunc_batch

def _list_data_to_batch_iterator(
    prng_key: Optional[jax.random.KeyArray], 
    dataset: Any, 
    bsize: int, 
    postproc_f: Optional[Callable]=None, 
    truncate: bool=True
) -> Iterator:
    if postproc_f is None:
        postproc_f = lambda x: x
    
    batches, trunc_batch = _batch_idxs(prng_key, len(dataset), bsize)
    for idxs in batches:
        yield postproc_f(dataset[idxs])
    
    if not truncate and len(trunc_batch) > 0:
        yield postproc_f(dataset[trunc_batch])

def _iterable_data_to_batch_iterator(
    dataset: Any, 
    bsize: int, 
    postproc_f: Optional[Callable]=None, 
    truncate: bool=True, 
) -> Iterator:
    if postproc_f is None:
        postproc_f = lambda x: x
    
    batch = []
    for item in dataset:
        batch.append(item)
        if len(batch) == bsize:
            yield postproc_f(jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0) if isinstance(x[0], jnp.ndarray) else x, *batch))
            batch = []
    
    if not truncate and len(batch) > 0:
        yield postproc_f(jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0) if isinstance(x[0], jnp.ndarray) else x, *batch))

# convert dataset to dataloader

def dataloader(
    prng_key: Optional[jax.random.KeyArray], 
    dataset: Union[Dataset, IterableDataset], 
    bsize: int, 
    postproc_f: Optional[Callable]=None, truncate: bool=True
):
    if isinstance(dataset, Dataset):
        iterator = _list_data_to_batch_iterator(prng_key, dataset, bsize, postproc_f=postproc_f, truncate=truncate)
    elif isinstance(dataset, IterableDataset):
        iterator = _iterable_data_to_batch_iterator(dataset, bsize, postproc_f=postproc_f, truncate=truncate)
    else:
        raise NotImplementedError
    
    return iterator

def uuid_name(base_name: str, include_uuid: bool=True) -> str:
    """Generate a unique name based on the current time and a uuid"""
    if not include_uuid:
        return f"{base_name}.{str(datetime.utcnow().isoformat(sep='-', timespec='milliseconds')).replace(':', '-')}"
    return f"{base_name}.{str(datetime.utcnow().isoformat(sep='-', timespec='milliseconds')).replace(':', '-')}.{str(uuid.uuid1().hex)}"

def setup_experiment_save(
    exp_name: Optional[str], 
    outputs_path: Optional[str], 
    input_args: Dict[str, Any], 
    script__file__: Any, 
    is_main_process: bool=True, 
    add_unique_id: bool=True, 
) -> Tuple[Optional[str], str]:
    save_dir = None
    if exp_name is None:
        exp_name = uuid_name(base_name="exp", include_uuid=True)
    elif add_unique_id:
        exp_name = uuid_name(base_name=exp_name, include_uuid=True)
    if outputs_path is not None:
        save_dir = os.path.join(outputs_path, exp_name)
        if (not save_dir.startswith('gcs://')) and (not os.path.exists(save_dir)):
            os.makedirs(save_dir)
        
        # copy training script to outputs as a cheap form of config logging
        with open(script__file__, 'r') as f_local:
            with open(
                get_enabled_save_path(
                    os.path.join(save_dir, 'config.py'), 
                    enabled=is_main_process, 
                ), 'w') as f_save:
                f_save.write(f_local.read())
        with open(
            get_enabled_save_path(
                os.path.join(save_dir, 'input_args.pkl'), 
                enabled=is_main_process, 
            ), 'wb') as f:
            pkl.dump(input_args, f)
    return save_dir, exp_name

def strip_prompt_from_completion(
    prompt: str, 
    completion: str, 
) -> str:
    """Strip the prompt from the completion, handles special cases where bos tokens are not in the completion but in the prompt."""
    match = SequenceMatcher(None, prompt, completion, autojunk=False).find_longest_match(0, len(prompt), 0, len(completion))
    if match.a+match.size == len(prompt) and match.b == 0:
        return completion[match.size:]
    return completion

def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'checkpoint_dots': jax.checkpoint_policies.checkpoint_dots,
        'checkpoint_dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]
