from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, Dict, Iterator
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import freeze, FrozenDict
import jax
import itertools
import collections
from jaxtyping import PyTree

# pad sequence utilities

def pad_sequence(sequence: np.ndarray, max_len: int, pad_value: Any, pad_right: bool=True):
    assert sequence.shape[0] <= max_len, 'sequence has size larger than max_len'
    
    if isinstance(pad_value, np.ndarray):
        pad_tokens = np.full((max_len-sequence.shape[0],)+pad_value.shape, pad_value)
    else:
        pad_tokens = np.full((max_len-sequence.shape[0],), pad_value)
    
    if pad_right:
        return np.concatenate((sequence, pad_tokens))
    return np.concatenate((pad_tokens, sequence))    

def block_sequences(sequences: Union[List[List[int]], np.ndarray, List[np.ndarray]], 
                    max_len: Optional[int], pad_value: Any, dtype: Any, 
                    pad_right: bool=True, trunc_last: bool=True) -> np.ndarray:
    if max_len is None:
        max_len = max(map(lambda x: len(x), sequences))
    
    full_sequences = []
    for i in range(len(sequences)):
        if trunc_last:
            new_toks = sequences[i][:max_len]
        else:
            new_toks = sequences[i][-max_len:]
        full_sequences.append(pad_sequence(np.asarray(new_toks), max_len, pad_value, pad_right=pad_right))
    
    return np.asarray(full_sequences, dtype=dtype)

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

def _batch_idxs(rng: Optional[jax.random.KeyArray], data_size: int, bsize: int) -> np.ndarray:
    steps_per_epoch = data_size // bsize
    
    if rng is not None:
        permutations = np.asarray(jax.random.permutation(rng, data_size))
    else:
        permutations = np.arange(data_size)
    
    trunc_batch = permutations[steps_per_epoch * bsize:]
    
    permutations = permutations[:steps_per_epoch * bsize]
    permutations = permutations.reshape(steps_per_epoch, bsize)
    
    return permutations, trunc_batch

def _list_data_to_batch_iterator(rng: Optional[jax.random.KeyArray], dataset: Any, bsize: int, 
                                 postproc_f: Optional[Callable]=None, truncate: bool=True) -> Iterator:
    if postproc_f is None:
        postproc_f = lambda x: x
    
    batches, trunc_batch = _batch_idxs(rng, len(dataset), bsize)
    for idxs in batches:
        yield postproc_f(dataset[idxs])
    
    if not truncate and len(trunc_batch) > 0:
        yield postproc_f(dataset[trunc_batch])

def _iterable_data_to_batch_iterator(dataset: Any, bsize: int, 
                                     postproc_f: Optional[Callable]=None, truncate: bool=True) -> Iterator:
    if postproc_f is None:
        postproc_f = lambda x: x
    
    batch = []
    for item in dataset:
        batch.append(item)
        if len(batch) == bsize:
            yield postproc_f(jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0) if isinstance(x[0], jnp.ndarray) else x, *batch),)
            batch = []
    
    if not truncate and len(batch) > 0:
        yield postproc_f(jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0) if isinstance(x[0], jnp.ndarray) else x, *batch),)

# pretetching operation for device data transfer optimization

def _prefetch(iterator: Iterator, queue_size: int = 2) -> Iterator:
    # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
    # queue_size = 2 should be ok for one GPU.

    queue = collections.deque()

    def enqueue(n):
        for item in itertools.islice(iterator, n):
            queue.append(item)

    enqueue(queue_size)
    while queue:
        yield queue.popleft()
        enqueue(1)

# convert dataset to dataloader

def dataloader(rng: Optional[jax.random.KeyArray], 
               dataset: Union[Dataset, IterableDataset], 
               bsize: int, prefetch_batches: Optional[int]=None, 
               postproc_f: Optional[Callable]=None, truncate: bool=True):
    
    if isinstance(dataset, Dataset):
        iterator = _list_data_to_batch_iterator(rng, dataset, bsize, postproc_f=postproc_f, truncate=truncate)
    elif isinstance(dataset, IterableDataset):
        iterator = _iterable_data_to_batch_iterator(dataset, bsize, postproc_f=postproc_f, truncate=truncate)
    else:
        raise NotImplementedError
    
    if prefetch_batches is not None:
        iterator = _prefetch(iterator, prefetch_batches)
    
    return iterator
