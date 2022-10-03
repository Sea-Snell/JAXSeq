# minimalist local flax huggingface model parameter loading.
# Adapted from: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_flax_utils.py

import gc
from typing import Union
import os
import jax
import jax.numpy as jnp
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
from pickle import UnpicklingError
import msgpack.exceptions
try:
    from transformers.utils import (
        FLAX_WEIGHTS_NAME,
        FLAX_WEIGHTS_INDEX_NAME,
    )
except:
    from transformers.utils import (
        FLAX_WEIGHTS_NAME,
    )
    FLAX_WEIGHTS_INDEX_NAME = None

def load_flax_sharded_weights(cls, shard_files):
    """
    This is the same as [`flax.serialization.from_bytes`]
    (https:lax.readthedocs.io/en/latest/_modules/flax/serialization.html#from_bytes) but for a sharded checkpoint.
    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.
    Args:
        shard_files (`List[str]`:
            The list of shard files to load.
    Returns:
        `Dict`: A nested dictionary of the model parameters, in the expected format for flax models : `{'model':
        {'params': {'...'}}}`.
    """

    # Load the index
    state_sharded_dict = dict()

    for shard_file in shard_files:
        # load using msgpack utils
        try:
            with open(shard_file, "rb") as state_f:
                state = from_bytes(cls, state_f.read())
        except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
            with open(shard_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please"
                        " install git-lfs and run `git lfs install` followed by `git lfs pull` in the"
                        " folder you cloned."
                    )
                else:
                    raise ValueError from e
        except (UnicodeDecodeError, ValueError):
            raise EnvironmentError(f"Unable to convert {shard_file} to Flax deserializable object. ")

        state = flatten_dict(state, sep="/")
        state_sharded_dict.update(state)
        del state
        gc.collect()

    # the state dict is unflattened to the match the format of model.params
    return unflatten_dict(state_sharded_dict, sep="/")

def from_path(
    cls, 
    model_path: Union[str, os.PathLike], 
):
    is_sharded=False
    if os.path.isfile(os.path.join(model_path, FLAX_WEIGHTS_NAME)):
        # Load from a Flax checkpoint
        archive_file = os.path.join(model_path, FLAX_WEIGHTS_NAME)
    elif os.path.isfile(os.path.join(model_path, FLAX_WEIGHTS_INDEX_NAME)):
        # Load from a sharded Flax checkpoint
        archive_file = os.path.join(model_path, FLAX_WEIGHTS_INDEX_NAME)
        is_sharded = True
    # At this stage we don't have a weight file so we will raise an error.
    else:
        raise EnvironmentError(
            f"Error no file named {FLAX_WEIGHTS_NAME} found in directory "
            f"{model_path}."
        )
    if is_sharded:
        state = load_flax_sharded_weights(cls, archive_file)
    else:
        try:
            with open(archive_file, "rb") as state_f:
                state = from_bytes(cls, state_f.read())
        except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
            try:
                with open(archive_file) as f:
                    if f.read().startswith("version"):
                        raise OSError(
                            "You seem to have cloned a repository without having git-lfs installed. Please"
                            " install git-lfs and run `git lfs install` followed by `git lfs pull` in the"
                            " folder you cloned."
                        )
                    else:
                        raise ValueError from e
            except (UnicodeDecodeError, ValueError):
                raise EnvironmentError(f"Unable to convert {archive_file} to Flax deserializable object. ")
    state = jax.tree_util.tree_map(jnp.array, state)
    return state
