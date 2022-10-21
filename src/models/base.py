from typing import Any, NamedTuple, Optional, Tuple, Union
import jax.numpy as jnp
import jax
from jaxtyping import PyTree
from transformers.modeling_flax_utils import FlaxPreTrainedModel
import tempfile
import gcsfs
import pickle as pkl

from utils.gcs_manager import open_pp

class HuggingfacePjitModelDescription(NamedTuple):
    model: FlaxPreTrainedModel
    params: PyTree
    shard_rules: Any

def get_dtype(use_fp16: bool):
    if use_fp16:
        if jax.default_backend() == 'tpu':
            return jnp.bfloat16
        return jnp.float16
    return jnp.float32

def handle_checkpoint(model_checkpoint_path: str, gcloud_project: Optional[str]=None, 
                           gcloud_token: Optional[Any]=None) -> PyTree:
    with open_pp(model_checkpoint_path, 'rb', gcloud_project=gcloud_project, gcloud_token=gcloud_token) as f:
        params = pkl.load(f)
    return params
