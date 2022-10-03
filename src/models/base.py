from typing import Any, NamedTuple, Optional, Tuple
import jax.numpy as jnp
import jax
from jaxtyping import PyTree
from transformers.modeling_flax_utils import FlaxPreTrainedModel
import tempfile
import gcsfs

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

def handle_checkpoint_path(model_checkpoint_path: str, gcloud_project: Optional[str]=None) -> Tuple[str, tempfile.TemporaryDirectory]:
    if model_checkpoint_path.startswith('gcs://'):
        model_checkpoint_path = model_checkpoint_path[len('gcs://'):]
        tmp_dir = tempfile.TemporaryDirectory()
        # download data
        gcsfs.GCSFileSystem(project=gcloud_project).get(model_checkpoint_path, tmp_dir.name, recursive=True)
        return tmp_dir.name, tmp_dir
    return model_checkpoint_path, None


