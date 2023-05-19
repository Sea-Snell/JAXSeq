import tyro
from JaxSeq.models.llama.load import load_params, ModelLoadMode, load_tokenizer
import jax.numpy as jnp
from JaxSeq.utils import load_mesh, convert_path
from JaxSeq.shard_model import get_sharding_from_model
from JaxSeq.checkpointing import save_pytree
import os
from JaxSeq.bucket_manager import open_with_bucket as open

def main(
    model_path: str, 
    tokenizer_path: str, 
    output_dir: str, 
    /,  # Mark the end of positional arguments.
):
    tokenizer = load_tokenizer(
        tokenizer_path, 
        bos_token="<s>", 
        eos_token="</s>", 
        add_bos_token=False, 
        add_eos_token=False, 
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id # set pad token to unk_token
    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})

    mesh = load_mesh((1, 1, -1), ('dp', 'fsdp', 'mp'))
    print(f"Mesh: {mesh}")

    params, model = load_params(
        model_load_mode=ModelLoadMode.OFFICIAL_WEIGHTS, 
        model_load_path=convert_path(model_path), 
        model_dtype=jnp.float32, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        params_dtype=jnp.float16, 
    )

    # dump model config
    with open(convert_path(os.path.join(output_dir, 'config.json')), 'w') as f:
        f.write(model.config.to_json_string())
    # dump params
    save_pytree(
        tree=params, 
        path=convert_path(os.path.join(output_dir, 'params.msgpack')), 
        sharding=get_sharding_from_model(model, params), 
    )

if __name__ == "__main__":
    tyro.cli(main)
