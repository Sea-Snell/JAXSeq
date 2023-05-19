import os
import tyro
from JaxSeq.checkpointing import load_pytree, save_pytree
from JaxSeq.utils import convert_path

def main(
    load_dir: str, 
    /,  # Mark the end of positional arguments.
):
    params = load_pytree(os.path.join(convert_path(load_dir), 'train_state.msgpack'))['params']
    save_pytree(params, os.path.join(convert_path(load_dir), 'params.msgpack'))

if __name__ == "__main__":
    tyro.cli(main)
