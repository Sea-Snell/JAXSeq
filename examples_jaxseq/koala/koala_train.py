from typing import Optional, Dict, Any, Tuple
import tyro
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import convert_path, load_mesh, setup_experiment_save, get_enabled_save_path, MapIterable, FileOpenIterable
import jax
import jax.numpy as jnp
from JaxSeq.utils import jsonl_load, get_weight_decay_mask, jsonl_stream
import os
import optax
from JaxSeq.models.llama.interface import LLaMATrainMask, LLaMAInferenceMask
from JaxSeq.models.llama.load import load_train_state, ModelLoadMode, load_tokenizer
import pickle as pkl
from JaxSeq.data import MaskDataset, MaskIterableDataset
from JaxSeq.train import eval_loss, train_loop
from jaxtyping import PyTree
import re
from JaxSeq.optimizers import GPT3Optimizer

def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str, 
    tokenizer_path: str, 
    train_data_path: str, 
    eval_data_path: str, 

    /,  # Mark the end of positional arguments.

    exp_name: Optional[str]=None, 
    outputs_path: Optional[str]=None, 

    data_mesh_shape: int=1, 
    fsdp_mesh_shape: int=1, 
    model_mesh_shape: int=-1, 

    use_wandb: bool=False, 
    wandb_project: Optional[str]=None, 

    epochs: int=1, 
    max_steps: Optional[int]=None, 

    weight_decay: float=0.001, 
    init_lr: float=0.0, 
    end_lr: float=0.002, 
    lr: float=0.002, 
    lr_warmup_steps: int=1000, 
    lr_decay_steps: int=1001, # no decay, so just needs to be > warmup steps
    bf16_momentum: bool=False, 
    multiply_by_parameter_scale: bool=True, 

    resid_pdrop: float=0.05, 
    attn_pdrop: float=0.05, 
    embd_pdrop: float=0.05, 

    train_bsize: int=16, 
    grad_accum_steps: Optional[int]=1, 

    gradient_checkpointing: bool=False, 
    gradient_checkpointing_policy: str='nothing_saveable', 
    bf16_activations: bool=False, 

    max_length: int=2048, 

    log_every: int=256, 
    eval_every_steps: Optional[int]=256, 
    eval_every_epochs: Optional[int]=None, 
    eval_at_beginning: bool=False, 
    eval_at_end: bool=True, 
    
    save_every_steps: Optional[int]=None, 
    save_every_epochs: Optional[int]=None, 
    save_at_beginning: bool=False, 
    save_at_end: bool=False, 
    save_best: bool=True, 
    max_checkpoints: Optional[int]=None, 
    save_train_state: bool=True, 
    save_bf16: bool=True, 

    eval_loss_bsize: int=32, 
    eval_loss_batches: Optional[int]=None, 

    force_pad_embeddings: bool=False, 

    should_restore_loop_state: bool=False, 
):
    input_args = locals()
    print(input_args)

    tokenizer = load_tokenizer(
        tokenizer_path, 
        bos_token="<s>", 
        eos_token="</s>", 
        add_bos_token=False, 
        add_eos_token=False, 
    )
    tokenizer.pad_token_id = None # set pad token to None during training
    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})

    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    is_main_process = jax.process_index() == 0
    print(f"Mesh: {mesh}")
    print(f"Is main process: {is_main_process}")

    # load data

    train_data = MaskIterableDataset.packed_from_str_segments_iterable(
        MapIterable(lambda x: x+[(tokenizer.eos_token, 1.0)], FileOpenIterable(convert_path(train_data_path), 'r', pipe=jsonl_stream)), 
        tokenizer, 
        max_length=max_length, 
        truncate=True, 
        buffer_start_str=tokenizer.bos_token, 
    )

    eval_data = MaskIterableDataset.packed_from_str_segments_iterable(
        MapIterable(lambda x: x+[(tokenizer.eos_token, 1.0)], FileOpenIterable(convert_path(eval_data_path), 'r', pipe=jsonl_stream)), 
        tokenizer, 
        max_length=max_length, 
        truncate=True, 
        buffer_start_str=tokenizer.bos_token, 
    )

    def optim_getter(params: PyTree):
        mask = get_weight_decay_mask((
            re.escape("['attention_norm']['kernel']"), 
            re.escape("['ffn_norm']['kernel']"), 
            re.escape("['transformer']['ln_f']['kernel']"), 
        ))(params)

        optimizer_config = GPT3Optimizer(
            init_lr=init_lr, 
            end_lr=end_lr, 
            lr=lr, 
            lr_warmup_steps=lr_warmup_steps, 
            lr_decay_steps=lr_decay_steps, 
            weight_decay=weight_decay, 
            bf16_momentum=bf16_momentum, 
            multiply_by_parameter_scale=multiply_by_parameter_scale, 
        )

        optim, _ = optimizer_config.get_optim(mask)

        if grad_accum_steps is not None:
            return optax.MultiSteps(optim, every_k_schedule=grad_accum_steps)
        return optim

    model_prng_key = jax.random.PRNGKey(2)
    train_state, model = load_train_state(
        model_load_mode=model_load_mode, 
        model_load_path=convert_path(model_load_path) if model_load_mode != ModelLoadMode.HF else model_load_path, 
        model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32, 
        optim_getter=optim_getter, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=model_prng_key, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=jnp.float32, 
    )
    model.config.gradient_checkpointing = gradient_checkpointing
    model.config.gradient_checkpointing_policy = gradient_checkpointing_policy
    model.config.resid_pdrop = resid_pdrop
    model.config.embd_pdrop = embd_pdrop
    model.config.attn_pdrop = attn_pdrop

    loop_state = dict()
    if should_restore_loop_state and (model_load_mode in {ModelLoadMode.TRAIN_STATE, 
                                                          ModelLoadMode.TRAIN_STATE_PARAMS, 
                                                          ModelLoadMode.PARAMS}):
        with open(os.path.join(convert_path(model_load_path), 'loop_state.pkl'), 'rb') as f:
            loop_state = pkl.load(f)
    
    trainer = LLaMATrainMask.load_train(
        train_state=train_state, 
        model=model, 
        tokenizer=tokenizer, 
    )

    inference = LLaMAInferenceMask.load_inference(
        params=train_state.params, 
        model=model, 
        tokenizer=tokenizer, 
    )

    save_dir, exp_name = setup_experiment_save(
        exp_name=exp_name, 
        outputs_path=convert_path(outputs_path), 
        input_args=input_args, 
        script__file__=__file__, 
        is_main_process=is_main_process, 
    )

    def evaluator(inference: LLaMAInferenceMask):
        loss_metrics = eval_loss(
            inference=inference, 
            dataset=eval_data, 
            prng_key=None, 
            bsize=eval_loss_bsize, 
            eval_batches=eval_loss_batches, 
        )

        return loss_metrics['loss'], loss_metrics
    
    train_prng = jax.random.PRNGKey(1)
    save_dtype = jnp.bfloat16 if save_bf16 else jnp.float32
    trainer, inference = train_loop(
        trainer=trainer, 
        inference=inference, 
        evaluator=evaluator, 
        dataset=train_data, 
        prng_key=train_prng, 
        save_dir=save_dir, 
        epochs=epochs, 
        max_steps=max_steps, 
        bsize=train_bsize, 
        log_every=log_every, 
        eval_every_steps=eval_every_steps, 
        eval_every_epochs=eval_every_epochs, 
        eval_at_beginning=eval_at_beginning, 
        eval_at_end=eval_at_end, 
        save_every_steps=save_every_steps, 
        save_every_epochs=save_every_epochs, 
        save_at_beginning=save_at_beginning, 
        save_at_end=save_at_end, 
        save_best=save_best, 
        max_checkpoints=max_checkpoints, 
        save_train_state=save_train_state, 
        save_dtype=save_dtype, 
        use_wandb=use_wandb, 
        wandb_project=wandb_project, 
        wandb_run_name=exp_name, 
        wandb_config=None, 
        is_main_process=is_main_process, 
        **loop_state, 
    )

if __name__ == "__main__":
    tyro.cli(main)
