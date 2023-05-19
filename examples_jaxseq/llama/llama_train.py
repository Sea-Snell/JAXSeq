from typing import Optional, Dict, Any, Tuple
import tyro
from JaxSeq.bucket_manager import open_with_bucket as open
from transformers import AutoTokenizer
from JaxSeq.utils import jsonl_stream, convert_path, load_mesh, get_dtype, setup_experiment_save
import jax
import jax.numpy as jnp
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, uuid_name, jsonl_load, get_weight_decay_mask, create_path, get_enabled_save_path
import os
import optax
from JaxSeq.models.llama.interface import LLaMATrain, LLaMAInference
from JaxSeq.models.llama.load import load_train_state, ModelLoadMode, load_tokenizer
from JaxSeq.models.llama.tokenizer import LLaMATokenizer
import pickle as pkl
from JaxSeq.data import Seq2SeqDataset
from JaxSeq.train import eval_loss, train_loop
from JaxSeq.generation_eval import generate_language, compute_metrics
from transformers.generation import GenerationConfig
from jaxtyping import PyTree
import re
import tempfile
import json
from datetime import datetime

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
    
    lr: float=1e-5, 
    weight_decay: float=0.0, 

    train_bsize: int=16, 
    grad_accum_steps: int=1, 

    gradient_checkpointing: bool=False, 
    gradient_checkpointing_policy: str='nothing_saveable', 

    max_input_length: int=512, 
    max_output_length: int=512, 

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

    generation_bsize: int=32, 
    generation_batches: Optional[int]=None, 
    generation_do_sample: bool=True, 
    generation_num_beams: int=1, 

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
    tokenizer.pad_token_id = tokenizer.unk_token_id # set pad token to unk_token
    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})

    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    is_main_process = jax.process_index() == 0
    print(f"Mesh: {mesh}")
    print(f"Is main process: {is_main_process}")

    # load data
    with open(convert_path(train_data_path), 'r') as f:
        train_json_data = jsonl_load(f)
    with open(convert_path(eval_data_path), 'r') as f:
        eval_json_data = jsonl_load(f)

    train_data = Seq2SeqDataset.from_str_list(
        list(map(lambda x: (tokenizer.bos_token+x['in_text'].removeprefix(tokenizer.bos_token), x['out_text']), train_json_data)), 
        tokenizer, 
        in_blocking_strategy=BlockingStrategy(
            padding=Padding.LEFT, 
            truncation=Truncation.LEFT, 
            max_length=max_input_length
        ), 
        out_blocking_strategy=BlockingStrategy(
            padding=Padding.RIGHT, 
            truncation=Truncation.RIGHT, 
            max_length=max_output_length
        ), 
    )

    eval_data = Seq2SeqDataset.from_str_list(
        list(map(lambda x: (tokenizer.bos_token+x['in_text'].removeprefix(tokenizer.bos_token), x['out_text']), eval_json_data)), 
        tokenizer, 
        in_blocking_strategy=BlockingStrategy(
            padding=Padding.LEFT, 
            truncation=Truncation.LEFT,
            max_length=max_input_length
        ), 
        out_blocking_strategy=BlockingStrategy(
            padding=Padding.RIGHT, 
            truncation=Truncation.RIGHT, 
            max_length=max_output_length
        ), 
    )

    def optim_getter(params: PyTree):
        mask = get_weight_decay_mask((
            re.escape("['attention_norm']['kernel']"), 
            re.escape("['ffn_norm']['kernel']"), 
            re.escape("['transformer']['ln_f']['kernel']"), 
        ))(params)
        return optax.MultiSteps(
            optax.adamw(
                learning_rate=lr, 
                b1=0.9, 
                b2=0.999, 
                eps=1e-6, 
                weight_decay=weight_decay, 
                mask=mask, 
            ), 
            every_k_schedule=grad_accum_steps, 
        )

    model_prng_key = jax.random.PRNGKey(2)
    train_state, model = load_train_state(
        model_load_mode=model_load_mode, 
        model_load_path=convert_path(model_load_path) if model_load_mode != ModelLoadMode.HF else model_load_path, 
        model_dtype=jnp.float32, 
        optim_getter=optim_getter, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=model_prng_key, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=jnp.float32, 
    )
    model.config.gradient_checkpointing = gradient_checkpointing
    model.config.gradient_checkpointing_policy = gradient_checkpointing_policy

    loop_state = dict()
    if should_restore_loop_state and (model_load_mode in {ModelLoadMode.TRAIN_STATE, 
                                                          ModelLoadMode.TRAIN_STATE_PARAMS, 
                                                          ModelLoadMode.PARAMS}):
        with open(os.path.join(convert_path(model_load_path), 'loop_state.pkl'), 'rb') as f:
            loop_state = pkl.load(f)
    
    trainer = LLaMATrain.load_train(
        train_state=train_state, 
        model=model, 
        tokenizer=tokenizer, 
    )

    inference = LLaMAInference.load_inference(
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

    eval_prng = jax.random.PRNGKey(0)
    def evaluator(inference: LLaMAInference):
        nonlocal eval_prng

        loss_metrics = eval_loss(
            inference=inference, 
            dataset=eval_data, 
            prng_key=None, 
            bsize=eval_loss_bsize, 
            eval_batches=eval_loss_batches, 
        )

        eval_prng, new_prng = jax.random.split(eval_prng)
        generation_prompts = [
            eval_json_data[i] for i in jax.random.permutation(
                new_prng, 
                jnp.arange(len(eval_json_data)), 
            ).tolist()
        ]
        eval_prng, new_prng = jax.random.split(eval_prng)
        generation_data = generate_language(
            inference=inference, 
            prompts=list(map(lambda x: tokenizer.bos_token+x['in_text'].removeprefix(tokenizer.bos_token), generation_prompts)), 
            references=list(map(lambda x: [x['out_text']], generation_prompts)), 
            prng_key=new_prng, 
            bsize=generation_bsize, 
            generation_batches=generation_batches, 
            blocking_strategy=BlockingStrategy(
                padding=Padding.LEFT, 
                truncation=Truncation.LEFT, 
                max_length=max_input_length
            ), 
            generation_config=GenerationConfig(
                max_length=max_input_length+max_output_length, 
                do_sample=generation_do_sample, 
                num_beams=generation_num_beams, 
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id, 
                temperature=1.0, 
                top_k=None, 
                top_p=None, 
            ), 
        )

        if save_dir is not None:
            generations_save_dir = os.path.join(save_dir, 'generations')
            if is_main_process:
                create_path(generations_save_dir)
            with open(get_enabled_save_path(
                os.path.join(generations_save_dir, uuid_name('generations', include_uuid=False)+'.json'), 
                enabled=is_main_process, 
            ), 'w') as f:
                json.dump(generation_data, f)
        
        reference_metrics = compute_metrics(generation_data)

        return loss_metrics['loss'], {'loss_metrics': loss_metrics, 'reference_metrics': reference_metrics}
    
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
