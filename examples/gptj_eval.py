import random
from typing import Any, Optional
from transformers import AutoTokenizer
from models.gptj import load_gptj_model
import jax
import optax
from seq2seq import Seq2SeqInference, load_gpt_dec_inference
from seq2seq_data import Seq2SeqDataset
from utils.path import convert_path
import json
import contextlib
import numpy as np
from jax.experimental.maps import Mesh
from shard import shard_data_iterable, shard_data_list, shard_optim_and_params, OptimType, shard_params
from functools import partial
from seq2seq_train import train_loop, eval_loss
from evaluate import generate_language, compute_metrics
import os
import pickle as pkl
import tree
import tyro
from utils.multihost_shard_utils import convert_bsize, get_mesh_idxs, get_mesh_lens

def main(
    model_name: str, # EleutherAI/gpt-j-6B [6.5B]
    data_json_path: str, # should be dict of shape {'train': [{'in_text', 'out_text'}, ...], 'eval': [{'in_text', 'out_text'}, ...]}
    
    /,  # Mark the end of positional arguments.

    checkpoint_path: Optional[str]=None, 

    do_pjit: bool=True, 
    model_p_shape: int=1, 
    data_p_shape: int=1, 

    eval_batches: Optional[int]=None, 

    gradient_checkpoint: bool=True, 

    max_input_length: int=512, 
    max_output_length: int=512, 
    
    trunc_inputs_last: bool=True, 
    trunc_outputs_last: bool=True, 

    inference_bsize: int=32, 
    inference_do_sample: bool=True, 

    gcloud_project: Optional[str]=None, 
    gcloud_token_path: Optional[str]=None, 
):
    input_args = locals()
    print(input_args)

    from utils.gcs_manager import open_pp as open
    open = partial(open, gcloud_project=gcloud_project, gcloud_token=gcloud_token_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    # mesh definition
    if do_pjit:
        mesh_devices = np.array(jax.devices()).reshape(data_p_shape, model_p_shape)
        print('using mesh shape:', mesh_devices.shape)
        print('full mesh:', mesh_devices)
        mesh = Mesh(mesh_devices, ("dp", "mp"))
        process_idxs = get_mesh_idxs(jax.process_index(), mesh.devices)
        process_shape = get_mesh_lens(mesh.devices)
        print(f'current process index {jax.process_index()}, in position {process_idxs} of {process_shape}')
    else:
        mesh = contextlib.nullcontext()
    
    # divide bsize by number of data-parallel nodes
    if do_pjit:
        inference_bsize = convert_bsize(inference_bsize, mesh.devices, 0)

    with open(convert_path(data_json_path), 'r') as f:
        raw_data = json.load(f)
    
    raw_eval_data = raw_data['eval']
    
    # shard data by number of data parallel processes
    if do_pjit:
        raw_eval_data = shard_data_list(raw_eval_data, mesh, 0)

    eval_data = Seq2SeqDataset.from_str_list(
        list(map(lambda x: (x['in_text'], x['out_text']), raw_eval_data)), 
        tokenizer, 
        max_input_length=max_input_length, 
        max_output_length=max_output_length, 
        pad_inputs_right=False, 
        trunc_inputs_last=trunc_inputs_last, 
        trunc_outputs_last=trunc_outputs_last, 
    )

    model, params, shard_rules = load_gptj_model(
        model_str=model_name, 
        from_pretrained=True, 
        checkpoint_path=checkpoint_path, 
        use_fp16=jax.default_backend() == 'tpu', 
        tokenizer=tokenizer, 
        gradient_checkpoint=gradient_checkpoint, 
        seed=0, 
        gcloud_project=gcloud_project, 
        gcloud_token=gcloud_token_path, 
    )

    # shard params and optimizer
    if do_pjit:
        params, param_spec = shard_params(partial(model.init_weights, input_shape=(1, 1)), 
                                                  params, shard_rules, mesh, 1)
    else:
        param_spec = None

    inference = load_gpt_dec_inference(
        model=model, 
        params=params, 
        param_spec=param_spec, 
        tokenizer=tokenizer, 
        do_pjit=do_pjit, 
    )

    def evaluator(inference: Seq2SeqInference):
        rng = jax.random.PRNGKey(0)
        
        rng, new_rng = jax.random.split(rng)
        loss_metrics = eval_loss(
            inference=inference, 
            dataset=eval_data, 
            rng=new_rng, 
            bsize=inference_bsize, 
            eval_batches=eval_batches, 
        )

        rng, new_rng = jax.random.split(rng)
        generation_prompts = list(raw_eval_data)
        # random.shuffle(generation_prompts)
        generation_data = generate_language(
            inference=inference, 
            prompts=list(map(lambda x: x['in_text'], generation_prompts)), 
            references=list(map(lambda x: [x['out_text']], generation_prompts)), 
            rng=new_rng, 
            bsize=inference_bsize, 
            eval_batches=eval_batches, 
            max_input_length=max_input_length, 
            in_str_preproc=None, 
            out_str_postproc=None, 
            max_length=max_input_length+max_output_length, 
            do_sample=inference_do_sample, 
            num_beams=1, 
            pad_token_id=tokenizer.pad_token_id, 
            eos_token_id=tokenizer.eos_token_id, 
        )
        # print('\n=====\n=====\n'.join(random.sample(list(map(lambda x: str((x['prompt'], x['generation'],)), generation_data)), 10)))
        reference_metrics = compute_metrics(generation_data)

        return {'loss_metrics': loss_metrics, 'reference_metrics': reference_metrics}
    
    with mesh:
        print(evaluator(inference))

if __name__ == "__main__":
    tyro.cli(main)
