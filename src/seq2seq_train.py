import contextlib
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from jaxtyping import PyTree
from jax.random import KeyArray
from jax.experimental.maps import Mesh
from collections import deque
import jax
from tqdm.auto import tqdm
from data import Dataset, dataloader
from seq2seq_data import Seq2SeqDataset, Seq2SeqIterableDataset
from seq2seq import Seq2SeqTrainer, Seq2SeqInference
from logs import combine_logs, label_logs, log, pull_logs
import os
from transformers.modeling_flax_utils import FlaxPreTrainedModel
import wandb
import tempfile
import gcsfs

def save_checkpoint_path(model_output_path: str, model: FlaxPreTrainedModel, 
                         params: PyTree, gcloud_project: Optional[str]=None) -> None:
    if model_output_path.startswith('gcs://'):
        model_output_path = model_output_path[len('gcs://'):]
        # save to tmp_dir
        tmp_dir = tempfile.TemporaryDirectory()
        model.save_pretrained(
            tmp_dir, 
            params=params, 
        )
        # upload to gcloud bucket
        gcsfs.GCSFileSystem(project=gcloud_project).put(tmp_dir.name, model_output_path, recursive=True)
        # delete temp_dir
        tmp_dir.cleanup()
    else:
        model.save_pretrained(
            model_output_path, 
            params=params, 
        )

def delete_checkpoint(checkpoint_path: str, gcloud_project: Optional[str]=None) -> None:
    if checkpoint_path.startswith('gcs://'):
        checkpoint_path = checkpoint_path[len('gcs://'):]
        gcsfs.GCSFileSystem(project=gcloud_project).rm(checkpoint_path, recursive=True)
    else:
        os.system('rm -rf %s' % (checkpoint_path))

def eval_loss(
    inference: Seq2SeqInference, 
    dataset: Union[Seq2SeqDataset, Seq2SeqIterableDataset], 
    rng: Optional[KeyArray], 
    bsize: int, 
    eval_batches: Optional[int], 
    prefetch_batches: Optional[int]=None, 
):
    # setup evaluator loop state
    eval_logs = []

    # eval on batches
    rng, new_rng = jax.random.split(rng) if rng is not None else (None, None)
    d = dataloader(new_rng, dataset, bsize, prefetch_batches=prefetch_batches, truncate=True)
    for i, items in enumerate(tqdm(d)):
        
        # conditionally terminate early
        if eval_batches is not None and i >= eval_batches:
            break

        # get eval logs
        _, info = inference.eval_loss(items)
        eval_logs.append(info)
    
    # gather and postproc eval logs
    eval_logs = pull_logs(combine_logs(eval_logs))

    return eval_logs

def train_loop(
    model: FlaxPreTrainedModel, 
    trainer: Seq2SeqTrainer, 
    inference: Seq2SeqInference, 
    evaluator: Optional[Callable[[Seq2SeqInference], Tuple[float, Dict[str, Any]]]], 
    dataset: Union[Seq2SeqDataset, Seq2SeqIterableDataset], 
    rng: KeyArray, 
    save_dir: Optional[str], 
    epochs: int, 
    max_steps: Optional[int], 
    bsize: int, 
    log_every: int, 
    eval_every: int, 
    save_every: Optional[int], 
    save_at_end: bool, 
    save_best: bool, 
    max_checkpoints: Optional[int], 
    use_wandb: bool, 
    wandb_project: Optional[str], 
    wandb_run_name: Optional[str], 
    wandb_config: Optional[Dict[str, Any]], 
    prefetch_batches: Optional[int]=None, 
    gcloud_project: Optional[str]=None, 
) -> Tuple[Seq2SeqTrainer, Seq2SeqInference]:
    assert (not use_wandb) or (use_wandb and wandb_project is not None)
    
    # initalize wandb
    if use_wandb and jax.process_index() == 0:
        wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config, reinit=True)

    # initalize training loop state
    train_logs = []
    best_perf = float('inf')
    saved_checkpoints = deque([])
    step = 0
    steps_per_epoch = len(dataset) // bsize if isinstance(dataset, Dataset) else None

    # begin training loop
    for epoch in tqdm(range(epochs), disable=jax.process_index() > 0):
        rng, new_rng = jax.random.split(rng)
        d = dataloader(new_rng, dataset, bsize, prefetch_batches=prefetch_batches, truncate=True)
        for items in tqdm(d, total=steps_per_epoch, disable=jax.process_index() > 0):
            
            # step model and get training logs
            rng, new_rng = jax.random.split(rng)
            _, info, trainer = trainer.train_step(items, new_rng)
            train_logs.append(info)
            
            # publish training logs and clear logs
            if (step + 1) % log_every == 0:
                logs = combine_logs(train_logs)
                logs = pull_logs(label_logs(logs, 'train', {'step': step+1, 'epoch': epoch}))
                if jax.process_index() == 0:
                    log(logs, use_wandb)
                train_logs = []
            
            # begin evaluation
            if (evaluator is not None) and (step + 1) % eval_every == 0:

                # get eval logs
                inference = inference.set_params(trainer.params)
                eval_perf, eval_logs = evaluator(inference)

                # publish eval logs
                eval_logs = pull_logs(label_logs(eval_logs, 'eval', {'step': step+1, 'epoch': epoch}))
                if jax.process_index() == 0:
                    log(eval_logs, use_wandb)

                # conditionally save best model and optimizer state
                if save_dir is not None and save_best and eval_perf < best_perf:
                    print('new best model! Saving ...')
                    model_dir = os.path.join(save_dir, 'model')
                    save_checkpoint_path(
                        model_output_path=model_dir, 
                        model=model, 
                        params=jax.device_get(trainer.params), 
                        gcloud_project=gcloud_project, 
                    )
                    print('saved.')
                    best_perf = eval_perf
            
            # periodically save checkpoint
            if save_dir is not None and save_every is not None and (step + 1) % save_every == 0:
                print('saving checkpoint...')

                # conditionally delete old checkpoints
                if (max_checkpoints is not None) and (len(saved_checkpoints) >= max_checkpoints):
                    delete_checkpoint(saved_checkpoints.popleft(), gcloud_project=gcloud_project)

                model_dir = os.path.join(save_dir, 'model_%d' % (step+1))
                save_checkpoint_path(
                    model_output_path=model_dir, 
                    model=model, 
                    params=jax.device_get(trainer.params), 
                    gcloud_project=gcloud_project, 
                )
                saved_checkpoints.append(model_dir)
                print('saved.')
            
            # conditionally terminate
            if max_steps is not None and (step + 1) >= max_steps:
                break

            step += 1
        
        # conditionally terminate
        if max_steps is not None and (step + 1) >= max_steps:
            break
    
    # save final checkpoint
    if save_dir is not None and save_at_end:
        print('saving checkpoint...')
        model_dir = os.path.join(save_dir, 'model_%d' % (step+1))
        save_checkpoint_path(
            model_output_path=model_dir, 
            model=model, 
            params=jax.device_get(trainer.params), 
            gcloud_project=gcloud_project, 
        )
        print('saved.')

    # stop wandb
    if use_wandb and jax.process_index() == 0:
        wandb_run.finish()
    
    inference = inference.set_params(trainer.params)
    return trainer, inference
