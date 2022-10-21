from __future__ import annotations
from collections import namedtuple
from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Protocol
import jax
from flax.core.frozen_dict import freeze
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental.maps import Mesh
import numpy as np
from jax.random import KeyArray
from optax import softmax_cross_entropy_with_integer_labels
from flax.core.frozen_dict import FrozenDict
import optax
from jaxtyping import PyTree
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from jax.experimental.pjit import pjit, with_sharding_constraint
from flax import struct
from jax.experimental import PartitionSpec
from core import LogProbsOutput, Trainer, Inference, TrainStepOutput
from data import block_sequences
from models.opt import force_bos_to_start

# sseq2seq trainer

class Seq2SeqTrainer(Trainer):
    def train_step_from_str(
        self, 
        in_strs: List[str], 
        out_strs: List[str], 
        max_input_length: int, 
        max_output_length: int, 
        rng_key: KeyArray, 
        in_str_preproc: Optional[Callable[[str], str]]=None, 
        out_str_preproc: Optional[Callable[[str], str]]=None, 
    ) -> jnp.ndarray:
        
        if in_str_preproc is not None:
            in_strs = list(map(in_str_preproc, in_strs))
        in_tokens = [self.tokenizer.encode(item) for item in in_strs]
        in_tokens = block_sequences(in_tokens, max_input_length, self.tokenizer.pad_token_id, dtype=np.int32, pad_right=False)

        if out_str_preproc is not None:
            out_strs = list(map(out_str_preproc, out_strs))
        out_tokens = [self.tokenizer.encode(item) for item in out_strs]
        out_tokens = block_sequences(out_tokens, max_output_length, self.tokenizer.pad_token_id, dtype=np.int32)

        batch = (jnp.asarray(in_tokens), jnp.asarray(out_tokens),)

        loss = self.train_step(batch, rng_key)

        return loss

# seq2seq inference

class Seq2SeqInference(Inference):
    def generate_from_str(
        self, 
        in_strs: List[str], 
        max_input_length: int, 
        rng_key: KeyArray, 
        in_str_preproc: Optional[Callable[[str], str]]=None, 
        out_str_postproc: Optional[Callable[[str], str]]=None, 
        **generation_kwargs: Dict[str, Any]
    ) -> List[str]:
        
        if in_str_preproc is not None:
            in_strs = list(map(in_str_preproc, in_strs))
        tokens = [self.tokenizer.encode(item) for item in in_strs]
        tokens = block_sequences(tokens, max_input_length, self.tokenizer.pad_token_id, dtype=np.int32, pad_right=False)
        outputs = self.generate(jnp.asarray(tokens), rng_key, **generation_kwargs)

        out_strs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if out_str_postproc is not None:
            out_strs = list(map(out_str_postproc, out_strs))
        
        return out_strs
    
    def log_probs_from_str(
        self, 
        in_strs: List[str], 
        out_strs: List[str], 
        max_input_length: int, 
        max_output_length: int, 
        in_str_preproc: Optional[Callable[[str], str]]=None, 
        out_str_preproc: Optional[Callable[[str], str]]=None, 
    ) -> LogProbsOutput:
        
        if in_str_preproc is not None:
            in_strs = list(map(in_str_preproc, in_strs))
        in_tokens = [self.tokenizer.encode(item) for item in in_strs]
        in_tokens = block_sequences(in_tokens, max_input_length, self.tokenizer.pad_token_id, dtype=np.int32, pad_right=False)

        if out_str_preproc is not None:
            out_strs = list(map(out_str_preproc, out_strs))
        out_tokens = [self.tokenizer.encode(item) for item in out_strs]
        out_tokens = block_sequences(out_tokens, max_output_length, self.tokenizer.pad_token_id, dtype=np.int32)

        log_prob_output = self.log_probs(jnp.asarray(in_tokens), jnp.asarray(out_tokens))

        return log_prob_output
    
    def eval_loss_from_str(
        self, 
        in_strs: List[str], 
        out_strs: List[str], 
        max_input_length: int, 
        max_output_length: int, 
        in_str_preproc: Optional[Callable[[str], str]]=None, 
        out_str_preproc: Optional[Callable[[str], str]]=None, 
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        
        if in_str_preproc is not None:
            in_strs = list(map(in_str_preproc, in_strs))
        in_tokens = [self.tokenizer.encode(item) for item in in_strs]
        in_tokens = block_sequences(in_tokens, max_input_length, self.tokenizer.pad_token_id, dtype=np.int32, pad_right=False)

        if out_str_preproc is not None:
            out_strs = list(map(out_str_preproc, out_strs))
        out_tokens = [self.tokenizer.encode(item) for item in out_strs]
        out_tokens = block_sequences(out_tokens, max_output_length, self.tokenizer.pad_token_id, dtype=np.int32)

        batch = (jnp.asarray(in_tokens), jnp.asarray(out_tokens),)

        loss = self.eval_loss(batch)

        return loss

# data shard spec

class Seq2SeqDataShardSpec(NamedTuple):
    batch_spec: Any = (PartitionSpec("dp", None), PartitionSpec("dp", None))
    tokens_spec: Any = PartitionSpec("dp", None)
    logits_spec: Any = PartitionSpec("dp", None, None)
    logprobs_spec: Any = PartitionSpec("dp")

class Seq2SeqNullDataShardSpec(Seq2SeqDataShardSpec):
    batch_spec: Any = None
    tokens_spec: Any = None
    logits_spec: Any = None
    logprobs_spec: Any = None

# loss

class _LossFnType(Protocol):
    def __call__(
        self, 
        model: FlaxPreTrainedModel, 
        input_ids: jnp.ndarray, 
        decoder_input_ids: jnp.ndarray, 
        attention_mask: jnp.ndarray, 
        decoder_attention_mask: jnp.ndarray, 
        params: PyTree, 
        rng: Optional[KeyArray], 
        train: bool, 
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]: ...

def t5_enc_dec_loss(
    model: FlaxPreTrainedModel, 
    input_ids: jnp.ndarray, 
    decoder_input_ids: jnp.ndarray, 
    attention_mask: jnp.ndarray, 
    decoder_attention_mask: jnp.ndarray, 
    params: PyTree, 
    rng: Optional[KeyArray], 
    train: bool, 
) -> jnp.ndarray:
    logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, 
                   attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask, 
                   params=params, dropout_rng=rng, train=train).logits
    token_losses = softmax_cross_entropy_with_integer_labels(logits[:, :-1, :], decoder_input_ids[:, 1:]) * decoder_attention_mask[:, 1:]
    loss = token_losses.sum() / decoder_attention_mask[:, 1:].sum()
    return loss, {'loss': loss}

def gpt_dec_loss(
    model: FlaxPreTrainedModel, 
    input_ids: jnp.ndarray, 
    decoder_input_ids: jnp.ndarray, 
    attention_mask: jnp.ndarray, 
    decoder_attention_mask: jnp.ndarray, 
    params: PyTree, 
    rng: Optional[KeyArray], 
    train: bool, 
) -> jnp.ndarray:
    full_input_ids = jnp.concatenate((input_ids, decoder_input_ids,), axis=1)
    full_attn_mask = jnp.concatenate((attention_mask, decoder_attention_mask), axis=1)
    full_position_ids = jnp.maximum(jnp.cumsum(full_attn_mask, axis=1) - 1, 0).astype(jnp.int32)
    logits = model(input_ids=full_input_ids, attention_mask=full_attn_mask, position_ids=full_position_ids, 
                   params=params, dropout_rng=rng, train=train).logits
    token_losses = softmax_cross_entropy_with_integer_labels(logits[:, (input_ids.shape[1]-1):-1, :], decoder_input_ids) * decoder_attention_mask
    loss = token_losses.sum() / decoder_attention_mask.sum()
    return loss, {'loss': loss}

def opt_dec_loss(
    model: FlaxPreTrainedModel, 
    input_ids: jnp.ndarray, 
    decoder_input_ids: jnp.ndarray, 
    attention_mask: jnp.ndarray, 
    decoder_attention_mask: jnp.ndarray, 
    params: PyTree, 
    rng: Optional[KeyArray], 
    train: bool, 
) -> jnp.ndarray:
    full_input_ids = jnp.concatenate((input_ids, decoder_input_ids,), axis=1)
    full_attn_mask = jnp.concatenate((attention_mask, decoder_attention_mask), axis=1)
    full_input_ids = force_bos_to_start(full_input_ids, full_attn_mask)
    full_attn_mask = force_bos_to_start(full_attn_mask, full_attn_mask)
    full_position_ids = jnp.maximum(jnp.cumsum(full_attn_mask, axis=1) - 1, 0).astype(jnp.int32)
    logits = model(input_ids=full_input_ids, attention_mask=full_attn_mask, position_ids=full_position_ids, 
                   params=params, dropout_rng=rng, deterministic=(not train)).logits
    token_losses = softmax_cross_entropy_with_integer_labels(logits[:, (input_ids.shape[1]-1):-1, :], decoder_input_ids) * decoder_attention_mask
    loss = token_losses.sum() / decoder_attention_mask.sum()
    return loss, {'loss': loss}

# load model parallel t5_encdec jax trainer

def load_t5_enc_dec_trainer(
    model: FlaxPreTrainedModel, 
    params: PyTree, 
    param_spec: Optional[Any], 
    tokenizer: PreTrainedTokenizer, 
    optim: optax.GradientTransformation, 
    optim_state: PyTree, 
    optim_state_spec: Optional[Any], 
    do_pjit: bool, 
    loss_fn: _LossFnType=t5_enc_dec_loss, 
    data_shard_spec: Seq2SeqDataShardSpec=Seq2SeqDataShardSpec(), 
) -> Seq2SeqTrainer:

    pad_id = jnp.asarray(tokenizer.pad_token_id, dtype=jnp.int32)

    # define seq2seq training step
    def step_fn(params: PyTree, optim_state: PyTree, rng: KeyArray, batch: PyTree):
        # ensure it is sharded properly
        if do_pjit:
            batch = with_sharding_constraint(batch, data_shard_spec.batch_spec)
        in_tokens, out_tokens = batch
        in_attn_mask = (in_tokens != pad_id).astype(jnp.int32)
        out_attn_mask = (out_tokens != pad_id).astype(jnp.int32)
        out_attn_mask = out_attn_mask.at[:, 0].set(1)
        def grad_loss(params: PyTree):
            loss, info = loss_fn(model, in_tokens, out_tokens, in_attn_mask, out_attn_mask, params, rng, True)
            return loss, info
        (loss, info), grads = jax.value_and_grad(grad_loss, has_aux=True)(params)
        if do_pjit:
            grads = with_sharding_constraint(grads, param_spec)
        updates, optim_state = optim.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return TrainStepOutput(loss, info, params, optim_state)

    if do_pjit:
        p_step_fn = pjit(
            step_fn, 
            in_axis_resources=(param_spec, optim_state_spec, None, data_shard_spec.batch_spec,), 
            out_axis_resources=TrainStepOutput(None, None, param_spec, optim_state_spec), 
            donate_argnums=(0, 1), 
        )
    else:
        p_step_fn = step_fn
    
    train_interface = Seq2SeqTrainer(params, optim_state, tokenizer, p_step_fn)

    return train_interface

# load model parallel t5_encdec jax inference

def load_t5_enc_dec_inference(
    model: FlaxPreTrainedModel, 
    params: PyTree, 
    param_spec: Optional[Any], 
    tokenizer: PreTrainedTokenizer, 
    do_pjit: bool, 
    loss_fn: Optional[_LossFnType]=t5_enc_dec_loss, 
    data_shard_spec: Seq2SeqDataShardSpec=Seq2SeqDataShardSpec(), 
) -> Seq2SeqInference:

    has_loss_fn = loss_fn is not None
    
    pad_id = jnp.asarray(tokenizer.pad_token_id, dtype=jnp.int32)
    
    # define generation_fn
    def generate_fn(params: PyTree, rng: KeyArray, in_tokens: jnp.ndarray, kwargs: Dict[str, Any]) -> jnp.ndarray:
        if do_pjit:
            in_tokens = with_sharding_constraint(in_tokens, data_shard_spec.tokens_spec)
        attn_mask = (in_tokens != pad_id).astype(jnp.int32)
        out_sequences = model.generate(in_tokens, attention_mask=attn_mask, params=params, prng_key=rng, **kwargs).sequences
        if do_pjit:
            out_sequences = with_sharding_constraint(out_sequences, data_shard_spec.tokens_spec)
        return out_sequences
    
    if do_pjit:
        p_generate_fn = pjit(
            generate_fn, 
            in_axis_resources=(param_spec, None, data_shard_spec.tokens_spec), 
            out_axis_resources=data_shard_spec.tokens_spec, 
            static_argnums=(3,), 
        )
    else:
        p_generate_fn = generate_fn
    
    # define logprob function
    def logprob_fn(params: PyTree, in_tokens: jnp.ndarray, out_tokens: jnp.ndarray) -> jnp.ndarray:
        if do_pjit:
            in_tokens = with_sharding_constraint(in_tokens, data_shard_spec.tokens_spec)
            out_tokens = with_sharding_constraint(out_tokens, data_shard_spec.tokens_spec)
        in_attn_mask = (in_tokens != pad_id).astype(jnp.int32)
        out_attn_mask = (out_tokens != pad_id).astype(jnp.int32)
        out_attn_mask = out_attn_mask.at[:, 0].set(1)
        logits = model(input_ids=in_tokens, attention_mask=in_attn_mask, 
                       decoder_input_ids=out_tokens, decoder_attention_mask=out_attn_mask, 
                       params=params, train=False).logits
        log_probs = -(softmax_cross_entropy_with_integer_labels(logits[:, :-1, :], out_tokens[:, 1:]) * out_attn_mask[:, 1:]).sum(axis=1)
        if do_pjit:
            logits = with_sharding_constraint(logits, data_shard_spec.logits_spec)
            log_probs = with_sharding_constraint(log_probs, data_shard_spec.logprobs_spec)
        return LogProbsOutput(log_probs, logits)
    
    if do_pjit:
        p_logprob_fn = pjit(
            logprob_fn, 
            in_axis_resources=(param_spec, data_shard_spec.tokens_spec, data_shard_spec.tokens_spec,), 
            out_axis_resources=LogProbsOutput(data_shard_spec.logprobs_spec, data_shard_spec.logits_spec), 
        )
    else:
        p_logprob_fn = logprob_fn
    
    # define eval loss
    def eval_loss_fn(params: PyTree, batch: PyTree) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        if not has_loss_fn:
            raise NotImplementedError
        if do_pjit:
            batch = with_sharding_constraint(batch, data_shard_spec.batch_spec)
        in_tokens, out_tokens = batch
        in_attn_mask = (in_tokens != pad_id).astype(jnp.int32)
        out_attn_mask = (out_tokens != pad_id).astype(jnp.int32)
        out_attn_mask = out_attn_mask.at[:, 0].set(1)
        loss, info = loss_fn(model, in_tokens, out_tokens, in_attn_mask, out_attn_mask, params, None, False)
        return loss, info
    
    if do_pjit and has_loss_fn:
        p_eval_loss_fn = pjit(
            eval_loss_fn, 
            in_axis_resources=(param_spec, data_shard_spec.batch_spec,), 
            out_axis_resources=None, 
        )
    else:
        p_eval_loss_fn = eval_loss_fn

    inference_inferface = Seq2SeqInference(params, tokenizer, p_generate_fn, p_logprob_fn, p_eval_loss_fn)

    return inference_inferface

# load model parallel gpt_dec jax trainer

def load_gpt_dec_trainer(
    model: FlaxPreTrainedModel, 
    params: PyTree, 
    param_spec: Optional[Any], 
    tokenizer: PreTrainedTokenizer, 
    optim: optax.GradientTransformation, 
    optim_state: PyTree, 
    optim_state_spec: Optional[Any], 
    do_pjit: bool, 
    loss_fn: _LossFnType=gpt_dec_loss, 
    data_shard_spec: Seq2SeqDataShardSpec=Seq2SeqDataShardSpec(), 
) -> Seq2SeqTrainer:

    pad_id = jnp.asarray(tokenizer.pad_token_id, dtype=jnp.int32)

    # define seq2seq training step
    def step_fn(params: PyTree, optim_state: PyTree, rng: KeyArray, batch: PyTree):
        
        # ensure it is sharded properly
        if do_pjit:
            batch = with_sharding_constraint(batch, data_shard_spec.batch_spec)
        
        in_tokens, out_tokens = batch
        if in_tokens.shape[1] == 0:
            in_tokens, out_tokens = out_tokens[:, :1], out_tokens[:, 1:]
        in_attn_mask = (in_tokens != pad_id).astype(jnp.int32)
        out_attn_mask = (out_tokens != pad_id).astype(jnp.int32)

        def grad_loss(params: PyTree):
            loss, info = loss_fn(model, in_tokens, out_tokens, in_attn_mask, out_attn_mask, params, rng, True)
            return loss, info
        (loss, info), grads = jax.value_and_grad(grad_loss, has_aux=True)(params)
        if do_pjit:
            grads = with_sharding_constraint(grads, param_spec)
        
        updates, optim_state = optim.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        
        return TrainStepOutput(loss, info, params, optim_state)

    if do_pjit:
        p_step_fn = pjit(
            step_fn, 
            in_axis_resources=(param_spec, optim_state_spec, None, data_shard_spec.batch_spec,), 
            out_axis_resources=TrainStepOutput(None, None, param_spec, optim_state_spec), 
            donate_argnums=(0, 1), 
        )
    else:
        p_step_fn = step_fn
    
    train_interface = Seq2SeqTrainer(params, optim_state, tokenizer, p_step_fn)

    return train_interface

# load model parallel gpt_dec jax inference

def load_gpt_dec_inference(
    model: FlaxPreTrainedModel, 
    params: PyTree, 
    param_spec: Optional[Any], 
    tokenizer: PreTrainedTokenizer, 
    do_pjit: bool, 
    loss_fn: Optional[_LossFnType]=gpt_dec_loss, 
    data_shard_spec: Seq2SeqDataShardSpec=Seq2SeqDataShardSpec(), 
) -> Seq2SeqInference:

    has_loss_fn = loss_fn is not None
    
    pad_id = jnp.asarray(tokenizer.pad_token_id, dtype=jnp.int32)
    
    # define generation_fn
    def generate_fn(params: PyTree, rng: KeyArray, in_tokens: jnp.ndarray, kwargs: Dict[str, Any]) -> jnp.ndarray:
        if do_pjit:
            in_tokens = with_sharding_constraint(in_tokens, data_shard_spec.tokens_spec)
        attn_mask = (in_tokens != pad_id).astype(jnp.int32)
        out_sequences = model.generate(in_tokens, attention_mask=attn_mask, params=params, prng_key=rng, **kwargs).sequences[:, in_tokens.shape[1]:]
        if do_pjit:
            out_sequences = with_sharding_constraint(out_sequences, data_shard_spec.tokens_spec)
        return out_sequences
    
    if do_pjit:
        p_generate_fn = pjit(
            generate_fn, 
            in_axis_resources=(param_spec, None, data_shard_spec.tokens_spec), 
            out_axis_resources=data_shard_spec.tokens_spec, 
            static_argnums=(3,), 
        )
    else:
        p_generate_fn = generate_fn
    
    # define logprob function
    def logprob_fn(params: PyTree, in_tokens: jnp.ndarray, out_tokens: jnp.ndarray) -> jnp.ndarray:
        if do_pjit:
            in_tokens = with_sharding_constraint(in_tokens, data_shard_spec.tokens_spec)
            out_tokens = with_sharding_constraint(out_tokens, data_shard_spec.tokens_spec)
        
        in_attn_mask = (in_tokens != pad_id).astype(jnp.int32)
        out_attn_mask = (out_tokens != pad_id).astype(jnp.int32)

        full_tokens = jnp.concatenate((in_tokens, out_tokens,), axis=1)
        full_attn_mask = jnp.concatenate((in_attn_mask, out_attn_mask,), axis=1)
        full_position_ids = jnp.maximum(jnp.cumsum(full_attn_mask, axis=1) - 1, 0).astype(jnp.int32)

        logits = model(input_ids=full_tokens, attention_mask=full_attn_mask, 
                       position_ids=full_position_ids, params=params, train=False).logits
        log_probs = -(softmax_cross_entropy_with_integer_labels(logits[:, (in_tokens.shape[1]-1):-1, :], out_tokens) * out_attn_mask).sum(axis=1)
        if do_pjit:
            logits = with_sharding_constraint(logits, data_shard_spec.logits_spec)
            log_probs = with_sharding_constraint(log_probs, data_shard_spec.logprobs_spec)
        return LogProbsOutput(log_probs, logits[:, (in_tokens.shape[1]-1):-1, :])
    
    if do_pjit:
        p_logprob_fn = pjit(
            logprob_fn, 
            in_axis_resources=(param_spec, data_shard_spec.tokens_spec, data_shard_spec.tokens_spec,), 
            out_axis_resources=LogProbsOutput(data_shard_spec.logprobs_spec, data_shard_spec.logits_spec), 
        )
    else:
        p_logprob_fn = logprob_fn
    
    # define eval loss
    def eval_loss_fn(params: PyTree, batch: PyTree) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        if not has_loss_fn:
            raise NotImplementedError
        # ensure it is sharded properly
        if do_pjit:
            batch = with_sharding_constraint(batch, data_shard_spec.batch_spec)
        
        in_tokens, out_tokens = batch
        if in_tokens.shape[1] == 0:
            in_tokens, out_tokens = out_tokens[:, :1], out_tokens[:, 1:]
        in_attn_mask = (in_tokens != pad_id).astype(jnp.int32)
        out_attn_mask = (out_tokens != pad_id).astype(jnp.int32)

        loss, info = loss_fn(model, in_tokens, out_tokens, in_attn_mask, out_attn_mask, params, None, False)
        return loss, info
    
    if do_pjit and has_loss_fn:
        p_eval_loss_fn = pjit(
            eval_loss_fn, 
            in_axis_resources=(param_spec, data_shard_spec.batch_spec,), 
            out_axis_resources=None, 
        )
    else:
        p_eval_loss_fn = eval_loss_fn

    inference_inferface = Seq2SeqInference(params, tokenizer, p_generate_fn, p_logprob_fn, p_eval_loss_fn)

    return inference_inferface

# load model parallel gpt_dec jax trainer

def load_opt_dec_trainer(
    model: FlaxPreTrainedModel, 
    params: PyTree, 
    param_spec: Optional[Any], 
    tokenizer: PreTrainedTokenizer, 
    optim: optax.GradientTransformation, 
    optim_state: PyTree, 
    optim_state_spec: Optional[Any], 
    do_pjit: bool, 
    loss_fn: _LossFnType=opt_dec_loss, 
    data_shard_spec: Seq2SeqDataShardSpec=Seq2SeqDataShardSpec(), 
) -> Seq2SeqTrainer:

    pad_id = jnp.asarray(tokenizer.pad_token_id, dtype=jnp.int32)

    # define seq2seq training step
    def step_fn(params: PyTree, optim_state: PyTree, rng: KeyArray, batch: PyTree):
        
        # ensure it is sharded properly
        if do_pjit:
            batch = with_sharding_constraint(batch, data_shard_spec.batch_spec)
        
        in_tokens, out_tokens = batch
        if in_tokens.shape[1] == 0:
            in_tokens, out_tokens = out_tokens[:, :1], out_tokens[:, 1:]
        in_attn_mask = (in_tokens != pad_id).astype(jnp.int32)
        out_attn_mask = (out_tokens != pad_id).astype(jnp.int32)

        def grad_loss(params: PyTree):
            loss, info = loss_fn(model, in_tokens, out_tokens, in_attn_mask, out_attn_mask, params, rng, True)
            return loss, info
        (loss, info), grads = jax.value_and_grad(grad_loss, has_aux=True)(params)
        if do_pjit:
            grads = with_sharding_constraint(grads, param_spec)
        
        updates, optim_state = optim.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        
        return TrainStepOutput(loss, info, params, optim_state)

    if do_pjit:
        p_step_fn = pjit(
            step_fn, 
            in_axis_resources=(param_spec, optim_state_spec, None, data_shard_spec.batch_spec,), 
            out_axis_resources=TrainStepOutput(None, None, param_spec, optim_state_spec), 
            donate_argnums=(0, 1), 
        )
    else:
        p_step_fn = step_fn
    
    train_interface = Seq2SeqTrainer(params, optim_state, tokenizer, p_step_fn)

    return train_interface

# load model parallel gpt_dec jax inference

def load_opt_dec_inference(
    model: FlaxPreTrainedModel, 
    params: PyTree, 
    param_spec: Optional[Any], 
    tokenizer: PreTrainedTokenizer, 
    do_pjit: bool, 
    loss_fn: Optional[_LossFnType]=opt_dec_loss, 
    data_shard_spec: Seq2SeqDataShardSpec=Seq2SeqDataShardSpec(), 
) -> Seq2SeqInference:

    has_loss_fn = loss_fn is not None
    
    pad_id = jnp.asarray(tokenizer.pad_token_id, dtype=jnp.int32)
    
    # define generation_fn
    def generate_fn(params: PyTree, rng: KeyArray, in_tokens: jnp.ndarray, kwargs: Dict[str, Any]) -> jnp.ndarray:
        if do_pjit:
            in_tokens = with_sharding_constraint(in_tokens, data_shard_spec.tokens_spec)
        attn_mask = (in_tokens != pad_id).astype(jnp.int32)
        in_tokens = force_bos_to_start(in_tokens, attn_mask)
        attn_mask = force_bos_to_start(attn_mask, attn_mask)
        out_sequences = model.generate(in_tokens, attention_mask=attn_mask, params=params, prng_key=rng, **kwargs).sequences[:, in_tokens.shape[1]:]
        if do_pjit:
            out_sequences = with_sharding_constraint(out_sequences, data_shard_spec.tokens_spec)
        return out_sequences
    
    if do_pjit:
        p_generate_fn = pjit(
            generate_fn, 
            in_axis_resources=(param_spec, None, data_shard_spec.tokens_spec), 
            out_axis_resources=data_shard_spec.tokens_spec, 
            static_argnums=(3,), 
        )
    else:
        p_generate_fn = generate_fn
    
    # define logprob function
    def logprob_fn(params: PyTree, in_tokens: jnp.ndarray, out_tokens: jnp.ndarray) -> jnp.ndarray:
        if do_pjit:
            in_tokens = with_sharding_constraint(in_tokens, data_shard_spec.tokens_spec)
            out_tokens = with_sharding_constraint(out_tokens, data_shard_spec.tokens_spec)
        
        in_attn_mask = (in_tokens != pad_id).astype(jnp.int32)
        out_attn_mask = (out_tokens != pad_id).astype(jnp.int32)

        full_tokens = jnp.concatenate((in_tokens, out_tokens,), axis=1)
        full_attn_mask = jnp.concatenate((in_attn_mask, out_attn_mask,), axis=1)
        full_tokens = force_bos_to_start(full_tokens, full_attn_mask)
        full_attn_mask = force_bos_to_start(full_attn_mask, full_attn_mask)
        full_position_ids = jnp.maximum(jnp.cumsum(full_attn_mask, axis=1) - 1, 0).astype(jnp.int32)

        logits = model(input_ids=full_tokens, attention_mask=full_attn_mask, 
                       position_ids=full_position_ids, params=params, train=False).logits
        log_probs = -(softmax_cross_entropy_with_integer_labels(logits[:, (in_tokens.shape[1]-1):-1, :], out_tokens) * out_attn_mask).sum(axis=1)
        if do_pjit:
            logits = with_sharding_constraint(logits, data_shard_spec.logits_spec)
            log_probs = with_sharding_constraint(log_probs, data_shard_spec.logprobs_spec)
        return LogProbsOutput(log_probs, logits[:, (in_tokens.shape[1]-1):-1, :])
    
    if do_pjit:
        p_logprob_fn = pjit(
            logprob_fn, 
            in_axis_resources=(param_spec, data_shard_spec.tokens_spec, data_shard_spec.tokens_spec,), 
            out_axis_resources=LogProbsOutput(data_shard_spec.logprobs_spec, data_shard_spec.logits_spec), 
        )
    else:
        p_logprob_fn = logprob_fn
    
    # define eval loss
    def eval_loss_fn(params: PyTree, batch: PyTree) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        if not has_loss_fn:
            raise NotImplementedError
        # ensure it is sharded properly
        if do_pjit:
            batch = with_sharding_constraint(batch, data_shard_spec.batch_spec)
        
        in_tokens, out_tokens = batch
        if in_tokens.shape[1] == 0:
            in_tokens, out_tokens = out_tokens[:, :1], out_tokens[:, 1:]
        in_attn_mask = (in_tokens != pad_id).astype(jnp.int32)
        out_attn_mask = (out_tokens != pad_id).astype(jnp.int32)

        loss, info = loss_fn(model, in_tokens, out_tokens, in_attn_mask, out_attn_mask, params, None, False)
        return loss, info
    
    if do_pjit and has_loss_fn:
        p_eval_loss_fn = pjit(
            eval_loss_fn, 
            in_axis_resources=(param_spec, data_shard_spec.batch_spec,), 
            out_axis_resources=None, 
        )
    else:
        p_eval_loss_fn = eval_loss_fn

    inference_inferface = Seq2SeqInference(params, tokenizer, p_generate_fn, p_logprob_fn, p_eval_loss_fn)

    return inference_inferface
