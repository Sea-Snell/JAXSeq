from typing import Dict, Optional, Any, Callable, Generator, List
from transformers.generation.flax_logits_process import FlaxLogitsProcessorList
from transformers.generation.flax_utils import GreedyState, FlaxGreedySearchOutput, SampleState, FlaxSampleOutput
from transformers.generation.configuration_utils import GenerationConfig
from transformers.tokenization_utils import PreTrainedTokenizerBase
import jax.numpy as jnp
from jax import lax
import jax
import logging
import warnings
import copy
from transformers.utils import logging
import json
from JaxSeq.serve import SSEServer
from JaxSeq.serve import Config as ServeConfig
import threading
import time
from JaxSeq.utils import uuid_name
from functools import partial
from sseclient import SSEClient
import base64
from jax.experimental import host_callback
from dataclasses import dataclass

logger = logging.get_logger(__name__)

@dataclass
class DtypePlaceholder:
    shape: Any
    dtype: Any

class StreamingGenerationConfig(GenerationConfig):
    """
    Configuration class for :class:`~JaxSeq.StreamingGenerationMixin`.
    """
    def __init__(self, **kwargs):
        self.streaming_callback = kwargs.pop("streaming_callback", None)
        self.stop_callback = kwargs.pop("stop_callback", None)
        super().__init__(**kwargs)
    
    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        # remove streaming_callback, stop_callback since they can't be serialized
        config_dict["streaming_callback"] = None
        config_dict["stop_callback"] = None
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

class FlaxStreamGenerationMixin:
    """
    overrides Huggingface sampling and greedy generation functions to support token streaming in Jax.
    """
    
    def generate(
        self,
        input_ids: jnp.ndarray,
        generation_config: Optional[GenerationConfig] = None,
        prng_key: Optional[jnp.ndarray] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        **kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head.

        Parameters:
            input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            trace (`bool`, *optional*, defaults to `True`):
                Whether to trace generation. Setting `trace=False` should only be used for debugging and will lead to a
                considerably slower runtime.
            params (`Dict[str, jnp.ndarray]`, *optional*):
                Optionally the model parameters can be passed. Can be useful for parallelized generation.
            kwargs:
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`].

        """
        # Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        self._validate_model_kwargs(model_kwargs.copy())

        # set init values
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask") is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        if generation_config.decoder_start_token_id is None and self.config.is_encoder_decoder:
            raise ValueError("`decoder_start_token_id` has to be defined for encoder-decoder generation.")

        # decoder-only models should use left-padding for generation (can't be checked with `trace=True`)
        if not self.config.is_encoder_decoder and not trace:
            if (
                generation_config.pad_token_id is not None
                and jnp.sum(input_ids[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            if model_kwargs.get("encoder_outputs") is None:
                model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, params, model_kwargs)
            # prepare decoder_input_ids for generation
            input_ids = jnp.ones((input_ids.shape[0], 1), dtype="i4") * generation_config.decoder_start_token_id

        # Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                "Neither `max_length` nor `max_new_tokens` have been set, `max_length` will default to"
                f" {generation_config.max_length} (`generation_config.max_length`). Controlling `max_length` via the"
                " config is deprecated and `max_length` will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif has_default_max_length and generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        elif not has_default_max_length and generation_config.max_new_tokens is not None:
            raise ValueError(
                "Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a"
                " limit to the generated output length. Remove one of those arguments. Please refer to the"
                " documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasable length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing`max_new_tokens`."
            )

        logits_processor = self._get_logits_processor(generation_config=generation_config)

        if not generation_config.do_sample and generation_config.num_beams == 1:
            return self._greedy_search(
                input_ids,
                generation_config.max_length,
                generation_config.pad_token_id,
                generation_config.eos_token_id,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
                streaming_callback=generation_config.streaming_callback \
                    if hasattr(generation_config, "streaming_callback") else None, 
                stop_callback=generation_config.stop_callback \
                    if hasattr(generation_config, "stop_callback") else None, 
            )
        elif generation_config.do_sample and generation_config.num_beams == 1:
            logits_warper = self._get_logits_warper(generation_config=generation_config)
            return self._sample(
                input_ids,
                generation_config.max_length,
                generation_config.pad_token_id,
                generation_config.eos_token_id,
                prng_key,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
                streaming_callback=generation_config.streaming_callback \
                    if hasattr(generation_config, "streaming_callback") else None, 
                stop_callback=generation_config.stop_callback \
                    if hasattr(generation_config, "stop_callback") else None, 
            )
        elif not generation_config.do_sample and generation_config.num_beams > 1:
            assert (not hasattr(generation_config, "streaming_callback")) or \
                (generation_config.streaming_callback is None), "Beam search does not support streaming callback"
            # broadcast input_ids & encoder_outputs
            input_ids = self._expand_to_num_beams(input_ids, num_beams=generation_config.num_beams)

            if "encoder_outputs" in model_kwargs:
                model_kwargs["encoder_outputs"]["last_hidden_state"] = self._expand_to_num_beams(
                    model_kwargs["encoder_outputs"]["last_hidden_state"], num_beams=generation_config.num_beams
                )

            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = self._expand_to_num_beams(
                    model_kwargs["attention_mask"], num_beams=generation_config.num_beams
                )

            return self._beam_search(
                input_ids,
                generation_config.max_length,
                generation_config.pad_token_id,
                generation_config.eos_token_id,
                length_penalty=generation_config.length_penalty,
                early_stopping=generation_config.early_stopping,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        else:
            raise NotImplementedError("`Beam sampling is currently not implemented.")
    
    def _greedy_search(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
        streaming_callback: Optional[Callable[[Optional[jax.Array]], None]]=None, 
        stop_callback: Optional[Callable[[None], bool]]=None, 
    ):
        # init values
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self
        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)

        # initialize state
        state = GreedyState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            model_kwargs=model_kwargs,
        )

        def greedy_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            # check host if should stop early
            if stop_callback is not None:
                should_force_stop = host_callback.call(
                    stop_callback, 
                    None, 
                    result_shape=DtypePlaceholder(tuple(), jnp.bool_), 
                )
                finish_generation = jnp.logical_or(finish_generation, should_force_stop)
            # notify streaming callback that generation is done
            def did_finish_f():
                if streaming_callback is not None:
                    jax.debug.callback(streaming_callback, None)
                return None
            jax.lax.cond(
                finish_generation, 
                did_finish_f, 
                lambda: None, 
            )
            return ~finish_generation

        def greedy_search_body_fn(state):
            """state update fn."""
            model_outputs = model(state.running_token, params=params, **state.model_kwargs)
            logits = model_outputs.logits[:, -1]

            # apply min_length, ...
            logits = logits_processor(state.sequences, logits, state.cur_len)

            next_token = jnp.argmax(logits, axis=-1)

            next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)

            # stream tokens to host
            if streaming_callback is not None:
                jax.debug.callback(streaming_callback, next_sequences)

            return GreedyState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[1] > 1:
            state = greedy_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(greedy_search_cond_fn, greedy_search_body_fn, state)
        else:
            state = lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)

        return FlaxGreedySearchOutput(sequences=state.sequences)
    

    def _sample(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        prng_key: Optional[jnp.ndarray] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        logits_warper: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
        streaming_callback: Optional[Callable[[Optional[jax.Array]], None]]=None, 
        stop_callback: Optional[Callable[[], bool]]=None, 
    ):
        # init values
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self

        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)

        # initialize state
        state = SampleState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            prng_key=prng_key,
            model_kwargs=model_kwargs,
        )

        def sample_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            # check host if should stop early
            if stop_callback is not None:
                should_force_stop = host_callback.call(
                    stop_callback, 
                    None, 
                    result_shape=DtypePlaceholder(tuple(), jnp.bool_), 
                )
                finish_generation = jnp.logical_or(finish_generation, should_force_stop)
            # notify streaming callback that generation is done
            def did_finish_f():
                if streaming_callback is not None:
                    jax.debug.callback(streaming_callback, None)
                return None
            jax.lax.cond(
                finish_generation, 
                did_finish_f, 
                lambda: None, 
            )
            return ~finish_generation

        def sample_search_body_fn(state):
            """state update fn."""
            prng_key, prng_key_next = jax.random.split(state.prng_key)
            model_outputs = model(state.running_token, params=params, **state.model_kwargs)

            logits = model_outputs.logits[:, -1]

            # apply min_length, ...
            logits = logits_processor(state.sequences, logits, state.cur_len)
            # apply top_p, top_k, temperature
            logits = logits_warper(logits, logits, state.cur_len)

            next_token = jax.random.categorical(prng_key, logits, axis=-1)

            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token * ~next_is_sent_finished + pad_token_id * next_is_sent_finished
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)

            # stream tokens to host
            if streaming_callback is not None:
                jax.debug.callback(streaming_callback, next_sequences)

            return SampleState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
                prng_key=prng_key_next,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[1] > 1:
            state = sample_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(sample_search_cond_fn, sample_search_body_fn, state)
        else:
            state = lax.while_loop(sample_search_cond_fn, sample_search_body_fn, state)
        
        return FlaxSampleOutput(sequences=state.sequences)


# publish tokens from callback_f to sse_server

class TokenCallbackHandler:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.sse_server = SSEServer()
        self._request_id = None
        self._stop = False
        self._lock = threading.Lock()

        def _streaming_callback_f(tokens: Optional[jax.Array]):
            assert self._request_id is not None
            try:
                if tokens is not None:
                    token_strs = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
                    self.sse_server.publish(
                        dict(
                            data=dict(
                                strs=token_strs, 
                                request_id=self._request_id, 
                            ), 
                            status='success', 
                        ), 
                        channel=self._request_id, 
                        type='message', 
                    )
                else:
                    self.sse_server.publish('', channel=self._request_id, type=ServeConfig.sse_exit_type)
            except Exception as e:
                print(e)
                self.sse_server.publish(
                    dict(
                        status='error', 
                        error_message=str(e), 
                    ), 
                    channel=self._request_id, 
                    type='error', 
                )
            if tokens is None:
                self._stop = False
                self._request_id = None
                self._lock.release()
        
        def _stop_callback_f(_) -> bool:
            return self._stop
        
        self.streaming_callback_f = _streaming_callback_f
        self.stop_callback_f = _stop_callback_f
    
    def get_streaming_callback_f(self) -> Callable:
        return self.streaming_callback_f
    
    def get_stop_callback_f(self) -> Callable:
        return self.stop_callback_f
    
    def get_listener_channel(self, seed: str='') -> str:
        self._lock.acquire()
        assert self._request_id is None
        self._request_id = base64.encodebytes(uuid_name(seed, include_uuid=True).encode('utf-8')).decode('utf-8')
        return self._request_id
    
    def force_stop(self, request_id: str) -> bool:
        # only stop if the request id matches
        if (self._request_id is not None) and (self._request_id == request_id):
            self._stop = True
            return True
        return False


# maps the token stream to a callback function

def map_token_stream(
    sse_client: SSEClient, 
    callback: Callable[[str, List[str], List[str]], None], # takes (request id, full strings, next tokens) as input
    error_callback: Optional[Callable[[str], None]]=None, # takes an error message as input
) -> None:
    past_prefixes = None
    for message in sse_client.events():
        message = json.loads(message.data)
        if message['status'] == 'error':
            error_callback(message['error_message'])
            break
        elif message['status'] == 'success':
            if past_prefixes is None:
                past_prefixes = ['' for _ in message['data']]
            callback(
                message['data']['request_id'], 
                message['data']['strs'], 
                list(map(lambda prefix, d: d.removeprefix(prefix), past_prefixes, message['data']['strs'])), 
            )
            past_prefixes = message['data']['strs']
        else:
            raise NotImplementedError
