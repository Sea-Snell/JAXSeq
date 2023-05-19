from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from transformers.generation import GenerationConfig
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as PS
from jaxtyping import PyTree
from flax import struct
from functools import partial
from typing import List, Optional, Union, Tuple, Callable, NamedTuple
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
from JaxSeq.utils import with_named_sharding_constraint, match_partition_rules, BlockingStrategy, block_sequences, Padding, Truncation
from optax import softmax_cross_entropy_with_integer_labels
from flax.training.train_state import TrainState
from transformers.modeling_flax_outputs import FlaxCausalLMOutput
from transformers.generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput, FlaxSampleOutput
from flax.core import FrozenDict, freeze
from jax.experimental.pjit import pjit

def initialize_attn_mask_pos_ids(
    input_ids: jax.Array,  
    pad_token_id: Optional[Union[int, jax.Array]], 
    attention_mask: Optional[jax.Array]=None, 
    position_ids: Optional[jax.Array]=None, 
    position_id_shift: Optional[jax.Array]=None, 
) -> Tuple[jax.Array, jax.Array]:
    if attention_mask is None:
        if pad_token_id is None:
            attention_mask = jnp.ones_like(input_ids).astype(jnp.int32)
        else:
            attention_mask = (input_ids != pad_token_id).astype(jnp.int32)
    if position_ids is None:
        position_ids = jnp.maximum(jnp.cumsum(attention_mask, axis=1) - 1, 0).astype(jnp.int32)
        if position_id_shift is not None:
            position_ids = position_ids + position_id_shift[:, None]
    return attention_mask, position_ids

class GenerationFromStrOutput(NamedTuple):
    output_strs: List[str]
    scores: np.ndarray

# inference based on sequence-to-sequence input/outputs

class Inference(struct.PyTreeNode):
    params: PyTree
    model: FlaxPreTrainedModel = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _generate: Callable = struct.field(pytree_node=False)
    _forward: Callable = struct.field(pytree_node=False)
    _eval_loss: Optional[Callable] = struct.field(pytree_node=False, default=None)
    
    # def _generate(
    #     params: PyTree, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray], 
    #     generation_config: Optional[FrozenDict]=None, 
    #     trace: bool=True, 
    # ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
    #     raise NotImplementedError
    
    # def _forward(
    #     params: PyTree, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     output_attentions: Optional[bool]=None, 
    #     output_hidden_states: Optional[bool]=None, 
    #     train: bool=False, 
    # ) -> FlaxCausalLMOutput:
    #     raise NotImplementedError

    # def _eval_loss(
    #     params: PyTree, 
    #     input_ids: jax.Array, 
    #     input_attention_mask: jax.Array, 
    #     input_position_ids: jax.Array, 
    #     target_ids: jax.Array, 
    #     target_attention_mask: jax.Array, 
    #     target_position_ids: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray], 
    #     train: bool=False, 
    # ) -> Tuple[jax.Array, PyTree]:
    #     raise NotImplementedError

    @staticmethod
    @pjit
    def logprobs_from_logits(
        logits: jax.Array, 
        input_ids: jax.Array, 
        target_ids: jax.Array, 
        target_attention_mask: jax.Array, 
    ) -> jax.Array:
        log_logits = -softmax_cross_entropy_with_integer_labels(logits[:, (input_ids.shape[1]-1):-1, :].astype(jnp.float32), target_ids)
        log_probs = jnp.where(target_attention_mask == 1, log_logits, 0.0).sum(axis=1)
        return log_probs
    
    def generate(
        self, 
        input_ids: jax.Array, 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        generation_config: Optional[GenerationConfig]=None, 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        trace: bool=True, 
    ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        return self._generate(
            self.params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            prng_key, 
            freeze(generation_config.to_dict()) if generation_config is not None else None, 
            trace, 
        )
    
    def generate_from_str(
        self, 
        input_strs: List[str], 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), 
        generation_config: Optional[GenerationConfig]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        target_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        trace: bool=True, 
    ) -> GenerationFromStrOutput:
        if input_token_process is None:
            input_token_process = lambda x: x
        if target_token_process is None:
            target_token_process = lambda x: x
        # tokenize
        tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        tokens = block_sequences(tokens, self.tokenizer.pad_token_id, np.int32, blocking_strategy)
        # generate
        outputs = self.generate(
            jnp.asarray(tokens), 
            prng_key, 
            generation_config=generation_config, 
            trace=trace
        )
        # process outputs
        output_sequences = list(map(target_token_process, outputs.sequences.tolist()))
        output_scores = None
        if isinstance(outputs, FlaxBeamSearchOutput):
            output_scores = np.asarray(outputs.scores)
        # decode tokens
        output_strs = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        return GenerationFromStrOutput(output_strs, output_scores)
    
    def forward(
        self, 
        input_ids: jax.Array, 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        output_attentions: Optional[bool]=None, 
        output_hidden_states: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    ) -> FlaxCausalLMOutput:
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        return self._forward(
            self.params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            prng_key, 
            output_attentions, 
            output_hidden_states, 
            train, 
        )
    
    def forward_from_str(
        self, 
        input_strs: List[str], 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=None), 
        output_attentions: Optional[bool]=None, 
        output_hidden_states: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> FlaxCausalLMOutput:
        if input_token_process is None:
            input_token_process = lambda x: x
        # tokenize
        tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        tokens = block_sequences(tokens, self.tokenizer.pad_token_id, np.int32, blocking_strategy)
        # forward
        outputs = self.forward(
            jnp.asarray(tokens), 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
            train=train, 
            prng_key=prng_key, 
        )
        return outputs
    
    def logprob(
        self, 
        input_ids: jax.Array, 
        target_ids: jax.Array, 
        input_attention_mask: Optional[jax.Array]=None, 
        input_position_ids: Optional[jax.Array]=None, 
        target_attention_mask: Optional[jax.Array]=None, 
        target_position_ids: Optional[jax.Array]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    ) -> jax.Array:
        input_attention_mask, input_position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            input_attention_mask, 
            input_position_ids, 
        )
        target_attention_mask, target_position_ids = initialize_attn_mask_pos_ids(
            target_ids, 
            self.tokenizer.pad_token_id, 
            target_attention_mask, 
            target_position_ids, 
            position_id_shift=input_position_ids.max(axis=1)+(input_attention_mask.sum(axis=1) > 0).astype(jnp.int32), 
        )

        full_ids = jnp.concatenate((input_ids, target_ids), axis=1)
        full_attention_mask = jnp.concatenate((input_attention_mask, target_attention_mask), axis=1)
        full_position_ids = jnp.concatenate((input_position_ids, target_position_ids), axis=1)

        logits = self.forward(
            full_ids, 
            attention_mask=full_attention_mask, 
            position_ids=full_position_ids, 
            train=train, 
            prng_key=prng_key, 
        ).logits
        
        log_probs = self.logprobs_from_logits(
            logits, 
            input_ids, 
            target_ids, 
            target_attention_mask, 
        )

        return log_probs
    
    def logprob_from_str(
        self, 
        input_strs: List[str], 
        target_strs: List[str], 
        input_blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), 
        target_blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=None), 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        target_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> jax.Array:
        if input_token_process is None:
            input_token_process = lambda x: x
        if target_token_process is None:
            target_token_process = lambda x: x
        # tokenize
        input_tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        input_tokens = block_sequences(input_tokens, self.tokenizer.pad_token_id, np.int32, input_blocking_strategy)
        target_tokens = [target_token_process(self.tokenizer.encode(item)) for item in target_strs]
        target_tokens = block_sequences(target_tokens, self.tokenizer.pad_token_id, np.int32, target_blocking_strategy)
        # logprobs
        log_probs = self.logprob(
            jnp.asarray(input_tokens), 
            jnp.asarray(target_tokens), 
            train=train, 
            prng_key=prng_key, 
        )
        return log_probs
    
    def eval_loss(
        self, 
        input_ids: jax.Array, 
        target_ids: jax.Array, 
        input_attention_mask: Optional[jax.Array]=None, 
        input_position_ids: Optional[jax.Array]=None, 
        target_attention_mask: Optional[jax.Array]=None, 
        target_position_ids: Optional[jax.Array]=None, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        train: bool=False, 
    ) -> Tuple[jax.Array, PyTree]:
        input_attention_mask, input_position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            input_attention_mask, 
            input_position_ids, 
        )
        target_attention_mask, target_position_ids = initialize_attn_mask_pos_ids(
            target_ids, 
            self.tokenizer.pad_token_id, 
            target_attention_mask, 
            target_position_ids, 
            position_id_shift=input_position_ids.max(axis=1)+(input_attention_mask.sum(axis=1) > 0).astype(jnp.int32), 
        )

        return self._eval_loss(
            self.params, 
            input_ids, 
            input_attention_mask, 
            input_position_ids, 
            target_ids, 
            target_attention_mask, 
            target_position_ids, 
            prng_key, 
            train, 
        )
    
    def eval_loss_from_str(
        self, 
        input_strs: List[str], 
        target_strs: List[str], 
        input_blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), 
        target_blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=None), 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        train: bool=False, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        target_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> Tuple[jax.Array, PyTree]:
        if input_token_process is None:
            input_token_process = lambda x: x
        if target_token_process is None:
            target_token_process = lambda x: x
        # tokenize
        input_tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        input_tokens = block_sequences(input_tokens, self.tokenizer.pad_token_id, np.int32, input_blocking_strategy)
        target_tokens = [target_token_process(self.tokenizer.encode(item)) for item in target_strs]
        target_tokens = block_sequences(target_tokens, self.tokenizer.pad_token_id, np.int32, target_blocking_strategy)
        # loss
        return self.eval_loss(
            jnp.asarray(input_tokens), 
            jnp.asarray(target_tokens), 
            prng_key=prng_key, 
            train=train, 
        )

# train based on sequence-to-sequence input/outputs

class Train(struct.PyTreeNode):
    train_state: TrainState
    model: FlaxPreTrainedModel = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _step: Callable = struct.field(pytree_node=False)
    
    # def _step(
    #     train_state: TrainState, 
    #     input_ids: jax.Array, 
    #     input_attention_mask: jax.Array, 
    #     input_position_ids: jax.Array, 
    #     target_ids: jax.Array, 
    #     target_attention_mask: jax.Array, 
    #     target_position_ids: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray], 
    #     train: bool=True, 
    # ) -> Tuple[TrainState, jax.Array, PyTree]:
    #     raise NotImplementedError
    
    def step(
        self, 
        input_ids: jax.Array, 
        target_ids: jax.Array, 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        input_attention_mask: Optional[jax.Array]=None, 
        input_position_ids: Optional[jax.Array]=None, 
        target_attention_mask: Optional[jax.Array]=None, 
        target_position_ids: Optional[jax.Array]=None, 
        train: bool=True, 
    ) -> Tuple[Train, jax.Array, PyTree]:
        input_attention_mask, input_position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            input_attention_mask, 
            input_position_ids, 
        )
        target_attention_mask, target_position_ids = initialize_attn_mask_pos_ids(
            target_ids, 
            self.tokenizer.pad_token_id, 
            target_attention_mask, 
            target_position_ids, 
            position_id_shift=input_position_ids.max(axis=1)+(input_attention_mask.sum(axis=1) > 0).astype(jnp.int32), 
        )
        
        train_state, loss, logs = self._step(
            self.train_state, 
            input_ids, 
            input_attention_mask, 
            input_position_ids, 
            target_ids, 
            target_attention_mask, 
            target_position_ids, 
            prng_key, 
            train, 
        )
        return self.replace(train_state=train_state), loss, logs
    
    def step_from_str(
        self, 
        input_strs: List[str], 
        target_strs: List[str], 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        input_blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), 
        target_blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=None), 
        train: bool=True, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        target_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> Tuple[Train, jax.Array, PyTree]:
        if input_token_process is None:
            input_token_process = lambda x: x
        if target_token_process is None:
            target_token_process = lambda x: x
        # tokenize
        input_tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        input_tokens = block_sequences(input_tokens, self.tokenizer.pad_token_id, np.int32, input_blocking_strategy)
        target_tokens = [target_token_process(self.tokenizer.encode(item)) for item in target_strs]
        target_tokens = block_sequences(target_tokens, self.tokenizer.pad_token_id, np.int32, target_blocking_strategy)
        # step
        return self.step(
            jnp.asarray(input_tokens), 
            jnp.asarray(target_tokens), 
            prng_key, 
            train=train, 
        )

# inference based on inputs+binary mask

class InferenceMask(Inference):

    # def _eval_loss(
    #     params: PyTree, 
    #     input_ids: jax.Array, 
    #     input_attention_mask: jax.Array, 
    #     input_position_ids: jax.Array, 
    #     input_training_mask: jax.Array, # float32 0/1 indicating whether loss to apply the token (can also be a continuous weight)
    #     prng_key: Optional[jax.random.PRNGKeyArray], 
    #     train: bool=False, 
    # ) -> Tuple[jax.Array, PyTree]:
    #     raise NotImplementedError

    def eval_loss(
        self, 
        input_ids: jax.Array, 
        input_training_mask: jax.Array, 
        input_attention_mask: Optional[jax.Array]=None, 
        input_position_ids: Optional[jax.Array]=None, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        train: bool=False, 
    ) -> Tuple[jax.Array, PyTree]:
        input_attention_mask, input_position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            input_attention_mask, 
            input_position_ids, 
        )

        return self._eval_loss(
            self.params, 
            input_ids, 
            input_attention_mask, 
            input_position_ids, 
            input_training_mask, 
            prng_key, 
            train, 
        )
    
    def eval_loss_from_str_segments_list(
        self, 
        str_segments_list: List[List[Tuple[str, float]]], 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), 
        train: bool=True, 
        token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> Tuple[Train, jax.Array, PyTree]:
        if token_process is None:
            token_process = lambda x: x
        # tokenize
        in_tokens = []
        in_training_mask = []
        for segments in str_segments_list:
            sequence_tokens = []
            sequence_training_mask = []
            for segment in segments:
                segment_tokens = token_process(self.tokenizer.encode(segment[0]))
                sequence_tokens.extend(segment_tokens)
                sequence_training_mask.extend([segment[1]] * len(segment_tokens))
            in_tokens.append(sequence_tokens)
            in_training_mask.append(sequence_training_mask)
        
        in_tokens = block_sequences(
            in_tokens, 
            pad_value=self.tokenizer.pad_token_id, 
            dtype=np.int32, 
            blocking_strategy=blocking_strategy, 
        )
        in_training_mask = block_sequences(
            in_training_mask, 
            pad_value=0.0, 
            dtype=np.float32, 
            blocking_strategy=blocking_strategy, 
        )
        
        assert not np.any(in_training_mask[:, 0] > 0.0)
    
        # loss
        return self.eval_loss(
            jnp.asarray(in_tokens), 
            jnp.asarray(in_training_mask), 
            prng_key=prng_key, 
            train=train, 
        )
    
    def eval_loss_from_str(self, *args, **kwargs) -> Tuple[jax.Array, PyTree]:
        raise NotImplementedError

# train based on inputs+binary mask

class TrainMask(Train):
    train_state: TrainState
    model: FlaxPreTrainedModel = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _step: Callable = struct.field(pytree_node=False)
    
    # def _step(
    #     train_state: TrainState, 
    #     input_ids: jax.Array, 
    #     input_attention_mask: jax.Array, 
    #     input_position_ids: jax.Array, 
    #     input_training_mask: jax.Array, # float32 0/1 indicating whether loss to apply the token (can also be a continuous weight)
    #     prng_key: Optional[jax.random.PRNGKeyArray], 
    #     train: bool=True, 
    # ) -> Tuple[TrainState, jax.Array, PyTree]:
    #     raise NotImplementedError
    
    def step(
        self, 
        input_ids: jax.Array, 
        input_training_mask: jax.Array, 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        input_attention_mask: Optional[jax.Array]=None, 
        input_position_ids: Optional[jax.Array]=None, 
        train: bool=True, 
    ) -> Tuple[Train, jax.Array, PyTree]:
        assert not jnp.any(input_training_mask[:, 0] > 0.0).item(), "input_training_mask[:, 0] should be all 0s, since cannot train on first token"

        input_attention_mask, input_position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            input_attention_mask, 
            input_position_ids, 
        )
        
        train_state, loss, logs = self._step(
            self.train_state, 
            input_ids, 
            input_attention_mask, 
            input_position_ids, 
            input_training_mask, 
            prng_key, 
            train, 
        )
        return self.replace(train_state=train_state), loss, logs
    
    def step_from_str_segments_list(
        self, 
        str_segments_list: List[List[Tuple[str, float]]], 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), 
        train: bool=True, 
        token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> Tuple[Train, jax.Array, PyTree]:
        if token_process is None:
            token_process = lambda x: x
        # tokenize
        in_tokens = []
        in_training_mask = []
        for segments in str_segments_list:
            sequence_tokens = []
            sequence_training_mask = []
            for segment in segments:
                segment_tokens = token_process(self.tokenizer.encode(segment[0]))
                sequence_tokens.extend(segment_tokens)
                sequence_training_mask.extend([segment[1]] * len(segment_tokens))
            in_tokens.append(sequence_tokens)
            in_training_mask.append(sequence_training_mask)
        
        in_tokens = block_sequences(
            in_tokens, 
            pad_value=self.tokenizer.pad_token_id, 
            dtype=np.int32, 
            blocking_strategy=blocking_strategy, 
        )
        in_training_mask = block_sequences(
            in_training_mask, 
            pad_value=0.0, 
            dtype=np.float32, 
            blocking_strategy=blocking_strategy, 
        )

        assert not np.any(in_training_mask[:, 0] > 0.0)
        
        # step
        return self.step(
            jnp.asarray(in_tokens), 
            jnp.asarray(in_training_mask), 
            prng_key, 
            train=train, 
        )
    
    def step_from_str(self, *args, **kwargs) -> Tuple[Train, jax.Array, PyTree]:
        raise NotImplementedError
