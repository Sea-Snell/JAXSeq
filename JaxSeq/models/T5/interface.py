from __future__ import annotations
import jax
import jax.numpy as jnp
from transformers.generation import GenerationConfig
from JaxSeq.stream_tokens import StreamingGenerationConfig
from jax.sharding import PartitionSpec as PS
from jaxtyping import PyTree
from functools import partial
from typing import Optional, Union, Tuple, Callable, List
from transformers.tokenization_utils import PreTrainedTokenizerBase
from JaxSeq.utils import with_named_sharding_constraint, match_partition_rules
from optax import softmax_cross_entropy_with_integer_labels
from flax.training.train_state import TrainState
from transformers.modeling_flax_outputs import FlaxSeq2SeqLMOutput
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput, FlaxSampleOutput
from JaxSeq.models.base_interface import Inference, Train
from flax.core import FrozenDict
from jax.sharding import NamedSharding
from flax.core import freeze
from JaxSeq.utils import block_sequences, BlockingStrategy, Truncation, Padding
import numpy as np
from jax.experimental.pjit import pjit

def initialize_attn_mask_pos_ids(
    input_ids: jax.Array,  
    pad_token_id: Optional[Union[int, jax.Array]], 
    attention_mask: Optional[jax.Array]=None, 
    position_ids: Optional[jax.Array]=None, 
    is_decoder: bool=False, 
) -> Tuple[jax.Array, jax.Array]:
    if attention_mask is None:
        if pad_token_id is None:
            attention_mask = jnp.ones_like(input_ids).astype(jnp.int32)
        else:
            attention_mask = (input_ids != pad_token_id).astype(jnp.int32)
        if is_decoder:
            assert jnp.all(input_ids[:, 0] == pad_token_id), "decoder input_ids must start with pad_token_id"
            attention_mask = attention_mask.at[:, 0].set(1) # decoder starts with pad_id, make sure not to mask
    if position_ids is None:
        position_ids = jnp.maximum(jnp.cumsum(attention_mask, axis=1) - 1, 0).astype(jnp.int32)
    return attention_mask, position_ids

def loss_fn(
    model: FlaxPreTrainedModel, 
    params: PyTree, 
    input_ids: jax.Array, 
    input_attention_mask: jax.Array, 
    target_ids: jax.Array, 
    target_attention_mask: jax.Array, 
    prng_key: jax.random.PRNGKeyArray, 
    train: bool, 
) -> Tuple[jax.Array, PyTree]:
    
    model_output = model(
        input_ids=input_ids, 
        attention_mask=input_attention_mask, 
        decoder_input_ids=target_ids, 
        decoder_attention_mask=target_attention_mask, 
        params=params, 
        dropout_rng=prng_key, 
        train=train, 
    )
    
    target_logits = model_output.logits[:, :-1, :].astype(jnp.float32)
    token_losses = softmax_cross_entropy_with_integer_labels(target_logits, target_ids[:, 1:]) * target_attention_mask[:, 1:]
    loss = token_losses.sum() / target_attention_mask[:, 1:].sum()
    
    return loss, {'loss': loss}

class T5Train(Train):
    # def _step(
    #     train_state: TrainState, 
    #     input_ids: jax.Array, 
    #     input_attention_mask: jax.Array, 
    #     target_ids: jax.Array, 
    #     target_attention_mask: jax.Array, 
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
        target_attention_mask: Optional[jax.Array]=None, 
        train: bool=True, 
    ) -> Tuple[Train, jax.Array, PyTree]:
        input_attention_mask, _ = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            input_attention_mask, 
            position_ids=None, 
            is_decoder=False, 
        )
        target_attention_mask, _ = initialize_attn_mask_pos_ids(
            target_ids, 
            self.tokenizer.pad_token_id, 
            target_attention_mask, 
            position_ids=None, 
            is_decoder=True, 
        )
        
        train_state, loss, logs = self._step(
            self.train_state, 
            input_ids, 
            input_attention_mask, 
            target_ids, 
            target_attention_mask, 
            prng_key, 
            train, 
        )
        return self.replace(train_state=train_state), loss, logs
    
    @classmethod
    def load_train(
        cls, 
        train_state: TrainState, 
        model: FlaxPreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Callable=loss_fn, 
    ) -> T5Train:
        mesh = model.config.mesh
        assert mesh is not None
        train_state_partition_spec = match_partition_rules(model.config.get_partition_rules(), train_state)
        
        @partial(
            pjit, 
            donate_argnums=(0,), 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), train_state_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), train_state_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
        )
        def _step(
            train_state: TrainState, 
            input_ids: jax.Array, 
            input_attention_mask: jax.Array, 
            target_ids: jax.Array, 
            target_attention_mask: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=True, 
        ) -> Tuple[TrainState, jax.Array, PyTree]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            input_attention_mask = with_named_sharding_constraint(input_attention_mask, mesh, PS(("dp", "fsdp"), None))
            target_ids = with_named_sharding_constraint(target_ids, mesh, PS(("dp", "fsdp"), None))
            target_attention_mask = with_named_sharding_constraint(target_attention_mask, mesh, PS(("dp", "fsdp"), None))
            # NOTE: position_ids ignored by T5
            # define loss function
            def grad_loss(params: PyTree):
                loss, info = loss_fn(
                    model, 
                    params, 
                    input_ids, 
                    input_attention_mask, 
                    target_ids, 
                    target_attention_mask, 
                    prng_key, 
                    train, 
                )
                return loss, info
            train_state = train_state
            # take loss
            (loss, info), grads = jax.value_and_grad(grad_loss, has_aux=True)(train_state.params)
            # assert shard gradients
            grads = jax.tree_util.tree_map(lambda x, ps: with_named_sharding_constraint(x, mesh, ps), grads, train_state_partition_spec.params)
            # update params and optim state
            train_state = train_state.apply_gradients(grads=grads)

            return train_state, loss, info
        
        return cls(
            train_state=train_state, 
            model=model, 
            tokenizer=tokenizer, 
            _step=_step, 
        )

class T5Inference(Inference):

    # def _generate(
    #     params: PyTree, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray], 
    #     generation_config: Optional[FrozenDict]=None, 
    #     trace: bool=True, 
    # ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
    #     raise NotImplementedError
    
    # def _forward(
    #     params: PyTree, 
    #     input_ids: jax.Array, 
    #     input_attention_mask: jax.Array, 
    #     target_ids: jax.Array, 
    #     target_attention_mask: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     output_attentions: Optional[bool]=None, 
    #     output_hidden_states: Optional[bool]=None, 
    #     train: bool=False, 
    # ) -> FlaxSeq2SeqLMOutput:
    #     raise NotImplementedError

    # def _eval_loss(
    #     params: PyTree, 
    #     input_ids: jax.Array, 
    #     input_attention_mask: jax.Array, 
    #     target_ids: jax.Array, 
    #     target_attention_mask: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray], 
    #     train: bool=False, 
    # ) -> Tuple[Train, jax.Array, PyTree]:
    #     raise NotImplementedError

    @staticmethod
    @pjit
    def logprobs_from_logits(
        logits: jax.Array, 
        target_ids: jax.Array, 
        target_attention_mask: jax.Array, 
    ) -> jax.Array:
        log_logits = -softmax_cross_entropy_with_integer_labels(logits[:, :-1, :], target_ids[:, 1:])
        log_probs = jnp.where(target_attention_mask[:, 1:] == 1, log_logits, 0.0).sum(axis=1)
        return log_probs
    
    def generate(
        self, 
        input_ids: jax.Array, 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        generation_config: Optional[GenerationConfig]=None, 
        attention_mask: Optional[jax.Array]=None, 
        trace: bool=True, 
    ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
        attention_mask, _ = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids=None, 
            is_decoder=False, 
        )

        return self._generate(
            self.params, 
            input_ids, 
            attention_mask, 
            prng_key, 
            freeze(generation_config.to_dict()) if generation_config is not None else None, 
            trace, 
        )

    def forward(
        self, 
        input_ids: jax.Array, 
        target_ids: jax.Array, 
        input_attention_mask: Optional[jax.Array]=None, 
        target_attention_mask: Optional[jax.Array]=None, 
        output_attentions: Optional[bool]=None, 
        output_hidden_states: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    ) -> FlaxSeq2SeqLMOutput:
        input_attention_mask, _ = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            input_attention_mask, 
            position_ids=None, 
            is_decoder=False, 
        )
        target_attention_mask, _ = initialize_attn_mask_pos_ids(
            target_ids, 
            self.tokenizer.pad_token_id, 
            target_attention_mask, 
            position_ids=None, 
            is_decoder=True, 
        )
        
        return self._forward(
            self.params, 
            input_ids, 
            input_attention_mask, 
            target_ids, 
            target_attention_mask, 
            prng_key, 
            output_attentions, 
            output_hidden_states, 
            train, 
        )
    
    def forward_from_str(
        self, 
        input_strs: List[str], 
        target_strs: List[str], 
        input_blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.LEFT, max_length=None), 
        target_blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=None), 
        output_attentions: Optional[bool]=None, 
        output_hidden_states: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        target_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> FlaxSeq2SeqLMOutput:
        if input_token_process is None:
            input_token_process = lambda x: x
        # tokenize
        input_tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        input_tokens = block_sequences(input_tokens, self.tokenizer.pad_token_id, np.int32, input_blocking_strategy)
        target_tokens = [target_token_process(self.tokenizer.encode(item)) for item in target_strs]
        target_tokens = block_sequences(target_tokens, self.tokenizer.pad_token_id, np.int32, target_blocking_strategy)
        # forward
        outputs = self.forward(
            jnp.asarray(input_tokens), 
            jnp.asarray(target_tokens), 
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
        target_attention_mask: Optional[jax.Array]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    ) -> jax.Array:
        input_attention_mask, _ = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            input_attention_mask, 
            position_ids=None, 
            is_decoder=False, 
        )
        target_attention_mask, _ = initialize_attn_mask_pos_ids(
            target_ids, 
            self.tokenizer.pad_token_id, 
            target_attention_mask, 
            position_ids=None, 
            is_decoder=True, 
        )

        logits = self.forward(
            input_ids, 
            target_ids, 
            input_attention_mask=input_attention_mask, 
            target_attention_mask=target_attention_mask, 
            train=train, 
            prng_key=prng_key, 
        ).logits
        
        log_probs = self.logprobs_from_logits(
            logits, 
            target_ids, 
            target_attention_mask, 
        )

        return log_probs
    
    def eval_loss(
        self, 
        input_ids: jax.Array, 
        target_ids: jax.Array, 
        input_attention_mask: Optional[jax.Array]=None, 
        target_attention_mask: Optional[jax.Array]=None, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        train: bool=False, 
    ) -> Tuple[jax.Array, PyTree]:
        input_attention_mask, _ = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            input_attention_mask, 
            position_ids=None, 
            is_decoder=False, 
        )
        target_attention_mask, _ = initialize_attn_mask_pos_ids(
            target_ids, 
            self.tokenizer.pad_token_id, 
            target_attention_mask, 
            position_ids=None, 
            is_decoder=True, 
        )

        return self._eval_loss(
            self.params, 
            input_ids, 
            input_attention_mask, 
            target_ids, 
            target_attention_mask, 
            prng_key, 
            train, 
        )


    @classmethod
    def load_inference(
        cls, 
        params: PyTree, 
        model: FlaxPreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Optional[Callable]=loss_fn, 
        dp_shard_logits: bool=True, 
    ) -> T5Inference:
        mesh = model.config.mesh
        assert mesh is not None
        params_partition_spec = match_partition_rules(model.config.get_partition_rules(), params)

        @partial(
            pjit, 
            static_argnames=('generation_config', 'trace'), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=NamedSharding(mesh, PS()), 
        )
        def _generate(
            params: PyTree, 
            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray], 
            generation_config: Optional[FrozenDict]=None, 
            trace: bool=True, 
        ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(("dp", "fsdp"), None))

            # generate from model
            output = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                params=params, 
                prng_key=prng_key, 
                generation_config=StreamingGenerationConfig.from_dict(generation_config) if generation_config is not None else None, 
                trace=trace, 
            )
            
            return output
    
        @partial(
            pjit, 
            static_argnames=('output_attentions', 'output_hidden_states', 'train'), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=FlaxSeq2SeqLMOutput(
                logits=NamedSharding(mesh, PS(("dp", "fsdp"), None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                # assume no sharding for everything else
                past_key_values=NamedSharding(mesh, PS()), 
                decoder_hidden_states=NamedSharding(mesh, PS()), 
                decoder_attentions=NamedSharding(mesh, PS()), 
                cross_attentions=NamedSharding(mesh, PS()), 
                encoder_last_hidden_state=NamedSharding(mesh, PS()), 
                encoder_hidden_states=NamedSharding(mesh, PS()), 
                encoder_attentions=NamedSharding(mesh, PS()), 
            )
        )
        def _forward(
            params: PyTree, 
            input_ids: jax.Array, 
            input_attention_mask: jax.Array, 
            target_ids: jax.Array, 
            target_attention_mask: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray]=None, 
            output_attentions: Optional[bool]=None, 
            output_hidden_states: Optional[bool]=None, 
            train: bool=False, 
        ) -> FlaxSeq2SeqLMOutput:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            input_attention_mask = with_named_sharding_constraint(input_attention_mask, mesh, PS(("dp", "fsdp"), None))
            target_ids = with_named_sharding_constraint(target_ids, mesh, PS(("dp", "fsdp"), None))
            target_attention_mask = with_named_sharding_constraint(target_attention_mask, mesh, PS(("dp", "fsdp"), None))

            # get logits
            output = model(
                input_ids=input_ids, 
                attention_mask=input_attention_mask, 
                decoder_input_ids=target_ids, 
                decoder_attention_mask=target_attention_mask, 
                params=params, 
                train=train, 
                output_attentions=output_attentions, 
                output_hidden_states=output_hidden_states, 
                dropout_rng=prng_key, 
            )
            # trunc padded logits
            output = output.replace(logits=output.logits.at[:, :, model.config.unpadded_vocab_size:].set(-float('inf')))

            # assert sharding on outputs
            if dp_shard_logits:
                output = output.replace(logits=with_named_sharding_constraint(output.logits, mesh, PS(("dp", "fsdp"), None, None)))
            return output
        
        @partial(
            pjit, 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=(
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
        )
        def _eval_loss(
            params: PyTree, 
            input_ids: jax.Array, 
            input_attention_mask: jax.Array, 
            target_ids: jax.Array, 
            target_attention_mask: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=False, 
        ) -> Tuple[jax.Array, PyTree]:
            assert loss_fn is not None, "loss_fn must be set to use eval_loss"
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            input_attention_mask = with_named_sharding_constraint(input_attention_mask, mesh, PS(("dp", "fsdp"), None))
            target_ids = with_named_sharding_constraint(target_ids, mesh, PS(("dp", "fsdp"), None))
            target_attention_mask = with_named_sharding_constraint(target_attention_mask, mesh, PS(("dp", "fsdp"), None))

            # define loss function
            loss, info = loss_fn(
                model, 
                params, 
                input_ids, 
                input_attention_mask, 
                target_ids, 
                target_attention_mask, 
                prng_key, 
                train, 
            )
            return loss, info
        
        return cls(
            params=params, 
            model=model, 
            tokenizer=tokenizer, 
            _generate=_generate, 
            _forward=_forward, 
            _eval_loss=None if loss_fn is None else _eval_loss, 
        )

