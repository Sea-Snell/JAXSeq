from __future__ import annotations
import jax
import jax.numpy as jnp
from JaxSeq.stream_tokens import StreamingGenerationConfig
from jax.sharding import PartitionSpec as PS
from jaxtyping import PyTree
from functools import partial
from typing import Optional, Union, Tuple, Callable
from transformers.tokenization_utils import PreTrainedTokenizerBase
from JaxSeq.utils import with_named_sharding_constraint, match_partition_rules
from optax import softmax_cross_entropy_with_integer_labels
from flax.training.train_state import TrainState
from transformers.modeling_flax_outputs import FlaxCausalLMOutput, FlaxCausalLMOutputWithCrossAttentions
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput, FlaxSampleOutput
from JaxSeq.models.base_interface import Inference, Train, TrainMask, InferenceMask
from flax.core import FrozenDict
from jax.sharding import NamedSharding
from jax.experimental.pjit import pjit

def loss_fn(
    model: FlaxPreTrainedModel, 
    params: PyTree, 
    input_ids: jax.Array, 
    input_attention_mask: jax.Array, 
    input_position_ids: jax.Array, 
    target_ids: jax.Array, 
    target_attention_mask: jax.Array, 
    target_position_ids: jax.Array, 
    prng_key: jax.random.PRNGKeyArray, 
    train: bool, 
) -> Tuple[jax.Array, PyTree]:
    
    full_input_ids = jnp.concatenate((input_ids, target_ids,), axis=1)
    full_attn_mask = jnp.concatenate((input_attention_mask, target_attention_mask), axis=1)
    full_position_ids = jnp.concatenate((input_position_ids, target_position_ids), axis=1)
    
    model_output = model(
        input_ids=full_input_ids, 
        attention_mask=full_attn_mask, 
        position_ids=full_position_ids, 
        params=params, 
        dropout_rng=prng_key, 
        train=train, 
    )
    
    target_logits = model_output.logits[:, (input_ids.shape[1]-1):-1, :].astype(jnp.float32)
    token_losses = softmax_cross_entropy_with_integer_labels(target_logits, target_ids) * target_attention_mask
    loss = token_losses.sum() / target_attention_mask.sum()
    
    return loss, {'loss': loss}

def loss_fn_mask(
    model: FlaxPreTrainedModel, 
    params: PyTree, 
    input_ids: jax.Array, 
    input_attention_mask: jax.Array, 
    input_position_ids: jax.Array, 
    input_training_mask: jax.Array, 
    prng_key: jax.random.PRNGKeyArray, 
    train: bool, 
) -> Tuple[jax.Array, PyTree]:
    model_output = model(
        input_ids=input_ids, 
        attention_mask=input_attention_mask, 
        position_ids=input_position_ids, 
        params=params, 
        dropout_rng=prng_key, 
        train=train, 
    )
    
    logits = model_output.logits[:, :-1, :].astype(jnp.float32)
    target_ids = input_ids[:, 1:]
    mask = input_attention_mask[:, 1:] * input_training_mask[:, 1:]
    token_losses = softmax_cross_entropy_with_integer_labels(logits, target_ids) * mask
    loss = token_losses.sum() / mask.sum()
    
    return loss, {'loss': loss}

class GPT2Train(Train):
    @classmethod
    def load_train(
        cls, 
        train_state: TrainState, 
        model: FlaxPreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Callable=loss_fn, 
    ) -> GPT2Train:
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
            input_position_ids: jax.Array, 
            target_ids: jax.Array, 
            target_attention_mask: jax.Array, 
            target_position_ids: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=True, 
        ) -> Tuple[TrainState, jax.Array, PyTree]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            input_attention_mask = with_named_sharding_constraint(input_attention_mask, mesh, PS(("dp", "fsdp"), None))
            input_position_ids = with_named_sharding_constraint(input_position_ids, mesh, PS(("dp", "fsdp"), None))
            target_ids = with_named_sharding_constraint(target_ids, mesh, PS(("dp", "fsdp"), None))
            target_attention_mask = with_named_sharding_constraint(target_attention_mask, mesh, PS(("dp", "fsdp"), None))
            target_position_ids = with_named_sharding_constraint(target_position_ids, mesh, PS(("dp", "fsdp"), None))

            # define loss function
            def grad_loss(params: PyTree):
                loss, info = loss_fn(
                    model, 
                    params, 
                    input_ids, 
                    input_attention_mask, 
                    input_position_ids, 
                    target_ids, 
                    target_attention_mask, 
                    target_position_ids, 
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

class GPT2TrainMask(TrainMask):
    @classmethod
    def load_train(
        cls, 
        train_state: TrainState, 
        model: FlaxPreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Callable=loss_fn_mask, 
    ) -> GPT2TrainMask:
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
            input_position_ids: jax.Array, 
            input_training_mask: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=True, 
        ) -> Tuple[TrainState, jax.Array, PyTree]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            input_attention_mask = with_named_sharding_constraint(input_attention_mask, mesh, PS(("dp", "fsdp"), None))
            input_position_ids = with_named_sharding_constraint(input_position_ids, mesh, PS(("dp", "fsdp"), None))
            input_training_mask = with_named_sharding_constraint(input_training_mask, mesh, PS(("dp", "fsdp"), None))

            # define loss function
            def grad_loss(params: PyTree):
                loss, info = loss_fn(
                    model, 
                    params, 
                    input_ids, 
                    input_attention_mask, 
                    input_position_ids, 
                    input_training_mask, 
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

class GPT2Inference(Inference):
    @classmethod
    def load_inference(
        cls, 
        params: PyTree, 
        model: FlaxPreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Optional[Callable]=loss_fn, 
        dp_shard_logits: bool=True, 
    ) -> GPT2Inference:
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
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=NamedSharding(mesh, PS()), 
        )
        def _generate(
            params: PyTree, 
            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray], 
            generation_config: Optional[FrozenDict]=None, 
            trace: bool=True, 
        ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(("dp", "fsdp"), None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS(("dp", "fsdp"), None))
            # NOTE: position_ids ignored by transformers

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
            ), 
            out_shardings=FlaxCausalLMOutputWithCrossAttentions(
                logits=NamedSharding(mesh, PS(("dp", "fsdp"), None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                past_key_values=NamedSharding(mesh, PS()), # assume no sharding for past key values
                hidden_states=NamedSharding(mesh, PS()), # assume no sharding for hidden states
                attentions=NamedSharding(mesh, PS()), # assume no sharding for attentions
                cross_attentions=NamedSharding(mesh, PS()), # assume no sharding for cross attentions
            )
        )
        def _forward(
            params: PyTree, 
            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray]=None, 
            output_attentions: Optional[bool]=None, 
            output_hidden_states: Optional[bool]=None, 
            train: bool=False, 
        ) -> FlaxCausalLMOutputWithCrossAttentions:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(("dp", "fsdp"), None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS(("dp", "fsdp"), None))

            # get logits
            output = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
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
            input_position_ids: jax.Array, 
            target_ids: jax.Array, 
            target_attention_mask: jax.Array, 
            target_position_ids: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=False, 
        ) -> Tuple[jax.Array, PyTree]:
            assert loss_fn is not None, "loss_fn must be set to use eval_loss"
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            input_attention_mask = with_named_sharding_constraint(input_attention_mask, mesh, PS(("dp", "fsdp"), None))
            input_position_ids = with_named_sharding_constraint(input_position_ids, mesh, PS(("dp", "fsdp"), None))
            target_ids = with_named_sharding_constraint(target_ids, mesh, PS(("dp", "fsdp"), None))
            target_attention_mask = with_named_sharding_constraint(target_attention_mask, mesh, PS(("dp", "fsdp"), None))
            target_position_ids = with_named_sharding_constraint(target_position_ids, mesh, PS(("dp", "fsdp"), None))

            # define loss function
            loss, info = loss_fn(
                model, 
                params, 
                input_ids, 
                input_attention_mask, 
                input_position_ids, 
                target_ids, 
                target_attention_mask, 
                target_position_ids, 
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

class GPT2InferenceMask(InferenceMask):
    @classmethod
    def load_inference(
        cls, 
        params: PyTree, 
        model: FlaxPreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Optional[Callable]=loss_fn_mask, 
        dp_shard_logits: bool=True, 
    ) -> GPT2InferenceMask:
        mesh = model.config.mesh
        assert mesh is not None
        temp_inference = GPT2Inference.load_inference(params, model, tokenizer, loss_fn=None, dp_shard_logits=dp_shard_logits)
        params_partition_spec = match_partition_rules(model.config.get_partition_rules(), params)

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
            input_position_ids: jax.Array, 
            input_training_mask: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=False, 
        ) -> Tuple[jax.Array, PyTree]:
            assert loss_fn is not None, "loss_fn must be set to use eval_loss"
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            input_attention_mask = with_named_sharding_constraint(input_attention_mask, mesh, PS(("dp", "fsdp"), None))
            input_position_ids = with_named_sharding_constraint(input_position_ids, mesh, PS(("dp", "fsdp"), None))
            input_training_mask = with_named_sharding_constraint(input_training_mask, mesh, PS(("dp", "fsdp"), None))

            # define loss function
            loss, info = loss_fn(
                model, 
                params, 
                input_ids, 
                input_attention_mask, 
                input_position_ids, 
                input_training_mask, 
                prng_key, 
                train, 
            )
            return loss, info
        
        return cls(
            params=params, 
            model=model, 
            tokenizer=tokenizer, 
            _generate=temp_inference._generate, 
            _forward=temp_inference._forward, 
            _eval_loss=_eval_loss, 
        )
