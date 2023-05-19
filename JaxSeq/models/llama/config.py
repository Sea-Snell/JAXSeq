# coding=utf-8
# Copyright 2022 The FAIR team of Meta AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" LLaMA model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import Optional, Any, Dict
import jax
import re
from jax.sharding import PartitionSpec as PS

logger = logging.get_logger(__name__)

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class LLaMAConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~LLaMAModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~LLaMAModel`] or [`~TFLLaMAModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_sequence_length (`int`, *optional*, defaults to 2048):
            Max sequence length for model (for RoPE computation)
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        Example:
    ```python
    >>> from transformers import LLaMAModel, LLaMAConfig
    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LLaMAConfig()
    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LLaMAModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "llama"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_sequence_length=2048,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=-1,
        bos_token_id=1,
        eos_token_id=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=False,
        gradient_checkpointing=True, 
        gradient_checkpointing_policy='nothing_saveable', 
        unpadded_vocab_size=None, 
        mesh: Optional[jax.sharding.Mesh]=None, 
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_policy = gradient_checkpointing_policy
        self.unpadded_vocab_size = unpadded_vocab_size
        if self.unpadded_vocab_size is None:
            self.unpadded_vocab_size = self.vocab_size
        self.mesh = mesh
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs, 
        )
    
    @staticmethod
    def get_partition_rules():
        return [
            # embeddings
            (re.escape("['transformer']['wte']['embedding']"), PS("mp", "fsdp")), 
            # self atention
            (''.join((re.escape("['attention']"), r"\['(wk|wq|wv)'\]", re.escape("['kernel']"))), PS("fsdp", "mp")), 
            (re.escape("['attention']['wo']['kernel']"), PS("mp", "fsdp")), 
            # mlp
            (re.escape("['feed_forward']['w1']['kernel']"), PS("fsdp", "mp")), 
            (re.escape("['feed_forward']['w2']['kernel']"), PS("mp", "fsdp")), 
            (re.escape("['feed_forward']['w3']['kernel']"), PS("fsdp", "mp")), 
            # layer norms
            (re.escape("['attention_norm']['kernel']"), PS()), 
            (re.escape("['ffn_norm']['kernel']"), PS()), 
            (re.escape("['transformer']['ln_f']['kernel']"), PS()), 
            # output head
            (re.escape("['lm_head']['kernel']"), PS("fsdp", "mp")), 
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        if self.mesh is None:
            return super().to_dict()
        else:
            new_conf = LLaMAConfig(**self.__dict__)
            new_conf.mesh = None
            return new_conf.to_dict()
