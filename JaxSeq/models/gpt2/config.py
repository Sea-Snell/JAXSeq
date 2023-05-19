""" GPT-2 model configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from jax.sharding import PartitionSpec as PS
import re
import jax
from typing import Optional, Dict, Any

logger = logging.get_logger(__name__)

class GPT2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`GPT2Model`] or a [`TFGPT2Model`]. It is used to
    instantiate a GPT-2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPT-2
    [gpt2](https://huggingface.co/gpt2) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPT2Model`] or [`TFGPT2Model`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`string`, *optional*, defaults to `"cls_index"`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].
            Has to be one of the following options:
                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].
            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            [`GPT2DoubleHeadsModel`].
            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].
            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].
            The dropout ratio to be used after the projection and activation.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.
    Example:
    ```python
    >>> from transformers import GPT2Model, GPT2Config
    >>> # Initializing a GPT2 configuration
    >>> configuration = GPT2Config()
    >>> # Initializing a model from the configuration
    >>> model = GPT2Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        gradient_checkpointing=True, 
        gradient_checkpointing_policy='nothing_saveable', 
        unpadded_vocab_size=None, 
        mesh: Optional[jax.sharding.Mesh]=None, 
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_policy = gradient_checkpointing_policy
        self.unpadded_vocab_size = unpadded_vocab_size
        if self.unpadded_vocab_size is None:
            self.unpadded_vocab_size = self.vocab_size

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mesh = mesh

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @staticmethod
    def get_partition_rules():
        return [
            # embeddings
            (re.escape("['transformer']['wpe']['embedding']"), PS("mp", "fsdp")), 
            (re.escape("['transformer']['wte']['embedding']"), PS("mp", "fsdp")), 
            # atention
            (''.join((re.escape("['attn']"), r"\['(q_attn|c_attn)'\]", re.escape("['kernel']"))), PS("fsdp", "mp")), 
            (''.join((re.escape("['attn']"), r"\['(q_attn|c_attn)'\]", re.escape("['bias']"))), PS("mp")), 
            (re.escape("['attn']['c_proj']['kernel']"), PS("mp", "fsdp")), 
            (re.escape("['attn']['c_proj']['bias']"), PS()), 
            # mlp
            (re.escape("['mlp']['c_fc']['kernel']"), PS("fsdp", "mp")), 
            (re.escape("['mlp']['c_fc']['bias']"), PS("mp")), 
            (re.escape("['mlp']['c_proj']['kernel']"), PS("mp", "fsdp")), 
            (re.escape("['mlp']['c_proj']['bias']"), PS()), 
            # layer norms
            ((r"\['ln_\d+'\]\['bias'\]"), PS()), 
            ((r"\['ln_\d+'\]\['scale'\]"), PS()), 
            (re.escape("['ln_f']['bias']"), PS()), 
            (re.escape("['ln_f']['scale']"), PS()), 
            # # output head
            (re.escape("['lm_head']['kernel']"), PS("fsdp", "mp")), 
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        if self.mesh is None:
            return super().to_dict()
        else:
            new_conf = GPT2Config(**self.__dict__)
            new_conf.mesh = None
            return new_conf.to_dict()
