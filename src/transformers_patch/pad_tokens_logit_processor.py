from transformers.generation_flax_logits_process import FlaxLogitsProcessor
import jax.numpy as jnp
import jax

class FlaxTokenPaddingLogitProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] sets probability of all padded logits to 0.
    Args:
        n_tokens (`int`):
            The number of actual tokens.
    """

    def __init__(self, n_tokens: int):
        if not isinstance(n_tokens, int) or n_tokens < 0:
            raise ValueError(f"`n_tokens` has to be a positive integer, but is {n_tokens}")

        self.n_tokens = n_tokens

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        scores = jax.lax.dynamic_update_slice(scores, jnp.full((scores.shape[0], scores.shape[1]-self.n_tokens), -float("inf"), dtype=scores.dtype), (0, self.n_tokens))
        return scores
