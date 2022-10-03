from typing import Dict, Iterable, List, Tuple
from data import Dataset, IterableDataset, block_sequences
import numpy as np
import jax.numpy as jnp
from transformers.tokenization_utils import PreTrainedTokenizer

class Seq2SeqDataset(Dataset):
    def __init__(self, in_tokens: np.ndarray, out_tokens: np.ndarray):
        assert in_tokens.shape[0] == out_tokens.shape[0]
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        assert in_tokens.shape[0] == len(self.meta)
    
    def __getitem__(self, index):
        in_tokens = self.in_tokens[index]
        out_tokens = self.out_tokens[index]
        return jnp.asarray(in_tokens, dtype=jnp.int32), jnp.asarray(out_tokens, dtype=jnp.int32)
    
    def __len__(self):
        return self.in_tokens.shape[0]
    
    @classmethod
    def from_str_list(
        cls, 
        str_list: List[Tuple[str, str]], 
        tokenizer: PreTrainedTokenizer, 
        max_input_length: int,
        max_output_length: int, 
        pad_inputs_right: bool=True, 
        pad_outputs_right: bool=True, 
        trunc_inputs_last: bool=True, 
        trunc_outputs_last: bool=True, 
    ) -> 'Seq2SeqDataset':
        
        in_tokens = list(map(lambda x: tokenizer.encode(x[0]), str_list))
        out_tokens = list(map(lambda x: tokenizer.encode(x[1]), str_list))
        
        in_tokens = block_sequences(
            in_tokens, 
            max_len=max_input_length, 
            pad_right=tokenizer.pad_token_id, 
            pad_right=pad_inputs_right, 
            trunc_last=trunc_inputs_last, 
        )
        out_tokens = block_sequences(
            out_tokens, 
            max_len=max_output_length, 
            pad_right=tokenizer.pad_token_id, 
            pad_right=pad_outputs_right, 
            trunc_last=trunc_outputs_last, 
        )

        return cls(in_tokens, out_tokens)

class Seq2SeqIterableDataset(IterableDataset):
    def __init__(self, in_out_tokens: Iterable[Tuple[np.ndarray, np.ndarray]]):
        self.in_out_tokens = in_out_tokens
    
    def __iter__(self):
        return self
    
    def __next__(self):
        in_tokens, out_tokens = next(self.in_out_tokens)
        return jnp.asarray(in_tokens, dtype=jnp.int32), jnp.asarray(out_tokens, dtype=jnp.int32)
    
    @classmethod
    def from_str_iterable(
        cls, 
        str_iterable: Iterable[Tuple[str, str]], 
        tokenizer: PreTrainedTokenizer, 
        max_input_length: int,
        max_output_length: int, 
        pad_inputs_right: bool=True, 
        pad_outputs_right: bool=True, 
        trunc_inputs_last: bool=True, 
        trunc_outputs_last: bool=True, 
    ):
        def _in_out_tokens():
            in_str, out_str = next(str_iterable)
            in_tokens = tokenizer.encode(in_str)
            out_tokens = tokenizer.encode(out_str)
            
            in_tokens = block_sequences(
                [in_tokens], 
                max_len=max_input_length, 
                pad_right=tokenizer.pad_token_id, 
                pad_right=pad_inputs_right, 
                trunc_last=trunc_inputs_last, 
            )[0]
            out_tokens = block_sequences(
                [out_tokens], 
                max_len=max_output_length, 
                pad_right=tokenizer.pad_token_id, 
                pad_right=pad_outputs_right, 
                trunc_last=trunc_outputs_last, 
            )[0]

            return in_tokens, out_tokens

        return cls(_in_out_tokens)
