from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Iterator
from JaxSeq.utils import Dataset, IterableDataset, block_sequences, BlockingStrategy, pack_sequences, MapIterable, pack_sequences_stream
import numpy as np
import jax.numpy as jnp
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Optional, Any

# dataset based on input/output sequence2sequence

class Seq2SeqDataset(Dataset):
    def __init__(self, in_tokens: np.ndarray, out_tokens: np.ndarray):
        assert in_tokens.shape[0] == out_tokens.shape[0]
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
    
    def __getitem__(self, index):
        in_tokens = self.in_tokens[index]
        out_tokens = self.out_tokens[index]
        return {
            'input_ids': jnp.asarray(in_tokens, dtype=jnp.int32), 
            'target_ids': jnp.asarray(out_tokens, dtype=jnp.int32), 
        }
    
    def __len__(self):
        return self.in_tokens.shape[0]
    
    @classmethod
    def from_str_list(
        cls, 
        str_list: List[Tuple[str, str]], 
        tokenizer: PreTrainedTokenizer, 
        in_blocking_strategy: BlockingStrategy, 
        out_blocking_strategy: BlockingStrategy, 
    ) -> Seq2SeqDataset:
        
        in_tokens = list(map(lambda x: tokenizer.encode(x[0]), str_list))
        out_tokens = list(map(lambda x: tokenizer.encode(x[1]), str_list))
        
        in_tokens = block_sequences(
            in_tokens, 
            pad_value=tokenizer.pad_token_id, 
            dtype=np.int32, 
            blocking_strategy=in_blocking_strategy, 
        )
        out_tokens = block_sequences(
            out_tokens, 
            pad_value=tokenizer.pad_token_id, 
            dtype=np.int32, 
            blocking_strategy=out_blocking_strategy, 
        )

        return cls(in_tokens, out_tokens)

class _Seq2SeqIteratorDataset:
    def __init__(self, in_out_tokens: Iterator[Tuple[np.ndarray, np.ndarray]]):
        self.in_out_tokens = in_out_tokens

    def __next__(self):
        in_tokens, out_tokens = next(self.in_out_tokens)
        return {
            'input_ids': jnp.asarray(in_tokens, dtype=jnp.int32), 
            'target_ids': jnp.asarray(out_tokens, dtype=jnp.int32), 
        }

class Seq2SeqIterableDataset(IterableDataset):
    def __init__(self, in_out_tokens: Iterable[Tuple[np.ndarray, np.ndarray]]):
        self.in_out_tokens = in_out_tokens
    
    def __iter__(self):
        return _Seq2SeqIteratorDataset(iter(self.in_out_tokens))
    
    @classmethod
    def from_str_iterable(
        cls, 
        str_iterable: Iterable[Tuple[str, str]], 
        tokenizer: PreTrainedTokenizer, 
        in_blocking_strategy: BlockingStrategy, 
        out_blocking_strategy: BlockingStrategy, 
    ) -> Seq2SeqIterableDataset:
        
        class _TokensIterable(Iterable):
            def _tokens_generator(self):
                for in_str, out_str in str_iterable:
                    in_tokens = tokenizer.encode(in_str)
                    out_tokens = tokenizer.encode(out_str)
                    
                    in_tokens = block_sequences(
                        [in_tokens], 
                        pad_value=tokenizer.pad_token_id, 
                        dtype=np.int32, 
                        blocking_strategy=in_blocking_strategy, 
                    )[0]
                    out_tokens = block_sequences(
                        [out_tokens], 
                        pad_value=tokenizer.pad_token_id, 
                        dtype=np.int32, 
                        blocking_strategy=out_blocking_strategy, 
                    )[0]

                    yield in_tokens, out_tokens

            def __iter__(self):
                return self._tokens_generator()

        return cls(_TokensIterable())

# dataset based on tokens/binary mask

class MaskDataset(Dataset):
    def __init__(self, in_tokens: np.ndarray, in_training_mask: np.ndarray):
        assert in_tokens.shape == in_training_mask.shape
        assert not np.any(in_training_mask[:, 0] > 0.0)

        self.in_tokens = in_tokens
        self.in_training_mask = in_training_mask
    
    def __getitem__(self, index):
        in_tokens = self.in_tokens[index]
        in_training_mask = self.in_training_mask[index]
        return {
            'input_ids': jnp.asarray(in_tokens, dtype=jnp.int32), 
            'input_training_mask': jnp.asarray(in_training_mask, dtype=jnp.float32), 
        }
    
    def __len__(self):
        return self.in_tokens.shape[0]
    
    @classmethod
    def blocked_from_str_segments_list(
        cls, 
        str_segments_list: List[List[Tuple[str, float]]], 
        tokenizer: PreTrainedTokenizer, 
        blocking_strategy: BlockingStrategy, 
    ) -> MaskDataset:
        
        in_tokens = []
        in_training_mask = []
        for segments in str_segments_list:
            sequence_tokens = []
            sequence_training_mask = []
            for segment in segments:
                segment_tokens = tokenizer.encode(segment[0])
                sequence_tokens.extend(segment_tokens)
                sequence_training_mask.extend([segment[1]] * len(segment_tokens))
            in_tokens.append(sequence_tokens)
            in_training_mask.append(sequence_training_mask)
        
        in_tokens = block_sequences(
            in_tokens, 
            pad_value=tokenizer.pad_token_id, 
            dtype=np.int32, 
            blocking_strategy=blocking_strategy, 
        )
        in_training_mask = block_sequences(
            in_training_mask, 
            pad_value=0.0, 
            dtype=np.float32, 
            blocking_strategy=blocking_strategy, 
        )
        return cls(in_tokens, in_training_mask)
    
    @classmethod
    def packed_from_str_segments_list(
        cls, 
        str_segments_list: List[List[Tuple[str, float]]], 
        tokenizer: PreTrainedTokenizer, 
        max_length: Optional[int], 
        truncate: bool, 
        buffer_start_str: Optional[str], 
    ) -> MaskDataset:
        
        in_tokens = []
        in_training_mask = []
        for segments in str_segments_list:
            sequence_tokens = []
            sequence_training_mask = []
            for segment in segments:
                segment_tokens = tokenizer.encode(segment[0])
                sequence_tokens.extend(segment_tokens)
                sequence_training_mask.extend([segment[1]] * len(segment_tokens))
            in_tokens.append(sequence_tokens)
            in_training_mask.append(sequence_training_mask)
        
        buffer_start_tokens = None if buffer_start_str is None else tokenizer.encode(buffer_start_str)
        in_tokens = pack_sequences(
            in_tokens, 
            dtype=np.int32, 
            max_len=max_length, 
            pad_value=None if truncate else tokenizer.pad_token_id, 
            initial_buffer=buffer_start_tokens, 
        )
        buffer_start_mask = None if buffer_start_tokens is None else [0.0]*len(buffer_start_tokens)
        in_training_mask = pack_sequences(
            in_training_mask, 
            dtype=np.float32, 
            max_len=max_length, 
            pad_value=None if truncate else 0.0, 
            initial_buffer=buffer_start_mask, 
        )
        if buffer_start_str is None:
            in_training_mask[:, 0] = 0.0 # always mask the first token
        return cls(in_tokens, in_training_mask)

class _MaskIteratorDataset:
    def __init__(self, in_mask_tokens: Iterator[Tuple[np.ndarray, np.ndarray]]):
        self.in_mask_tokens = in_mask_tokens

    def __next__(self):
        in_tokens, in_training_mask = next(self.in_mask_tokens)
        return {
            'input_ids': jnp.asarray(in_tokens, dtype=jnp.int32), 
            'input_training_mask': jnp.asarray(in_training_mask, dtype=jnp.float32), 
        }

class MaskIterableDataset(IterableDataset):
    def __init__(self, in_mask_tokens: Iterable[Tuple[np.ndarray, np.ndarray]]):
        self.in_mask_tokens = in_mask_tokens
    
    def __iter__(self):
        return _MaskIteratorDataset(iter(self.in_mask_tokens))
    
    @classmethod
    def blocked_from_str_segments_iterable(
        cls, 
        str_segments_iterable: Iterable[List[Tuple[str, float]]], 
        tokenizer: PreTrainedTokenizer, 
        blocking_strategy: BlockingStrategy, 
    ) -> MaskIterableDataset:
        
        class _TokensIterable(Iterable):
            def _tokens_generator(self):
                for segments in str_segments_iterable:
                    
                    in_tokens = []
                    in_training_mask = []
                    for segment in segments:
                        segment_tokens = tokenizer.encode(segment[0])
                        in_tokens.extend(segment_tokens)
                        in_training_mask.extend([segment[1]] * len(segment_tokens))
                    
                    in_tokens = block_sequences(
                        [in_tokens], 
                        pad_value=tokenizer.pad_token_id, 
                        dtype=np.int32, 
                        blocking_strategy=blocking_strategy, 
                    )[0]
                    in_training_mask = block_sequences(
                        [in_training_mask], 
                        pad_value=0.0, 
                        dtype=np.float32, 
                        blocking_strategy=blocking_strategy, 
                    )[0]

                    yield in_tokens, in_training_mask

            def __iter__(self):
                return self._tokens_generator()

        return cls(_TokensIterable())
    
    @classmethod
    def packed_from_str_segments_iterable(
        cls, 
        str_segments_iterable: Iterable[List[Tuple[str, float]]], 
        tokenizer: PreTrainedTokenizer, 
        max_length: Optional[int], 
        truncate: bool, 
        buffer_start_str: Optional[str], 
    ) -> MaskIterableDataset:
        class _TokensIterable(Iterable):
            def _tokens_generator(self):
                buffer_start_tokens = None if buffer_start_str is None else tokenizer.encode(buffer_start_str)
                in_tokens_stream = pack_sequences_stream(
                    MapIterable(lambda str_segments: sum(map(lambda x: tokenizer.encode(x[0]), str_segments), []), str_segments_iterable), 
                    dtype=np.int32, 
                    max_len=max_length, 
                    pad_value=None if truncate else tokenizer.pad_token_id, 
                    initial_buffer=buffer_start_tokens, 
                )
                buffer_start_mask = None if buffer_start_tokens is None else [0.0]*len(buffer_start_tokens)
                in_training_mask_stream = pack_sequences_stream(
                    MapIterable(lambda str_segments: sum(map(lambda x: [x[1]]*len(tokenizer.encode(x[0])), str_segments), []), str_segments_iterable), 
                    dtype=np.float32, 
                    max_len=max_length, 
                    pad_value=None if truncate else 0.0, 
                    initial_buffer=buffer_start_mask, 
                )

                for in_tokens, in_training_mask in zip(in_tokens_stream, in_training_mask_stream):
                    if buffer_start_str is None:
                        in_training_mask[0] = 0.0 # always mask the first token
                    yield in_tokens, in_training_mask

            def __iter__(self):
                return self._tokens_generator()

        return cls(_TokensIterable())
