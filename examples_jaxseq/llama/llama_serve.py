import tyro
from flask import Flask, request, Response
from flask_cors import CORS
from JaxSeq.models.llama.load import ModelLoadMode, load_params, load_tokenizer
from JaxSeq.models.llama.interface import LLaMAInference
from transformers import AutoTokenizer
from JaxSeq.utils import load_mesh, get_dtype, BlockingStrategy, Padding, Truncation, uuid_name, multihost_device_get
import jax
import jax.numpy as jnp
from typing import List, Dict, Any, Optional, Union, Callable, Generator
from JaxSeq.stream_tokens import StreamingGenerationConfig, TokenCallbackHandler
from JaxSeq.serve import serve_class, SSEServer
import random
import threading

# define thread safe model client

class InferenceServer:
    def __init__(self, 
        model_load_mode: ModelLoadMode, 
        model_load_path: str, 
        tokenizer_path: str, 

        data_mesh_shape: int=1, 
        fsdp_mesh_shape: int=1, 
        model_mesh_shape: int=-1, 
        dp_shard_logits: bool=True, 

        use_fp16_params: bool=True, 
        use_fp16_activations: bool=True, 
    ):
        tokenizer = load_tokenizer(
            tokenizer_path, 
            bos_token="<s>", 
            eos_token="</s>", 
            add_bos_token=False, 
            add_eos_token=False, 
        )
        tokenizer.pad_token_id = tokenizer.unk_token_id # set pad token to unk_token
        tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})

        mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
        is_main_process = jax.process_index() == 0
        print(f"Mesh: {mesh}")
        print(f"Is main process: {is_main_process}")

        model_dtype = get_dtype(use_fp16=use_fp16_activations)
        params_dtype = get_dtype(use_fp16=use_fp16_params)
        model_prng_key = jax.random.PRNGKey(0)
        params, model = load_params(
            model_load_mode=model_load_mode, 
            model_load_path=model_load_path, 
            model_dtype=model_dtype, 
            tokenizer=tokenizer, 
            mesh=mesh, 
            prng_key=model_prng_key, 
            params_dtype=params_dtype, 
        )

        self.inference = LLaMAInference.load_inference(
            params=params, 
            model=model, 
            tokenizer=tokenizer, 
            dp_shard_logits=dp_shard_logits, 
        )

        self.token_callback_handler = TokenCallbackHandler(self.inference.tokenizer)
        self.stream_child_thread = None
    
    def generate(
        self, 
        in_strs: List[str], 
        max_input_length: int, 
        rng: int, 
        streaming_callback_f: Optional[Callable[[Optional[jax.Array]], None]]=None, 
        stop_callback_f: Optional[Callable[[None], bool]]=None, 
        **generation_kwargs: Dict[str, Any], 
    ) -> Dict[str, Any]:
        try:
            if 'pad_token_id' not in generation_kwargs:
                generation_kwargs['pad_token_id'] = self.inference.tokenizer.pad_token_id
            if 'eos_token_id' not in generation_kwargs:
                generation_kwargs['eos_token_id'] = self.inference.tokenizer.eos_token_id
            return dict(
                data=self.inference.generate_from_str(
                    list(map(lambda x: self.inference.tokenizer.bos_token+x.removeprefix(self.inference.tokenizer.bos_token), in_strs)), 
                    jax.random.PRNGKey(rng), 
                    blocking_strategy=BlockingStrategy(
                        Padding.LEFT, 
                        Truncation.LEFT, 
                        max_length=max_input_length, 
                    ), 
                    generation_config=StreamingGenerationConfig(
                        **generation_kwargs, 
                        streaming_callback=streaming_callback_f, 
                        stop_callback=stop_callback_f, 
                    ), 
                ).output_strs, 
                status='success', 
            )
        except Exception as e:
            print(e)
            return dict(
                status='error', 
                error_message=str(e), 
            )
    
    def log_probs(
        self, 
        in_strs: List[str], 
        out_strs: List[str], 
        max_input_length: int, 
        max_output_length: int, 
    ) -> Dict[str, Any]:
        try:
            return dict(
                data=multihost_device_get(self.inference.logprob_from_str(
                    list(map(lambda x: self.inference.tokenizer.bos_token+x.removeprefix(self.inference.tokenizer.bos_token), in_strs)), 
                    out_strs, 
                    input_blocking_strategy=BlockingStrategy(
                        Padding.LEFT, 
                        Truncation.LEFT, 
                        max_length=max_input_length, 
                    ), 
                    target_blocking_strategy=BlockingStrategy(
                        Padding.RIGHT, 
                        Truncation.RIGHT, 
                        max_length=max_output_length, 
                    ), 
                ), mesh=self.inference.model.config.mesh).tolist(), 
                status='success', 
            )
        except Exception as e:
            print(e)
            return dict(
                status='error', 
                error_message=str(e), 
            )
    
    def generate_stream(
        self, 
        in_strs: List[str], 
        max_input_length: int, 
        rng: int, 
        **generation_kwargs: Dict[str, Any], 
    ) -> str:
        try:
            # can only run 1 stream per process, wait for previous stream to finish
            if self.stream_child_thread is not None:
                self.stream_child_thread.join()
            # add a bunch of random stuff to the seed to make it unique
            listner_channel_seed = f"{hash((tuple(in_strs), max_input_length, rng))}.{id(generation_kwargs)}.{id(self)}"
            listener_channel = self.token_callback_handler.get_listener_channel(seed=listner_channel_seed)

            thread = threading.Thread(
                target=self.generate, 
                args=(in_strs, max_input_length, rng), 
                kwargs=dict(
                    streaming_callback_f=self.token_callback_handler.get_streaming_callback_f(), 
                    stop_callback_f=self.token_callback_handler.get_stop_callback_f(), 
                    **generation_kwargs, 
                ), 
            )
            thread.start()
            self.stream_child_thread = thread
            return dict(
                data=listener_channel, 
                status='success', 
            )
        except Exception as e:
            print(e)
            return dict(
                status='error', 
                error_message=str(e), 
            )
    
    def stop_stream(self, request_id: str) -> Dict[str, Any]:
        try:
            did_stop = self.token_callback_handler.force_stop(request_id)
            return dict(
                data=did_stop, 
                status='success', 
            )
        except Exception as e:
            print(e)
            return dict(
                status='error', 
                error_message=str(e), 
            )


def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str, 
    tokenizer_path: str, 

    /,  # Mark the end of positional arguments.

    data_mesh_shape: int=1, 
    fsdp_mesh_shape: int=1, 
    model_mesh_shape: int=-1, 
    dp_shard_logits: bool=True, 

    use_fp16_params: bool=True, 

    host: str='0.0.0.0', 
    port: int=8000, 
):

    # setup app
    app = Flask(__name__)
    CORS(app)
    sse_server = SSEServer()

    InferenceServerMP = serve_class(InferenceServer)

    inference_server = InferenceServerMP(
        model_load_mode=model_load_mode, 
        model_load_path=model_load_path, 
        tokenizer_path=tokenizer_path, 

        data_mesh_shape=data_mesh_shape, 
        fsdp_mesh_shape=fsdp_mesh_shape, 
        model_mesh_shape=model_mesh_shape, 
        dp_shard_logits=dp_shard_logits, 

        use_fp16_params=use_fp16_params, 
    )

    def serve_generate():
        data = request.get_json()
        stream = data.pop('stream', False)
        if 'rng' not in data or data['rng'] is None:
            data['rng'] = random.randint(0, 2**32-1)
        generation_kwargs = data.pop('generation_kwargs', {})
        if not stream:
            return inference_server.generate(**data, **generation_kwargs)
        response = inference_server.generate_stream(**data, **generation_kwargs)
        if response['status'] == 'error':
            return response
        listener_channel = response['data']
        return Response(sse_server.listen(listener_channel), mimetype='text/event-stream')

    def serve_logprobs():
        data = request.get_json()
        return inference_server.log_probs(**data)
    
    def stop_stream():
        data = request.get_json()
        request_id = data.pop('request_id', '')
        return inference_server.stop_stream(request_id)

    app.post('/generate')(serve_generate)
    app.post('/log_probs')(serve_logprobs)
    app.post('/stop_stream')(stop_stream)

    app.run(host=host, port=port, threaded=True, processes=1)

if __name__ == "__main__":
    tyro.cli(main)
