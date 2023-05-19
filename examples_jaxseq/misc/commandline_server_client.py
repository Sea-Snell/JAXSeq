from enum import Enum
from typing import List, Any, Optional, Dict, Union
import requests
import json
import tyro
from sseclient import SSEClient
from JaxSeq.stream_tokens import map_token_stream
from urllib.parse import urljoin
import signal
import sys
import time
import threading
import random
from JaxSeq.utils import strip_prompt_from_completion

# commandline client for interacting with a model webserver

class Client:
    def __init__(self, hosts: List[str]):
        self.hosts = hosts

    def request(self, path: str, **kwargs):
        thread_results = {}
        def _request(host: str):
            nonlocal thread_results
            thread_results[host] = requests.post(urljoin(host, path), **kwargs)
        threads = [threading.Thread(target=_request, args=(host,)) for host in self.hosts]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return thread_results[self.hosts[0]]

    def generate(
        self, 
        prompts: List[str], 
        seed: Optional[int]=None, 
        max_input_length: int=512, 
		max_new_tokens: int=512, 
        do_sample: bool=True, 
        num_beams: int=1, 
        temperature: Optional[float]=None, 
        top_p: Optional[float]=None, 
        top_k: Optional[int]=None, 
        eos_token_id: Optional[int]=None, 
    ) -> Dict[str, Any]:
        if seed is None:
            seed = random.randint(0, 2**31-1)
        response = self.request(
            'generate', 
            json={
                'in_strs': prompts, 
                'max_input_length': max_input_length, 
				'rng': seed, 
                'generation_kwargs': {
                    'do_sample': do_sample, 
                    'num_beams': num_beams, 
					'max_new_tokens': max_new_tokens, 
                    'temperature': temperature, 
                    'top_p': top_p, 
                    'top_k': top_k, 
                    'eos_token_id': eos_token_id, 
                }, 
            }, 
        )
        try:
            response = response.json()
        except Exception as e:
            return dict(
                status='error', 
                error_message=str(e), 
            )
        return response
    
    def generate_stream(
        self, 
        prompts: List[str], 
        seed: Optional[int]=None, 
        max_input_length: int=512, 
		max_new_tokens: int=512, 
        do_sample: bool=True, 
        num_beams: int=1, 
        temperature: Optional[float]=None, 
        top_p: Optional[float]=None, 
        top_k: Optional[int]=None, 
        eos_token_id: Optional[int]=None, 
    ) -> Union[SSEClient, str]:
        if seed is None:
            seed = random.randint(0, 2**31-1)
        response = self.request(
            'generate', 
            json={
                'in_strs': prompts, 
                'max_input_length': max_input_length, 
				'rng': seed, 
                'generation_kwargs': {
                    'do_sample': do_sample, 
                    'num_beams': num_beams, 
					'max_new_tokens': max_new_tokens, 
                    'temperature': temperature, 
                    'top_p': top_p, 
                    'top_k': top_k, 
                    'eos_token_id': eos_token_id, 
                }, 
                'stream': True, 
            }, 
            stream=True, 
            headers={'Accept': 'text/event-stream'}, 
        )
        if 'text/event-stream' not in response.headers['Content-Type']:   
            # if not an event stream, it is an error message
            try:
                response = response.json()
            except Exception as e:
                return str(e)
            if response['status'] == 'error':
                return response['error_message']
            else:
                raise NotImplementedError
        client = SSEClient(response)
        return client
    
    def log_probs(
        self, 
        in_strs: List[str], 
        out_strs: List[str], 
		max_input_length: int=512, 
        max_output_length: int=512, 
    ) -> Dict[str, Any]:
        response = self.request(
            'log_probs', 
            json={
                'in_strs': in_strs, 
                'out_strs': out_strs, 
			    'max_input_length': max_input_length, 
				'max_output_length': max_output_length, 
            }, 
        )
        try:
            response = response.json()
        except Exception as e:
            return dict(
                status='error', 
                error_message=str(e), 
            )
        return response
    
    def stop_stream(
        self, 
        request_id: str, 
    ) -> Dict[str, Any]:
        response = self.request(
            'stop_stream', 
            json={
                'request_id': request_id, 
            }, 
        )
        try:
            response = response.json()
            return response
        except Exception as e:
            return dict(
                status='error', 
                error_message=str(e), 
            )

class FunctionOption(Enum):
    GENERATE: str = 'generate'
    LOGPROBS: str = 'log_probs'
    GENERATE_NO_STREAM: str = 'generate_no_stream'

def main(
    function: FunctionOption, 

    /,  # Mark the end of positional arguments.

    host: Union[str, List[str]]='http://127.0.0.1:8000/', 

    max_input_length: int=512, 
    max_output_length: int=512, 

    # generation settings, only for generate function
    remove_prompt_from_generation: bool=False, 
    seed: Optional[int]=None, 
    do_sample: bool=True, 
    num_beams: int=1, 
    temperature: Optional[float]=None, 
    top_p: Optional[float]=None, 
    top_k: Optional[float]=None, 
    eos_token_id: Optional[int]=None, 
):
    if isinstance(host, str):
        host = [host]
    client = Client(hosts=host)

    if function is FunctionOption.GENERATE:
        while True:
            prompt = input('>>> ')
            if prompt == '__exit__':
                break

            request_id = ''
            recieved_first_tokens = False
            
            sse_client = client.generate_stream(
                [prompt], 
                seed=seed, 
                max_input_length=max_input_length, 
                max_new_tokens=max_output_length, 
                do_sample=do_sample, 
                num_beams=num_beams, 
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k, 
                eos_token_id=eos_token_id, 
            )
            if isinstance(sse_client, str):
                print('__ERROR__:', sse_client)
                continue

            def handle_stop(sig, frame):
                response = client.stop_stream(request_id)
                if response['status'] == 'error':
                    print('\n__ERROR__:', response['error_message'])
                elif response['status'] != 'success':
                    raise NotImplementedError

            def token_stream_read(curr_request_id: str, full_texts: List[str], next_tokens: List[str]):
                nonlocal request_id
                nonlocal recieved_first_tokens
                request_id = curr_request_id
                if  (not recieved_first_tokens) and remove_prompt_from_generation:
                    next_tokens[0] = strip_prompt_from_completion(prompt, next_tokens[0])
                print(next_tokens[0], end='', flush=True)
                recieved_first_tokens = True
            
            signal.signal(signal.SIGINT, handle_stop)

            # print token stream
            map_token_stream(
                sse_client, 
                callback=token_stream_read, 
                error_callback=lambda e: print('\n__ERROR__:', e), 
            )
            print()

    elif function is FunctionOption.GENERATE_NO_STREAM:
         while True:
            prompt = input('>>> ')
            if prompt == '__exit__':
                break

            generation = client.generate(
                [prompt], 
                seed=seed, 
                max_input_length=max_input_length, 
                max_new_tokens=max_output_length, 
                do_sample=do_sample, 
                num_beams=num_beams, 
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k, 
                eos_token_id=eos_token_id, 
            )

            if generation['status'] == 'error':
                print('__ERROR__:', generation['error_message'])
            elif generation['status'] == 'success':
                print(generation['data'][0])
            else:
                raise NotImplementedError
    
    elif function is FunctionOption.LOGPROBS:
        while True: 
            prompt = input('[ prompt ] >>> ')
            if prompt == '__exit__':
                break
            completion = input('[ completion ] >>> ')
            if completion == '__exit__':
                break

            log_probs = client.log_probs(
                [prompt], 
                [completion], 
                max_input_length=max_input_length, 
                max_output_length=max_output_length, 
            )

            if log_probs['status'] == 'error':
                print('__ERROR__:', log_probs['error_message'])
            elif log_probs['status'] == 'success':
                print(log_probs['data'][0])
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError

if __name__ == "__main__":
    # `python commandline_server_client.py -h` to see all commandline options
    tyro.cli(main)
