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
from examples_jaxseq.misc.commandline_server_client import Client

# commandline client for interacting with koala chat model webserver

class KoalaChatClient:
    def __init__(self, hosts: List[str], sep_turn_text: str='</s>'):
        self.client = Client(hosts=hosts)
        self.user_messages = []
        self.model_responses = []
        self.sep_turn_text = sep_turn_text
    
    def add_user_message(self, message: str) -> None:
        self.user_messages.append(message)
    
    def add_model_response(self, response: str) -> None:
        self.model_responses.append(response)
    
    def clear_messages(self) -> None:
        self.user_messages = []
        self.model_responses = []
    
    def get_raw_prompt(self) -> str:
        return self.format_messages(self.user_messages, self.model_responses, self.sep_turn_text)
    
    def get_model_response(
        self, 
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
        return self.client.generate(
            [self.get_raw_prompt()], 
            seed=seed, 
            max_input_length=max_input_length, 
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample, 
            num_beams=num_beams, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            eos_token_id=eos_token_id, 
        )
    
    def get_model_response_stream(
        self, 
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
        return self.client.generate_stream(
            [self.get_raw_prompt()], 
            seed=seed, 
            max_input_length=max_input_length, 
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample, 
            num_beams=num_beams, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            eos_token_id=eos_token_id, 
        )
    
    def stop_stream(self, request_id: str) -> Dict[str, Any]:
        return self.client.stop_stream(request_id)
    
    @staticmethod
    def format_messages(user_messages: List[str], model_responses: List[str], sep_turn_text: str='</s>') -> str:
        assert len(user_messages) == (len(model_responses)+1)
        text = 'BEGINNING OF CONVERSATION: '
        for user_message, model_response in zip(user_messages, model_responses):
            text += f'USER: {user_message} GPT: {model_response} {sep_turn_text} '
        text += f'USER: {user_messages[-1]} GPT:'
        return text

    @staticmethod
    def postproc_model_response(raw_model_response: str, lstrip_space: bool=True) -> str:
        model_response = raw_model_response.split('GPT:')[-1]
        if lstrip_space:
            model_response = model_response.lstrip(' ')
        return model_response

def main(
    host: Union[str, List[str]]='http://127.0.0.1:8000/', 

    max_input_length: int=1024, 
    max_output_length: int=1024, 

    seed: Optional[int]=None, 
    do_sample: bool=True, 
    num_beams: int=1, 
    temperature: Optional[float]=None, 
    top_p: Optional[float]=None, 
    top_k: Optional[float]=None, 
    eos_token_id: Optional[int]=None, 
    sep_turn_text: str='</s>', 

    stream: bool=True, 
):
    if isinstance(host, str):
        host = [host]
    client = KoalaChatClient(hosts=host, sep_turn_text=sep_turn_text)

    while True:
        message = input('>>> ')
        if message == '__exit__':
            break
        if message == '__clear__':
            client.clear_messages()
            continue
        client.add_user_message(message)

        
        model_response = ''
        if stream:
            request_id = ''

            sse_client = client.get_model_response_stream(
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
                nonlocal model_response
                request_id = curr_request_id
                print(client.postproc_model_response(next_tokens[0], lstrip_space=False), end='', flush=True)
                model_response = client.postproc_model_response(full_texts[0])
            
            signal.signal(signal.SIGINT, handle_stop)

            # print token stream
            print('Koala:', end='', flush=True)
            map_token_stream(
                sse_client, 
                callback=token_stream_read, 
                error_callback=lambda e: print('\n__ERROR__:', e), 
            )
            print()
        else:
            response = client.get_model_response(
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
            if response['status'] == 'error':
                print('__ERROR__:', response['error_message'])
                continue
            elif response['status'] == 'success':
                model_response = client.postproc_model_response(response['full_texts'][0])
                print('Koala:', model_response)
            else:
                raise NotImplementedError

        client.add_model_response(model_response)

if __name__ == "__main__":
    # `python commandline_koala_client.py -h` to see all commandline options
    tyro.cli(main)
