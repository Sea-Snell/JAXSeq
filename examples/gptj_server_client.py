from enum import Enum
from typing import List, Any, Optional
import requests
import json
import random
import tyro

# client for interacting with the server created by running `python gptj_serve.py`

# localhost; change this if you are using a public server
HOST = 'http://127.0.0.1:8000/'

class Client:
	def __init__(self, host: str):
		self.host = host

	def request(self, function: str, data: Any):
		return json.loads(requests.post(self.host+function, json=data).text)

	def generate(self, prompts: List[str], seed: int, max_input_length: int=512, 
				 max_generation_length: int=512, do_sample: bool=True, 
                 temperature: Optional[float]=None, top_p: Optional[float]=None, 
                 top_k: Optional[int]=None):
		return self.request('generate', {'in_strs': prompts, 'max_input_length': max_input_length, 
										 'rng': seed, 'generation_kwargs': {'do_sample': do_sample, 
										 'max_length': max_generation_length+max_input_length, 
                                         'temperature': temperature, 'top_p': top_p, 'top_k': top_k}})

	def log_probs(self, in_strs: List[str], out_strs: List[str], 
				  max_input_length: int=512, max_output_length: int=512):
		return self.request('log_probs', {'in_strs': in_strs, 'out_strs': out_strs, 
										  'max_input_length': max_input_length, 
										  'max_output_length': max_output_length})


class FunctionOption(Enum):
    generate: str = 'generate'
    log_probs: str = 'log_probs'

def interact(
    function: FunctionOption, 
    prompt: str, 

    /,  # Mark the end of positional arguments.

    # generation settings, only for generate function
    do_sample: bool=True, 
    seed: Optional[int]=None, 
    temperature: Optional[float]=None, 
    top_p: Optional[float]=None, 
    top_k: Optional[float]=None, 

    # output string to measure logprob of; only for logprobs function
    completion: Optional[str]=None, 
):
    client = Client(HOST)

    if function is FunctionOption.generate:
        if seed is None:
            seed = random.randint(0, 2**30-1)
        generations = client.generate(
            [prompt], 
            seed=seed, 
            max_input_length=768, 
            max_generation_length=256, 
            do_sample=do_sample, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
        )
        print(generations[0])
    elif function is FunctionOption.log_probs:
        log_probs = client.log_probs(
            [prompt], 
            [completion], 
            max_input_length=768, 
            max_output_length=256, 
        )
        print(log_probs[0])
    else:
        raise NotImplementedError

if __name__ == "__main__":
    # `python gptj_server_client.py generate "Hello world!"` to generate a completion
    # `python gptj_server_client.py -h` to see all commandline options
    tyro.cli(interact)
