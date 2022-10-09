import contextlib
from functools import partial
from typing import Any, Dict, List, Optional
from flask import Flask, request
from flask_cors import CORS
from models.gptj import load_gptj_model
from seq2seq import load_dec_inference
from utils.serve_queue import serve_class
import jax
import json
from transformers import AutoTokenizer
import os
import numpy as np
from jax.experimental.maps import Mesh
from shard import shard_params
import tyro

# setup app

app = Flask(__name__)
CORS(app)

# setup thread safe model client

class InferenceServer:
    def __init__(
        self, 

        model_name: str, # EleutherAI/gpt-j-6B [6.5B]

        checkpoint_path: Optional[str]=None, 
        checkpoint_is_sharded: bool=True, 

        do_pjit: bool=True, 
        model_p_shape: int=1, 
        data_p_shape: int=1, 

        gcloud_project: Optional[str]=None, 
        gcloud_token_path: Optional[str]=None, 
    ):
        from utils.gcs_manager import open_pp as open
        open = partial(open, gcloud_project=gcloud_project, gcloud_token=gcloud_token_path)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

        if checkpoint_is_sharded and checkpoint_path is not None:
            tail_checkpoint, head_checkpoint = os.path.split(checkpoint_path.strip('/'))
            checkpoint_path = os.path.join(tail_checkpoint, 'shard_%d' % (jax.process_index()), head_checkpoint)

        model, params, shard_rules = load_gptj_model(
            model_str=model_name, 
            from_pretrained=True, 
            checkpoint_path=checkpoint_path, 
            use_fp16=jax.default_backend() == 'tpu', 
            tokenizer=tokenizer, 
            gradient_checkpoint=False, 
            seed=0, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token_path, 
        )

        # mesh definition
        if do_pjit:
            mesh_devices = np.array(jax.devices()).reshape(data_p_shape, model_p_shape)
            print('using mesh shape:', mesh_devices.shape)
            print('full mesh:', mesh_devices)
            mesh = Mesh(mesh_devices, ("dp", "mp"))
        else:
            mesh = contextlib.nullcontext()

        # shard params and optimizer
        if do_pjit:
            params, param_spec = shard_params(partial(model.init_weights, input_shape=(1, 1)), 
                                                      params, shard_rules, mesh)
        else:
            param_spec = None

        self.inference = load_dec_inference(
            model=model, 
            params=params, 
            param_spec=param_spec, 
            tokenizer=tokenizer, 
            do_pjit=do_pjit, 
        )
        self.mesh = mesh
    
    def generate(self, in_strs: List[str], max_input_length: int, 
                 rng: int, **generation_kwargs: Dict[str, Any]):
        with self.mesh:
            return self.inference.generate_from_str(in_strs, max_input_length, jax.random.PRNGKey(rng), **generation_kwargs)
    
    def log_probs(self, in_strs: List[str], out_strs: List[str], max_input_length: int, max_output_length: int):
        with self.mesh:
            return self.inference.log_probs_from_str(in_strs, out_strs, max_input_length, max_output_length).log_probs.tolist()

InferenceServerMP = serve_class(InferenceServer)

inference_server = tyro.cli(InferenceServerMP)

# flask endpoints

@app.route('/generate', methods=['POST'])
def generate():
    global inference_server
    
    in_strs = request.json.get('in_strs', None)
    assert in_strs is not None
    max_input_length = request.json.get('max_input_length', None)
    assert max_input_length is not None
    rng = request.json.get('rng', None)
    assert rng is not None
    generation_kwargs = request.json.get('generation_kwargs', None)
    assert generation_kwargs is not None
    
    result = inference_server.generate(in_strs, max_input_length, rng, **generation_kwargs)
    return json.dumps(result)

@app.route('/log_probs', methods=['POST'])
def log_probs():
    global inference_server
    
    in_strs = request.json.get('in_strs', None)
    assert in_strs is not None
    out_strs = request.json.get('out_strs', None)
    assert out_strs is not None
    max_input_length = request.json.get('max_input_length', None)
    assert max_input_length is not None
    max_output_length = request.json.get('max_output_length', None)
    assert max_output_length is not None
    
    result = inference_server.log_probs(in_strs, out_strs, max_input_length, max_output_length)
    return json.dumps(result)

# run app

# if using guncorn to serve, make sure to set workers=1, and worker-class=gthread
# for example run: `python -m gunicorn --worker-class=gthread --workers=1 --timeout=3600 -b 0.0.0.0:8000 gptj_serve:app`

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, threaded=True, processes=1)