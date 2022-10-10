# JaxSeq

## Overview

Built on top of [HuggingFace](https://huggingface.co)'s [Transformers](https://github.com/huggingface/transformers) library, JaxSeq enables training very large language models in [Jax](https://jax.readthedocs.io/en/latest/). Currently it supports GPT2, GPTJ, T5, and OPT models. JaxSeq is designed to be light-weight and easily extensible, with the aim being to demonstrate a workflow for training large language models without with the heft that is typical other existing frameworks.

Thanks to Jax's [pjit](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html) function, you can straightforwardly train models with arbitrary model and data parellelism; you can trade-off these two as you like. You can also do model parallelism across multiple hosts. Support for gradient checkpointing, gradient accumulation, and bfloat16 training/inference is provided as well for memory efficient training.

## installation

### **1. pull from github**

``` bash
git clone https://github.com/Sea-Snell/JAXSeq.git
cd JAXSeq
export PYTHONPATH=${PWD}/src/
```

### **2. install dependencies**

Install with conda (cpu, tpu, or gpu).

**install with conda (cpu):**
``` shell
conda env create -f environment.yml
conda activate JaxSeq
```

**install with conda (gpu):**
``` shell
conda env create -f environment.yml
conda activate JaxSeq
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**install with conda (tpu):**
``` shell
conda env create -f environment.yml
conda activate JaxSeq
python -m pip install --upgrade pip
python -m pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Workflow

We provide some example scripts for training and evaluating GPT2, GPTJ, OPT, and T5 models using JaxSeq. However you should feel free to build your own workflow for training. You can find these scripts in the `examples/` directory. Each script takes as input a json file which should be of shape:
``` json
{
"train": [{"in_text": "something", "out_text": "something else"}, ...], 
"eval": [{"in_text": "something else else", "out_text": "something else else else"}, ...], 
}
```

This code was largely tested, developed, and optimized for use on TPU-pods, though it should also work well on GPU clusters.

*NOTE: The T5 examples are the most stable; some of the others I still need to more extensively test.*

## Google Cloud Buckets

To further support TPU workflows the example scripts provide functionality for uploading / downloading data and or checkpoints to / from Google Cloud Storage buckets. This can be achieved by prefixing the path with `gcs://`. And depending on the permissions of the bucket, you may need to specify the google cloud project and provide an authentication token.

