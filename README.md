# JaxSeq

## Overview

Built on top of [HuggingFace](https://huggingface.co)'s [Transformers](https://github.com/huggingface/transformers) library, JaxSeq enables training very large language models in [Jax](https://jax.readthedocs.io/en/latest/). Currently it supports GPT2, GPTJ, T5, and OPT models. JaxSeq is designed to be light-weight and easily extensible, with the aim being to demonstrate a workflow for training large language models without with the heft that is typical other existing frameworks.

Thanks to Jax's [pjit](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html) function, you can straightforwardly train models with arbitrary model and data parellelism; you can trade-off these two as you like. You can also do model parallelism across multiple hosts. Support for gradient checkpointing, gradient accumulation, and bfloat16 training/inference is provided as well for memory efficient training.

***If you encounter an error or want to contribute, feel free to drop an issue!***

## installation

### **1. pull from github**

``` bash
git clone https://github.com/Sea-Snell/JAXSeq.git
cd JAXSeq
```

### **2. install dependencies**

Install with conda (cpu, tpu, or gpu).

**install with conda (cpu):**
``` shell
conda env create -f environment.yml
conda activate JaxSeq
python -m pip install --upgrade pip
python -m pip install -e .
```

**install with conda (gpu):**
``` shell
conda env create -f environment.yml
conda activate JaxSeq
python -m pip install --upgrade pip
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
python -m pip install -e .
```

**install with conda (tpu):**
``` shell
conda env create -f environment.yml
conda activate JaxSeq
python -m pip install --upgrade pip
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -e .
```

## Workflow

We provide some example scripts for training and evaluating GPT2, GPTJ, LLaMA, and T5 models using JaxSeq. However you should feel free to build your own workflow for training. You can find these scripts in the `examples/` directory. Each training script takes as input a jsonl file for eval and train data, each of which should be of shape:
``` json
{"in_text": "something", "out_text": "something else"}, 
{"in_text": "something else else", "out_text": "something else else else"}, 
...
```

The examples all use [tyro](https://github.com/brentyi/tyro) to manage commandline args (see their [documentation](https://brentyi.github.io/tyro)).

This code was largely tested, developed, and optimized for use on TPU-pods, though it should also work well on GPU clusters.

## Google Cloud Buckets

To further support TPU workflows the example scripts provide functionality for uploading / downloading data and or checkpoints to / from Google Cloud Storage buckets. This can be achieved by prefixing the path with `gcs://`. And depending on the permissions of the bucket, you may need to specify the google cloud project and provide an authentication token.


## Other Excellent References for Working with Large Models in Jax

* [DALL-E Mini Repo](https://t.co/BlM8e66utJ)
* [Huggingface Model Parallel Jax Demo](https://t.co/eGscnvtNDR)
* [GPT-J Repo](https://github.com/kingoflolz/mesh-transformer-jax) [uses xmap instead of pjit]
* [Alpa](https://github.com/alpa-projects/alpa)
* [Jaxformer](https://github.com/salesforce/jaxformer)
