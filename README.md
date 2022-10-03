# JaxSeq

## installation

### **1. pull from github**

``` bash
git clone https://github.com/Sea-Snell/JaxSeq.git
cd JaxSeq
export PYTHONPATH=${PWD}/src/
```

### **2. install dependencies**

Install with conda (cpu, tpu, or gpu) or docker (cpu or gpu only).

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
conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch
```

**install with conda (tpu):**
``` shell
conda env create -f environment.yml
conda activate JaxSeq
python -m pip install --upgrade pip
python -m pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
