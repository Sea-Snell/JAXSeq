# install basics
apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-utils \
    curl \
    git \
    vim \
    wget \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install miniforge
rm -rf Miniconda3-py39_4.12.0-Linux-x86_64.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b

# install dependencies
source miniconda3/bin/activate
conda init bash
conda env create -f environment.yml
conda activate JaxSeq
python -m pip install --upgrade pip && python -m pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# currently need to fix the jax version
python -m pip install jax==0.3.21 jaxlib==0.3.20

# clean up
rm -rf Miniconda3-py39_4.12.0-Linux-x86_64.sh
