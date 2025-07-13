#!/bin/bash

set -x
set -o pipefail
# Installation instructions for MLP-Offload (SC'25)
PWD=$(pwd)

install_cuda() {
    cd $PWD
	echo "Installing CUDA... Needs sudo access..."
	wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
	sudo sh cuda_12.3.0_545.23.06_linux.run
}

install_conda() {
    cd $PWD
	echo "Installing Conda"
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh
	echo "Please restart shell, or run 'source ~/.bashrc' or 'source ~/.bash_profile' "
}

build_env() {
    cd $PWD
	conda create -n dspeed_env python=3.12
	conda activate dspeed_env
	echo "Installing Pytorch"
	pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/rocm6.2
	pip install regex six sentencepiece pybind11 einops
	conda install anaconda::libaio
}

install_apex() {
    cd $PWD
	COMMIT_ID=c02c6c891eedfabf91f0de8127d7636d4292356d
	echo "Installing NVIDIA Apex... This takes ~10 minutes..."
	git clone https://github.com/NVIDIA/apex
	cd apex/
	# git checkout 6309120bf4158e5528 # This commit didn't give NCCL faults.
	git checkout $COMMIT_ID
	pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
}

install_megatron_ds() {
    cd $PWD
	echo "Using git submodule of Megatron-DeepSpeed"
	git submodule update --init --recursive
	cd Megatron-DeepSpeed
}

load_dataset() {
    cd $PWD
    echo "Downloading dataset..."
    mkdir -p $PWD/dataset
    cd dataset/
    if [ -d "$PWD/dataset/oscar-1GB.jsonl" ]; then
        echo "Dataset already exists, skipping download."
        return
    fi

    wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz # training dataset
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json #Vocabulary
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt # Merge File
    
    xz -d oscar-1GB.jsonl.xz     # extract the dataset.
    cd $PWD/Megatron-DeepSpeed
    python tools/preprocess_data.py \
        --input $PWD/dataset/oscar-1GB.jsonl \
        --output-prefix $PWD/dataset/my-gpt2 \
        --vocab-file $PWD/dataset/gpt2-vocab.json \
        --dataset-impl mmap \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file $PWD/dataset/gpt2-merges.txt \
        --append-eod \
        --workers 8
}

install_ds() {
	echo "Using git submodule of DeepSpeed..."
	git submodule update --init --recursive
	cd DeepSpeed
	DS_BUILD_AIO=1 DS_BUILD_CCL_COMM=1 DS_BUILD_CPU_ADAM=1 DS_BUILD_CPU_LION=0 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LION=0 DS_BUILD_CPU_ADAGRAD=1 DS_BUILD_FUSED_LAMB=1 DS_BUILD_QUANTIZER=0 DS_BUILD_RANDOM_LTD=0 DS_BUILD_SPARSE_ATTN=0 DS_BUILD_TRANSFORMER=1 DS_BUILD_TRANSFORMER_INFERENCE=0 DS_BUILD_STOCHASTIC_TRANSFORMER=0 pip install .
	echo "Checking if DeepSpeed installed successfully"
	ds_report
}

install_cuda
install_conda
build_env
install_apex
install_megatron_ds
load_dataset
install_ds


# For the system config available on Cloudlab
# https://www.wisc.cloudlab.us/portal/show-node.php?node_id=d8545-10s10505

# > cat /etc/os-release
# PRETTY_NAME="Ubuntu 24.04.2 LTS"
# NAME="Ubuntu"
# VERSION_ID="24.04"
# VERSION="24.04.2 LTS (Noble Numbat)"
# VERSION_CODENAME=noble
# ID=ubuntu
# ID_LIKE=debian
# HOME_URL="https://www.ubuntu.com/"
# SUPPORT_URL="https://help.ubuntu.com/"
# BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
# PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
# UBUNTU_CODENAME=noble
# LOGO=ubuntu-logo

# uname -r
# 6.8.0-59-generic

set +x