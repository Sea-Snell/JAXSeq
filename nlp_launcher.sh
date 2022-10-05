#!/bin/bash

PROJECT_HOME=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..
PROJECT_NAME=JAXSeq

function _nlp_setup {
    ip=$1

    scp $PROJECT_HOME/$PROJECT_NAME/tpu_vm_setup.sh $ip:~/ &
    scp $PROJECT_HOME/$PROJECT_NAME/environment.yml $ip:~/ &
    ssh $ip '~/tpu_vm_setup.sh' &
    wait &> /dev/null

    scp $PROJECT_HOME/$PROJECT_NAME/tpu_vm_setup.sh $ip:~/ &
    scp $PROJECT_HOME/$PROJECT_NAME/environment.yml $ip:~/ &
    ssh $ip '~/tpu_vm_setup.sh' &
    wait &> /dev/null

    ssh $ip 'rm -rf ~/environment.yml ~/tpu_vm_setup.sh' &
    wait &> /dev/null
}

function _nlp_gcs_setup {
    ip=$1
    key_name=$2

    ssh $ip 'gcloud init' &
    scp ~/.config/gcloud/$key_name.json $ip:~/.config/gcloud/ &
    ssh $ip 'gcloud auth activate-service-account --key-file=$PWD/.config/gcloud/'$key_name'.json' &
    wait &> /dev/null

    ssh $ip 'gcloud init' &
    scp ~/.config/gcloud/$key_name.json $ip:~/.config/gcloud/ &
    ssh $ip 'gcloud auth activate-service-account --key-file=$PWD/.config/gcloud/'$key_name'.json' &
    wait &> /dev/null
}

function _nlp_check {
    ip=$1

    ssh $ip 'tmux capture-pane -pt launch'
}

function _nlp_usage {
    ip=$1

    ssh $ip 'nvidia-smi'
}

function _nlp_copy {
    ip=$1

    rsync -avPI --exclude=data --exclude=outputs --exclude=__pycache__ --exclude=.git $PROJECT_HOME/$PROJECT_NAME $ip:~/ &
    wait &> /dev/null
    sleep 1s

    rsync -avPI --exclude=data --exclude=outputs --exclude=__pycache__ --exclude=.git $PROJECT_HOME/$PROJECT_NAME $ip:~/ &
    wait &> /dev/null
    sleep 1s
}

function _nlp_clear_hf_cache {
    ip=$1

    ssh $ip 'rm -rf ~/.cache/huggingface/transformers/ && mkdir ~/.cache/huggingface/transformers/' &
    wait &> /dev/null
}

function _nlp_launch {
    ip=$1
    command=$2

    if [ -z "$command" ]; then
        echo "Invalid syntax!"
        return 1
    fi
    
    echo "tmux new -d -s launch \"WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH=~/$PROJECT_NAME/src/ python3 /shared/cathychen/gpu_scheduler/reserve.py \\\"~/miniconda3/envs/JaxSeq/bin/python ~/$PROJECT_NAME/scripts/$command\\\"\""
    ssh $ip "tmux new -d -s launch \"WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH=~/$PROJECT_NAME/src/ python3 /shared/cathychen/gpu_scheduler/reserve.py \\\"~/miniconda3/envs/JaxSeq/bin/python ~/$PROJECT_NAME/scripts/$command\\\"\"" &
    wait &> /dev/null
}

function _nlp_ssh {
    ip=$1
    command="$2"

    if [ -z "$command" ]; then
        echo "Invalid syntax!"
        return 1
    fi

    ssh $ip "$command" &
    wait &> /dev/null
}

function nlp {
    trap "trap - SIGINT SIGTERM; return 1;" SIGINT SIGTERM

    if [ "$1" = "1" ]; then
        ip='128.32.162.167'
    elif [ "$1" = "2" ]; then
        ip='128.32.162.168'
    else
        echo "Invalid syntax!"
        trap - SIGINT SIGTERM
        return 1
    fi

    if [ "$2" = "check" ]; then
        _nlp_check $ip
    elif [ "$2" = "ip" ]; then
        echo $ip
    elif [ "$2" = "usage" ]; then
        _nlp_usage $ip
    elif [ "$2" = "setup" ]; then
        _nlp_setup $ip
    elif [ "$2" = "copy" ]; then
        _nlp_copy $ip
    elif [ "$2" = "launch" ]; then
        _nlp_launch $ip $3
    elif [ "$2" = "cl" ]; then
        _nlp_copy $ip
        _nlp_launch $ip $3
    elif [ "$2" = "ssh" ]; then
        _nlp_ssh $ip "$3"
    elif [ "$2" = "gcs_setup" ]; then
        _nlp_gcs_setup $ip $3
    elif [ "$2" = "clear_hf" ]; then
        _nlp_clear_hf_cache $ip
    else
        echo "Invalid syntax!"
        trap - SIGINT SIGTERM
        return 1
    fi
    trap - SIGINT SIGTERM
}

export -f nlp _nlp_setup _nlp_check _nlp_copy _nlp_launch _nlp_ssh _nlp_gcs_setup _nlp_clear_hf_cache _nlp_usage
