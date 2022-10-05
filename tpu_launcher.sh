#!/bin/bash

PROJECT_HOME=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..
PROJECT_NAME=JAXSeq

function _tpu_ips {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3
    gcloud alpha compute tpus tpu-vm describe $tpu_name --zone $tpu_zone --project $tpu_project | grep -oP 'externalIp: \K(.+)$'
}

function _tpu_create {
    tpu_zone=$1
    tpu_project=$2
    tpu_cores=$3
    tpu_name=$4
    if [ "$tpu_cores" = "8" ]; then
        software_version='tpu-vm-base'
    else
        software_version='tpu-vm-base'
    fi
    gcloud alpha compute tpus tpu-vm create \
        $tpu_name \
        --accelerator-type="v3-$tpu_cores" \
        --version $software_version \
        --zone $tpu_zone \
        --project $tpu_project
}

function _tpu_retry_create {
    while true; do
        _tpu_create "$@"
        sleep 30s
    done
}

function _tpu_setup {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=($(_tpu_ips $tpu_zone $tpu_project $tpu_name))
    for host in $tpu_ips; do
        scp $PROJECT_HOME/$PROJECT_NAME/tpu_vm_setup.sh $host:~/ &
        scp $PROJECT_HOME/$PROJECT_NAME/environment.yml $host:~/ &
        ssh $host '~/tpu_vm_setup.sh' &
    done
    wait &> /dev/null

    for host in $tpu_ips; do
        scp $PROJECT_HOME/$PROJECT_NAME/tpu_vm_setup.sh $host:~/ &
        scp $PROJECT_HOME/$PROJECT_NAME/environment.yml $host:~/ &
        ssh $host '~/tpu_vm_setup.sh' &
    done
    wait &> /dev/null

    for host in $tpu_ips; do
        ssh $host 'rm -rf ~/environment.yml ~/tpu_vm_setup.sh' &
    done
    wait &> /dev/null
}

function _gcs_setup {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3
    key_name=$4

    tpu_ips=($(_tpu_ips $tpu_zone $tpu_project $tpu_name))
    for host in $tpu_ips; do
        ssh $host 'gcloud init' &
        scp ~/.config/gcloud/$key_name.json $host:~/.config/gcloud/ &
        ssh $host 'gcloud auth activate-service-account --key-file=$PWD/.config/gcloud/'$key_name'.json' &
    done
    wait &> /dev/null

    for host in $tpu_ips; do
        ssh $host 'gcloud init' &
        scp ~/.config/gcloud/$key_name.json $host:~/.config/gcloud/ &
        ssh $host 'gcloud auth activate-service-account --key-file=$PWD/.config/gcloud/'$key_name'.json' &
    done
    wait &> /dev/null
}

function _tpu_check {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=($(_tpu_ips $tpu_zone $tpu_project $tpu_name))
    for host in $tpu_ips; do
        ssh $host 'tmux capture-pane -pt launch'
    done
}

function _tpu_copy {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=($(_tpu_ips $tpu_zone $tpu_project $tpu_name))
    for host in $tpu_ips; do
        rsync -avPI --exclude=data --evclude=outputs --exclude=__pycache__ --exclude=.git $PROJECT_HOME/$PROJECT_NAME $host:~/ &
    done
    wait &> /dev/null
    sleep 1s

    for host in $tpu_ips; do
        rsync -avPI --exclude=data --exclude=outputs --exclude=__pycache__ --exclude=.git $PROJECT_HOME/$PROJECT_NAME $host:~/ &
    done
    wait &> /dev/null
    sleep 1s
}

function _clear_hf_cache {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=($(_tpu_ips $tpu_zone $tpu_project $tpu_name))
    for host in $tpu_ips; do
        ssh $host 'rm -rf ~/.cache/huggingface/transformers/ && mkdir ~/.cache/huggingface/transformers/' &
    done
    wait &> /dev/null

}

function _tpu_stop {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=($(_tpu_ips $tpu_zone $tpu_project $tpu_name))
    for host in $tpu_ips; do
        ssh $host 'tmux kill-session -t launch ; pkill -9 python' &
    done
    wait &> /dev/null
}

function _tpu_launch {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3
    command=$4

    if [ -z "$command" ]; then
        echo "Invalid syntax!"
        return 1
    fi

    tpu_ips=($(_tpu_ips $tpu_zone $tpu_project $tpu_name))
    for host in $tpu_ips; do
        echo "tmux new -d -s launch \"WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH=~/$PROJECT_NAME/src/ ~/miniconda3/envs/JaxSeq/bin/python ~/$PROJECT_NAME/scripts/$command\""
        ssh $host "tmux new -d -s launch \"WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH=~/$PROJECT_NAME/src/ ~/miniconda3/envs/JaxSeq/bin/python ~/$PROJECT_NAME/scripts/$command\"" &
    done
    wait &> /dev/null
}

function _tpu_maintain {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    gcloud alpha compute tpus tpu-vm simulate-maintenance-event $tpu_name \
        --project $tpu_project \
        --zone=$tpu_zone \
        --workers=all
}

function _tpu_ssh {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3
    command="$4"

    if [ -z "$command" ]; then
        echo "Invalid syntax!"
        return 1
    fi

    tpu_ips=($(_tpu_ips $tpu_zone $tpu_project $tpu_name))
    for host in $tpu_ips; do
        ssh $host "$command" &
    done
    wait &> /dev/null
}

function _tpu_reboot {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=($(_tpu_ips $tpu_zone $tpu_project $tpu_name))
    for host in $tpu_ips; do
        ssh $host 'sudo reboot' &
    done
    wait &> /dev/null
}

function tpu {
    trap "trap - SIGINT SIGTERM; return 1;" SIGINT SIGTERM

    if [ "$1" = "rail" ]; then
        tpu_project='rail-tpus'
        tpu_zone='us-central1-a'
    elif [ "$1" = "rail-pod" ]; then
        tpu_project='rail-tpus'
        tpu_zone='us-east1-d'
    elif [ "$1" = "nlp" ]; then
        tpu_project='civic-boulder-204700'
        tpu_zone='us-central1-a'
    elif [ "$1" = "nlp-pod" ]; then
        tpu_project='civic-boulder-204700'
        tpu_zone='us-east1-d'
    else
        echo "Invalid syntax!"
        trap - SIGINT SIGTERM
        return 1
    fi

    if [ "$2" = "list" ]; then
        gcloud alpha compute tpus tpu-vm list --zone $tpu_zone --project $tpu_project
    elif [ "$2" = "describe" ]; then
        gcloud alpha compute tpus tpu-vm describe $3 --zone $tpu_zone --project $tpu_project
    elif [ "$2" = "ips" ]; then
        _tpu_ips $tpu_zone $tpu_project $3
    elif [ "$2" = "delete" ]; then
        gcloud alpha compute tpus tpu-vm delete $3 --zone $tpu_zone --project $tpu_project --quiet
    elif [ "$2" = "create" ]; then
        _tpu_create $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "retry_create" ]; then
        _tpu_retry_create $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "cs" ]; then
        _tpu_create $tpu_zone $tpu_project $3 $4
        sleep 90s
        _tpu_setup $tpu_zone $tpu_project $4
    elif [ "$2" = "check" ]; then
        _tpu_check $tpu_zone $tpu_project $3
    elif [ "$2" = "setup" ]; then
        _tpu_setup $tpu_zone $tpu_project $3
    elif [ "$2" = "copy" ]; then
        _tpu_copy $tpu_zone $tpu_project $3
    elif [ "$2" = "stop" ]; then
        _tpu_stop $tpu_zone $tpu_project $3
    elif [ "$2" = "launch" ]; then
        _tpu_launch $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "cl" ]; then
        _tpu_copy $tpu_zone $tpu_project $3
        _tpu_launch $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "maintain" ]; then
        _tpu_maintain $tpu_zone $tpu_project $3
    elif [ "$2" = "ssh" ]; then
        _tpu_ssh $tpu_zone $tpu_project $3 "$4"
    elif [ "$2" = "reboot" ]; then
        _tpu_reboot $tpu_zone $tpu_project $3
    elif [ "$2" = "gcs_setup" ]; then
        _gcs_setup $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "clear_hf" ]; then
        _clear_hf_cache $tpu_zone $tpu_project $3
    else
        echo "Invalid syntax!"
        trap - SIGINT SIGTERM
        return 1
    fi
    trap - SIGINT SIGTERM
}

export -f tpu _tpu_ips _tpu_create _tpu_setup _tpu_check _tpu_copy _tpu_stop _tpu_launch _tpu_maintain _tpu_ssh _tpu_reboot _gcs_setup _clear_hf_cache
