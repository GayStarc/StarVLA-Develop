#!/bin/bash

ROBOTWIN_PATH=/mnt/cpfs/guchenyang/Code/RoboTwin

policy_name="model2robotwin_interface"
task_name=${1:-adjust_bottle}
task_config=${2:-demo_clean}
ckpt_setting=${3:-qwen3_4b_oft_easy_10tasks_joint_absolute}
seed=${4:-0}
gpu_id=${5:-0} # default is 0

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

EVAL_FILES_PATH=$(pwd)
STARVLA_PATH=$EVAL_FILES_PATH/../../..
DEPLOY_POLICY_PATH=$EVAL_FILES_PATH/deploy_policy.yml

export PYTHONPATH=$ROBOTWIN_PATH:$PYTHONPATH
export PYTHONPATH=$STARVLA_PATH:$PYTHONPATH
export PYTHONPATH=$EVAL_FILES_PATH:$PYTHONPATH

cd $ROBOTWIN_PATH

echo "PYTHONPATH: $PYTHONPATH"

PYTHONWARNINGS=ignore::UserWarning \
# export CUROBO_TORCH_COMPILE_DISABLE=0
python script/eval_policy.py --config $DEPLOY_POLICY_PATH \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name}
