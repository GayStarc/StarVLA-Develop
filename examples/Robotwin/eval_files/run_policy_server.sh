#!/bin/bash
# 用法: ./run_policy_server.sh [ckpt_path] [port] [gpu_id]
# 示例: ./run_policy_server.sh /path/to/checkpoint.pt 5694 0

export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/mnt/cpfs/guchenyang/miniconda3/envs/starvla/bin/python

# 默认值
default_ckpt=/mnt/cpfs/guchenyang/Code/starVLA/playground/RoboTwin/0208_Easy_10Tasks_Qwen3_4B_OFT_Joint_Absolute/checkpoints/steps_50000_pytorch_model.pt
default_port=5694
default_gpu_id=0

# 可输入参数（按顺序：ckpt_path port gpu_id）
your_ckpt=${1:-$default_ckpt}
port=${2:-$default_port}
gpu_id=${3:-$default_gpu_id}
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
