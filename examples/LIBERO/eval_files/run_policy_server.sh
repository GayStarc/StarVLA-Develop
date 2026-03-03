#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH}
export star_vla_python=/mnt/nas/guchenyang/miniconda3/envs/starvla/bin/python

# --- 参数解析 ---
# $1: 端口号，默认为 5694
# $2: Checkpoint 路径
# $3: GPU ID，默认为 0
port=${1:-5694}
default_ckpt=/mnt/nas/guchenyang/Code/starVLA/playground/LIBERO/0222_Qwen3_4B_PI_LIBERO_One_Shot/checkpoints/steps_40000_pytorch_model.pt
your_ckpt=${2:-$default_ckpt}
gpu_id=${3:-0}
# ----------------

echo "---------------------------------------"
echo "Target Port: ${port}"
echo "Using CKPT:  ${your_ckpt}"
echo "GPU ID:      ${gpu_id}"
echo "---------------------------------------"

################# star Policy Server ######################

CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16