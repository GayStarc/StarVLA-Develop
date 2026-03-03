#!/bin/bash

# Usage: ./run_policy_server.sh [CHECKPOINT_PATH] [PORT] [GPU_ID]
# Example: ./run_policy_server.sh /path/to/model.pt 6679 0

cd /mnt/cpfs/guchenyang/Code/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}

#### Get parameters #####
# Parameter 1: Checkpoint path
if [ -n "$1" ]; then
  your_ckpt="$1"
else
  your_ckpt=/mnt/cpfs/guchenyang/Code/starVLA/playground/SimplerEnv/0202_PaligemmaPI_bridgev2_chunk_16/checkpoints/steps_70000_pytorch_model.pt
fi

# Parameter 2: Port (default: 6679)
port=${2:-6678}

# Parameter 3: GPU ID (default: 0)
gpu_id=${3:-0}

# export DEBUG=true
# export star_vla_python=/root/miniconda3/envs/starVLA/bin/python
export star_vla_python=/mnt/cpfs/guchenyang/miniconda3/envs/starvla/bin/python


echo "======================================"
echo "🚀 Starting Policy Server"
echo "======================================"
echo "Checkpoint: ${your_ckpt}"
echo "Port:       ${port}"
echo "GPU ID:     ${gpu_id}"
echo "======================================"

#### build output directory #####
ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"


#### run server #####
CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${log_file}"