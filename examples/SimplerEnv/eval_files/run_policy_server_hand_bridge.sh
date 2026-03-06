#!/bin/bash

# Usage: ./run_policy_server_hand_bridge.sh [CHECKPOINT_PATH] [PORT] [GPU_ID] [ACTION_CHUNK_SIZE] [USE_STATE]
# Example: ./run_policy_server_hand_bridge.sh /path/to/model.pt 6678 0 8 true

cd /mnt/ckp/guchenyang/Code/StarVLA-Develop
export PYTHONPATH=$(pwd):${PYTHONPATH}

#### Get parameters #####
# Parameter 1: Checkpoint path
if [ -n "$1" ]; then
  your_ckpt="$1"
else
  your_ckpt=/mnt/ckp/guchenyang/Code/StarVLA-Develop/playground/Hand-Pretrain/0302_Qwen3_4B_PI_Hand_Bridge_Pretrain/checkpoints/epoch_1_steps_29382_pytorch_model.pt
fi

# Parameter 2: Port (default: 6678)
port=${2:-6678}

# Parameter 3: GPU ID (default: 0)
gpu_id=${3:-0}

# Parameter 4: Action chunk size (optional; empty means use full model chunk)
action_chunk_size="${4:-}"

# Parameter 5: Use state input (default: true). Accepts: true/false, y/n, 1/0
use_state_input="${5:-true}"
case "${use_state_input,,}" in
  true|t|yes|y|1)
    use_state_flag="--use-state"
    use_state_enabled=true
    ;;
  false|f|no|n|0)
    use_state_flag="--no-use-state"
    use_state_enabled=false
    ;;
  *)
    echo "Invalid use_state value: ${use_state_input}. Use true/false (or y/n, 1/0)."
    exit 1
    ;;
esac

action_chunk_args=()
if [ -n "${action_chunk_size}" ]; then
  action_chunk_args=(--action-chunk-size "${action_chunk_size}")
fi

# export DEBUG=true
export star_vla_python=/mnt/ckp/guchenyang/Miniconda3/envs/starvla/bin/python


echo "======================================"
echo "Starting Policy Server (Hand-Bridge)"
echo "======================================"
echo "Checkpoint: ${your_ckpt}"
echo "Port:       ${port}"
echo "GPU ID:     ${gpu_id}"
echo "Act Chunk:  ${action_chunk_size:-full model chunk}"
echo "Use State:  ${use_state_enabled}"
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
    ${use_state_flag} \
    "${action_chunk_args[@]}" \
    --use_bf16 \
    2>&1 | tee "${log_file}"
