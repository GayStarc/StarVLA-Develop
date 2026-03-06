#!/bin/bash
set -euo pipefail

run_root_dir=${RUN_ROOT_DIR:-./playground/Bridge}
run_id=${RUN_ID:-0306_Qwen3_4B_PI_Bridge_Orig_V21_NormalHead}
num_processes=${NUM_PROCESSES:-8}
config_yaml=./starVLA/config/training/qwen/qwen3-4b-pi-bridge-orig-v21.yaml
deepspeed_cfg=${DEEPSPEED_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

launch_args=(
  --config_file "${deepspeed_cfg}"
  --num_processes "${num_processes}"
)

# If distributed env vars are provided, run in multi-node mode.
if [[ -n "${MASTER_ADDR:-}" ]]; then
  launch_args+=(
    --main_process_ip "${MASTER_ADDR}"
    --main_process_port "${MASTER_PORT:-29500}"
    --machine_rank "${RANK:-0}"
    --num_machines "${WORLD_SIZE:-1}"
  )
fi

accelerate launch \
  "${launch_args[@]}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
