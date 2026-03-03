#!/bin/bash
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

export WANDB_API_KEY=21a8bda930a645b08f2834efd21ba21e98cd83cf

run_root_dir=./playground/Checkpoints
run_id=0121_Qwen3_PI_Short_Data

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

# multi-node launch example

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --machine_rank=$RANK \
  --num_machines=$WORLD_SIZE \
  --num_processes=$((NPROC_PER_NODE * WORLD_SIZE)) \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/qwenpi_hand_data_nan_guard.yaml \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \

