#!/bin/bash
export WANDB_API_KEY=21a8bda930a645b08f2834efd21ba21e98cd83cf

run_root_dir=./playground/Checkpoints
run_id=0202_Qwen3PI_bridgev2_chunk_16

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

# multi-node launch example

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $RANK \
  --num_machines $WORLD_SIZE \
  --num_processes=$NPROC_PER_NODE \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/qwenpi_bridge_v2.yaml \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \

