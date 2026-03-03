#!/bin/bash

run_root_dir=./results/Checkpoints
run_id=Qwen3_4B_PI_LIBERO_Spatial_One_Shot

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/qwen/qwen3-4b-pi-libero-spatial-one-shot.yaml \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
