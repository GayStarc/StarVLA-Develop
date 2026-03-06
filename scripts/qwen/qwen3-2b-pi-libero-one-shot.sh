#!/bin/bash
export WANDB_API_KEY=21a8bda930a645b08f2834efd21ba21e98cd83cf

run_root_dir=./playground/LIBERO
run_id=0305_Qwen3_2B_PI_LIBERO_One_Shot_SmallHead

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 6 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/qwen/qwen3-2b-pi-libero-one-shot.yaml \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
