#!/bin/bash
export WANDB_API_KEY=21a8bda930a645b08f2834efd21ba21e98cd83cf

run_root_dir=./playground/LIBERO
run_id=0305_Qwen3_4B_GR00T_LIBERO_One_Shot

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

# multi-node launch example

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/qwen/qwen3-4b-gr00t-libero-one-shot.yaml \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
