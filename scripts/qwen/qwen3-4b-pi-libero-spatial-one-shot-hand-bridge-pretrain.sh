#!/bin/bash
export WANDB_API_KEY=21a8bda930a645b08f2834efd21ba21e98cd83cf

run_root_dir=./playground/LIBERO
run_id=0304_Qwen3_4B_PI_LIBERO_Spatial_One_Shot_Hand_Bridge_Pretrain

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

# Create a symlink without step info in the name so that completed_steps starts from 0
PRETRAIN_CKPT=./playground/Hand-Pretrain/0302_Qwen3_4B_PI_Hand_Bridge_Pretrain/checkpoints/epoch_1_steps_29382_pytorch_model.pt
SYMLINK_PATH=./playground/Hand-Pretrain/0302_Qwen3_4B_PI_Hand_Bridge_Pretrain/checkpoints/hand_bridge_pretrained_model.pt
if [ ! -e "${SYMLINK_PATH}" ]; then
  ln -s "$(realpath ${PRETRAIN_CKPT})" "${SYMLINK_PATH}"
  echo "Created symlink: ${SYMLINK_PATH} -> ${PRETRAIN_CKPT}"
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/qwen/qwen3-4b-pi-libero-spatial-one-shot-hand-bridge-pretrain.yaml \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
