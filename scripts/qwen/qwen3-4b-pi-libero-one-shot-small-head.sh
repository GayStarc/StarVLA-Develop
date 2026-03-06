#!/bin/bash
export WANDB_API_KEY=21a8bda930a645b08f2834efd21ba21e98cd83cf

export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000

run_root_dir=./playground/LIBERO
run_id=0305_Qwen3_4B_PI_LIBERO_One_Shot_SmallHead

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

GPUS_PER_NODE=${NPROC_PER_NODE:-2}
NNODES=${WORLD_SIZE:-2}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NUM_PROCESSES=$((GPUS_PER_NODE * NNODES))

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${NODE_RANK} \
  --num_machines ${NNODES} \
  --num_processes ${NUM_PROCESSES} \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/qwen/qwen3-4b-pi-libero-one-shot-small-head.yaml \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
