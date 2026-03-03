export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)


# === Please modify the following paths according to your environment ===
###########################################################################################

Framework_name=QwenPI
base_vlm=/mnt/cpfs/guchenyang/Pretrained/Qwen2.5-VL-3B-Instruct-Action
DIT_TYPE="DiT-B"
oxe_data_root=/mnt/cpfs/guchenyang/Data/HuggingFace/lerobot/gaystarc
data_mix=bridge_v2
run_root_dir=./playground/Checkpoints
run_id=0111_Qwen25_PI_bridgev2
export action_input_dim=2048
###########################################################################################


output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/qwenpi_bridge_v2.yaml \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project QwenPI \
  --wandb_entity gaystarc-peking-university \

