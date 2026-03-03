#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH}
export star_vla_python=/home/robot/guchenyang/miniconda3/envs/starVLA/bin/python

# Configuration
your_ckpt=/home/robot/guchenyang/Code/StarVLA-Dev/playground/Finetuned/QwenPI/checkpoints/steps_10000_pytorch_model.pt
gpu_id=0
port=5694

################# StarVLA Policy Server ######################
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16
