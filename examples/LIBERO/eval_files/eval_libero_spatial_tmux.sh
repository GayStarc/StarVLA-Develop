#!/bin/bash
#
# LIBERO Spatial 评测脚本（仅 eval client，server 需已启动）
# 使用 tmux 在后台运行评测
#
# 用法:
#   ./eval_libero_spatial_tmux.sh [port] [checkpoint_path] [gpu_id]
#
# 示例:
#   ./eval_libero_spatial_tmux.sh
#   ./eval_libero_spatial_tmux.sh 5694
#   ./eval_libero_spatial_tmux.sh 5694 /path/to/checkpoint.pt 0
#

set -e

# === 环境配置 ===
WORK_DIR=/mnt/ckp/guchenyang/Code/StarVLA-Develop
CONDA_BIN=/mnt/ckp/guchenyang/Miniconda3/bin/activate
LIBERO_HOME=/mnt/ckp/guchenyang/Code/VLA-Scratch-Dev/simulation/LIBERO
LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
LIBERO_Python=/mnt/ckp/guchenyang/Miniconda3/envs/libero/bin/python

# === 参数解析 ===
port=${1:-5694}
default_ckpt=/mnt/ckp/guchenyang/Code/StarVLA-Develop/playground/LIBERO/0303_Qwen3_4B_PI_LIBERO_Spatial_One_Shot_Hand_Bridge_Pretrain/checkpoints/epoch_1200_steps_16635_pytorch_model.pt
your_ckpt=${2:-$default_ckpt}
gpu_id=${3:-0}

# === 自动生成实验名和结果路径 ===
HOST="127.0.0.1"
NUM_TRIALS=50
TASK_SUITE="libero_spatial"

# 从 checkpoint 路径提取实验名 (取 checkpoints 上一级目录名)
EXP_NAME=$(basename $(dirname $(dirname "$your_ckpt")))
# 从 checkpoint 文件名提取 step 标识
CKPT_STEM=$(basename "$your_ckpt" .pt | sed 's/_pytorch_model//')

RESULTS_BASE="results/${EXP_NAME}/${CKPT_STEM}"
VIDEO_OUT="${RESULTS_BASE}/${TASK_SUITE}"

SESSION_NAME="libero_spatial_eval"

# === 检查 tmux session ===
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists."
    echo "  attach: tmux attach -t $SESSION_NAME"
    echo "  kill:   tmux kill-session -t $SESSION_NAME"
    exit 1
fi

echo "=========================================="
echo "LIBERO Spatial Eval (server 需已启动)"
echo "  Checkpoint: ${your_ckpt}"
echo "  Port:       ${port}"
echo "  GPU ID:     ${gpu_id}"
echo "  Task Suite: ${TASK_SUITE}"
echo "  Trials:     ${NUM_TRIALS} per task"
echo "  Results:    ${RESULTS_BASE}"
echo "=========================================="

# === 创建 tmux session 并运行 ===
tmux new-session -d -s $SESSION_NAME -n "${TASK_SUITE}"

eval_cmd="source ${CONDA_BIN} libero && cd ${WORK_DIR}"
eval_cmd="${eval_cmd} && export CUDA_VISIBLE_DEVICES=${gpu_id}"
eval_cmd="${eval_cmd} && export LIBERO_HOME=${LIBERO_HOME}"
eval_cmd="${eval_cmd} && export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
eval_cmd="${eval_cmd} && export PYTHONPATH=\$(pwd):\${PYTHONPATH}:${LIBERO_HOME}"
eval_cmd="${eval_cmd} && ${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py"
eval_cmd="${eval_cmd} --args.pretrained-path ${your_ckpt}"
eval_cmd="${eval_cmd} --args.host ${HOST}"
eval_cmd="${eval_cmd} --args.port ${port}"
eval_cmd="${eval_cmd} --args.task-suite-name ${TASK_SUITE}"
eval_cmd="${eval_cmd} --args.num-trials-per-task ${NUM_TRIALS}"
eval_cmd="${eval_cmd} --args.video-out-path ${VIDEO_OUT}"
eval_cmd="${eval_cmd} --args.results-dir ${RESULTS_BASE}"
# eval_cmd="${eval_cmd} --args.use-state"

tmux send-keys -t ${SESSION_NAME}:${TASK_SUITE} "${eval_cmd}" Enter

echo ""
echo "Tmux session '${SESSION_NAME}' 已启动!"
echo "  attach: tmux attach -t ${SESSION_NAME}"
echo "  detach: Ctrl+b then d"
echo "  kill:   tmux kill-session -t ${SESSION_NAME}"
echo "=========================================="
