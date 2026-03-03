#!/bin/bash
#
# 单次 eval 脚本，仅运行评估（server 需已启动）
# 用法: ./run_eval_only.sh <checkpoint_path> <port> [task_suite] [use_state]
#
# 示例:
#   ./run_eval_only.sh /path/to/checkpoint.pt 5698
#   ./run_eval_only.sh /path/to/checkpoint.pt 5698 libero_10
#   ./run_eval_only.sh /path/to/checkpoint.pt 5698 libero_10 n    # 不用 state
#

set -e

# 默认配置
WORK_DIR=/mnt/nas/guchenyang/Code/starVLA
LIBERO_HOME=/mnt/nas/guchenyang/Code/vla-scratch/examples/LIBERO
LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
LIBERO_Python=/mnt/nas/guchenyang/miniconda3/envs/libero/bin/python
CONDA_BIN=/mnt/nas/guchenyang/miniconda3/bin/activate

CKPT=${1:? "Usage: $0 <checkpoint_path> <port> [task_suite] [use_state: y/n]"}
PORT=${2:? "Usage: $0 <checkpoint_path> <port> [task_suite] [use_state: y/n]"}
TASK_SUITE=${3:-libero_10}
USE_STATE_INPUT=${4:-y}

HOST=${HOST:-127.0.0.1}
NUM_TRIALS=${NUM_TRIALS:-50}
EXP_NAME=${EXP_NAME:-eval_only}

# use_state: y=--args.use-state, n=--args.no-use-state
if [ "$USE_STATE_INPUT" = "n" ] || [ "$USE_STATE_INPUT" = "N" ]; then
    USE_STATE_FLAG="--args.no-use-state"
else
    USE_STATE_FLAG="--args.use-state"
fi

RESULTS_BASE="results/${EXP_NAME}"
VIDEO_OUT="${RESULTS_BASE}/${TASK_SUITE}"
RESULTS_DIR="${RESULTS_BASE}"

echo "=========================================="
echo "LIBERO Eval (server 已就绪)"
echo "  Checkpoint: ${CKPT}"
echo "  Port: ${PORT}"
echo "  Task suite: ${TASK_SUITE}"
echo "  Results: ${RESULTS_DIR}"
echo "=========================================="

source ${CONDA_BIN} libero
cd ${WORK_DIR}
export LIBERO_HOME=${LIBERO_HOME}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}
export PYTHONPATH=$(pwd):${PYTHONPATH}:${LIBERO_HOME}

${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${CKPT}" \
    --args.host "${HOST}" \
    --args.port "${PORT}" \
    --args.task-suite-name "${TASK_SUITE}" \
    --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.video-out-path "${VIDEO_OUT}" \
    --args.results-dir "${RESULTS_DIR}" \
    ${USE_STATE_FLAG}

echo "Eval 完成，结果保存在 ${RESULTS_DIR}"
