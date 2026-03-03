#!/bin/bash
#
# 仅启动 eval tmux session（server 需已挂载）
# 对应 run_eval_with_state.sh 的 Session 2 部分
# 5*4=20 个窗口，每个 checkpoint x 每个 task suite
#
# EVAL_GPU / $1: libero 推理评测使用的 GPU ID，默认 0
#

CKPT_DIR=/mnt/nas/guchenyang/Code/starVLA/playground/LIBERO/0205_PaligemmaPI_LIBERO_Chunk_8/checkpoints
WORK_DIR=/mnt/nas/guchenyang/Code/starVLA
CONDA_BIN=/mnt/nas/guchenyang/miniconda3/bin/activate

STEPS=(10000 20000 30000 40000 50000)
PORTS=(6945  6946  6947  6948  6949)

EXP_NAME="0205_PaligemmaPI_LIBERO_Chunk_8"
HOST="127.0.0.1"
NUM_TRIALS=50

EVAL_GPU=${1:-${EVAL_GPU:-1}}

LIBERO_HOME=/mnt/nas/guchenyang/Code/vla-scratch/examples/LIBERO
LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
LIBERO_Python=/mnt/nas/guchenyang/miniconda3/envs/libero/bin/python

TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

EVAL_SESSION="libero_eval_paligemma"

echo "Eval GPU: ${EVAL_GPU}"

# Kill old eval session
# tmux kill-session -t ${EVAL_SESSION} 2>/dev/null

# ============================================================
# Eval Session: 每个 checkpoint x 每个 task suite = 一个 tmux 窗口
# ============================================================
first_window=true

for i in "${!STEPS[@]}"; do
  step=${STEPS[$i]}
  port=${PORTS[$i]}
  ckpt="${CKPT_DIR}/steps_${step}_pytorch_model.pt"
  results_base="results/${EXP_NAME}/step_${step}"

  for task_suite in "${TASK_SUITES[@]}"; do
    win_name="s${step}_${task_suite}"
    video_out_path="${results_base}/${task_suite}"

    if [ "$first_window" = true ]; then
      tmux new-session -d -s ${EVAL_SESSION} -n "${win_name}"
      first_window=false
    else
      tmux new-window -t ${EVAL_SESSION} -n "${win_name}"
    fi

    eval_cmd="source ${CONDA_BIN} libero && cd ${WORK_DIR}"
    eval_cmd="${eval_cmd} && export CUDA_VISIBLE_DEVICES=${EVAL_GPU}"
    eval_cmd="${eval_cmd} && export LIBERO_HOME=${LIBERO_HOME}"
    eval_cmd="${eval_cmd} && export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
    eval_cmd="${eval_cmd} && export PYTHONPATH=\$(pwd):\${PYTHONPATH}:${LIBERO_HOME}"
    eval_cmd="${eval_cmd} && ${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py"
    eval_cmd="${eval_cmd} --args.pretrained-path ${ckpt}"
    eval_cmd="${eval_cmd} --args.host ${HOST}"
    eval_cmd="${eval_cmd} --args.port ${port}"
    eval_cmd="${eval_cmd} --args.task-suite-name ${task_suite}"
    eval_cmd="${eval_cmd} --args.num-trials-per-task ${NUM_TRIALS}"
    eval_cmd="${eval_cmd} --args.video-out-path ${video_out_path}"
    eval_cmd="${eval_cmd} --args.results-dir ${results_base}"
    eval_cmd="${eval_cmd} --args.use-state"

    tmux send-keys -t ${EVAL_SESSION}:${win_name} "${eval_cmd}" Enter
  done
done

echo "Eval session '${EVAL_SESSION}' 已启动，共 $(( ${#STEPS[@]} * ${#TASK_SUITES[@]} )) 个窗口"
echo "attach: tmux a -t ${EVAL_SESSION}"
