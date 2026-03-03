#!/bin/bash
#
# Launch tmux sessions for SimplerEnv evaluation with state input.
# Servers are loaded one by one (staggered) to avoid GPU OOM,
# but evals run in parallel once their server is ready.
#
# Session 1 (simpler_server): 5 windows, each serving a checkpoint on GPU 0.
# Session 2 (simpler_eval):   5 windows, each evaluating 4 widowx tasks.
#

CKPT_DIR=/mnt/cpfs/guchenyang/Code/starVLA/playground/SimplerEnv/0202_PaligemmaPI_bridgev2_chunk_16/checkpoints
WORK_DIR=/mnt/cpfs/guchenyang/Code/starVLA
CONDA_BIN=/mnt/cpfs/guchenyang/miniconda3/bin/activate

STEPS=(20000 30000 40000 50000 60000)
PORTS=(6678  6679  6680  6681  6682)
GPU=0
LOAD_INTERVAL=60  # seconds between launching each server

SERVER_SESSION="simpler_server"
EVAL_SESSION="simpler_eval"

# Kill old sessions if they exist
tmux kill-session -t ${SERVER_SESSION} 2>/dev/null
tmux kill-session -t ${EVAL_SESSION} 2>/dev/null

# ============================================================
# Session 1: Policy Servers (starvla, all on GPU 0)
# ============================================================
tmux new-session -d -s ${SERVER_SESSION} -n "step_${STEPS[0]}"

for i in "${!STEPS[@]}"; do
  step=${STEPS[$i]}
  port=${PORTS[$i]}
  ckpt="${CKPT_DIR}/steps_${step}_pytorch_model.pt"
  win_name="step_${step}"

  if [ "$i" -eq 0 ]; then
    tmux rename-window -t ${SERVER_SESSION}:0 "${win_name}"
  else
    tmux new-window -t ${SERVER_SESSION} -n "${win_name}"
  fi

  tmux send-keys -t ${SERVER_SESSION}:${win_name} \
    "source ${CONDA_BIN} starvla && cd ${WORK_DIR} && bash examples/SimplerEnv/eval_files/run_policy_server.sh ${ckpt} ${port} ${GPU}" Enter

  echo "Launched server step_${step} on port ${port}. Waiting ${LOAD_INTERVAL}s for model to load..."
  sleep ${LOAD_INTERVAL}
done

echo "All 5 servers launched."

# ============================================================
# Session 2: Evaluation (simpler_env, all in parallel)
# ============================================================
tmux new-session -d -s ${EVAL_SESSION} -n "eval_${STEPS[0]}"

for i in "${!STEPS[@]}"; do
  step=${STEPS[$i]}
  port=${PORTS[$i]}
  ckpt="${CKPT_DIR}/steps_${step}_pytorch_model.pt"
  video_dir="0202_Paligemma_Bridge_16_state/step_${step}"
  win_name="eval_${step}"

  if [ "$i" -eq 0 ]; then
    tmux rename-window -t ${EVAL_SESSION}:0 "${win_name}"
  else
    tmux new-window -t ${EVAL_SESSION} -n "${win_name}"
  fi

  tmux send-keys -t ${EVAL_SESSION}:${win_name} \
    "source ${CONDA_BIN} simpler_env && cd ${WORK_DIR} && export PYTHONPATH=\$(pwd):\${PYTHONPATH} && bash examples/SimplerEnv/eval_files/start_simpler_env_with_state.sh ${ckpt} ${port} ${video_dir}" Enter
done

echo "All 5 evals launched."
echo ""
echo "Use 'tmux attach -t ${SERVER_SESSION}' to monitor servers."
echo "Use 'tmux attach -t ${EVAL_SESSION}' to monitor evaluation."
