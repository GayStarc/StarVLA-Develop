#!/bin/bash
# Sweep evaluation for StarVLA across checkpoints and all 4 LIBERO task suites using tmux.
#
# Usage: bash examples/LIBERO/eval_files/sweep_eval.sh [gpu_id] [port] [ckpt_base] [use_state] [action_chunk_size]
#
# Creates a tmux session "libero-sweep-<timestamp>" with 6 windows:
#   - Window 0 (serve):          policy server
#   - Window 1 (libero_spatial): eval
#   - Window 2 (libero_object):  eval
#   - Window 3 (libero_goal):    eval
#   - Window 4 (libero_10):      eval
#   - Window 5 (orchestrator):   sweep controller
#
# For each checkpoint step, the script:
#   1. Starts server in window 0
#   2. Waits for server ready (port polling + warmup)
#   3. Launches all 4 evals in parallel (one per window)
#   4. Waits for all evals to finish (.done flag files)
#   5. Kills server, moves to next checkpoint

set -euo pipefail

# === 环境配置 ===
WORK_DIR=/mnt/ckp/guchenyang/Code/StarVLA-Develop
star_vla_python=/mnt/ckp/guchenyang/Miniconda3/envs/starvla/bin/python
LIBERO_HOME=/mnt/ckp/guchenyang/Code/VLA-Scratch-Dev/simulation/LIBERO
LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
LIBERO_Python=/mnt/ckp/guchenyang/Miniconda3/envs/libero/bin/python
CONDA_BIN=/mnt/ckp/guchenyang/Miniconda3/bin/activate

cd "$WORK_DIR"

# === 参数解析 ===
GPU_ID=${1:-0}
PORT=${2:-5694}
CKPT_BASE=${3:-/mnt/ckp/guchenyang/Code/StarVLA-Develop/playground/LIBERO/0305_Qwen3_4B_GR00T_LIBERO_One_Shot/checkpoints}
USE_STATE=${4:-n}
ACTION_CHUNK_SIZE=${5:-8}  # 留空则使用模型默认值

NUM_TRIALS=50
STEPS=(20000 10000)
TASK_SUITES=(libero_spatial libero_object libero_goal libero_10)

EXP_NAME="0305_Qwen3_4B_GR00T_LIBERO_One_Shot"
RESULTS_ROOT="${WORK_DIR}/results/LIBERO/${EXP_NAME}"

SESSION="libero-sweep-$(date +%Y%m%d-%H%M%S)"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

# === 创建 tmux session: window 0 = serve ===
tmux new-session -d -s "$SESSION" -n "serve" -c "$WORK_DIR"

# Create windows 1-4 for each task suite
for i in "${!TASK_SUITES[@]}"; do
    tmux new-window -t "$SESSION" -n "${TASK_SUITES[$i]}" -c "$WORK_DIR"
done

# === Write the orchestrator script ===
ORCH_SCRIPT=$(mktemp /tmp/libero_sweep_orch_XXXX.sh)
cat > "$ORCH_SCRIPT" << 'ORCHESTRATOR_EOF'
#!/bin/bash
set -euo pipefail

WORK_DIR="__WORK_DIR__"
cd "$WORK_DIR"

GPU_ID="__GPU_ID__"
PORT="__PORT__"
NUM_TRIALS=__NUM_TRIALS__
USE_STATE="__USE_STATE__"
ACTION_CHUNK_SIZE="__ACTION_CHUNK_SIZE__"
SESSION="__SESSION__"

STEPS=(__STEPS__)
TASK_SUITES=(__TASK_SUITES__)

EXP_NAME="__EXP_NAME__"
CKPT_BASE="__CKPT_BASE__"
RESULTS_ROOT="__RESULTS_ROOT__"

star_vla_python="__STAR_VLA_PYTHON__"
LIBERO_HOME="__LIBERO_HOME__"
LIBERO_CONFIG_PATH="__LIBERO_CONFIG_PATH__"
LIBERO_Python="__LIBERO_PYTHON__"
CONDA_BIN="__CONDA_BIN__"

wait_for_server() {
    local port=$1
    local max_wait=300
    local waited=0
    echo "Waiting for server on port ${port}..."
    while ! ss -tln | grep -q ":${port} "; do
        sleep 3
        waited=$((waited + 3))
        if [ $waited -ge $max_wait ]; then
            echo "ERROR: Server did not start within ${max_wait}s"
            return 1
        fi
    done
    # Extra wait for model warmup
    sleep 15
    echo "Server is ready (waited $((waited + 15))s)."
}

for STEP in "${STEPS[@]}"; do
    CKPT_PATH="${CKPT_BASE}/steps_${STEP}_pytorch_model.pt"
    STEP_NAME="steps_${STEP}"

    if [ ! -f "$CKPT_PATH" ]; then
        echo "WARNING: Checkpoint not found: ${CKPT_PATH}, skipping."
        continue
    fi

    echo ""
    echo "######################################################"
    echo "# Step: ${STEP}  Checkpoint: ${CKPT_PATH}"
    echo "######################################################"

    STEP_RESULTS="${RESULTS_ROOT}/${STEP_NAME}"
    mkdir -p "${STEP_RESULTS}"

    # --- Start server in window 0 ---
    SERVE_CMD="cd ${WORK_DIR} && export PYTHONPATH=\$(pwd):\${PYTHONPATH:-} && CUDA_VISIBLE_DEVICES=${GPU_ID} ${star_vla_python} deployment/model_server/server_policy.py --ckpt_path ${CKPT_PATH} --port ${PORT} --use_bf16"
    if [ -n "${ACTION_CHUNK_SIZE}" ]; then
        SERVE_CMD="${SERVE_CMD} --action-chunk-size ${ACTION_CHUNK_SIZE}"
    fi
    tmux send-keys -t "${SESSION}:serve" "$SERVE_CMD" Enter

    wait_for_server ${PORT}

    # --- Clean up stale done flags ---
    for TASK_SUITE in "${TASK_SUITES[@]}"; do
        rm -f "${STEP_RESULTS}/.done_${TASK_SUITE}"
    done

    # --- Build use-state flag ---
    USE_STATE_FLAG=""
    if [ "${USE_STATE}" = "y" ] || [ "${USE_STATE}" = "Y" ]; then
        USE_STATE_FLAG="--args.use-state"
    fi

    # --- Launch all 4 evals in parallel ---
    for i in "${!TASK_SUITES[@]}"; do
        TASK_SUITE="${TASK_SUITES[$i]}"
        WIN_NAME="${TASK_SUITE}"
        VIDEO_DIR="${STEP_RESULTS}/${TASK_SUITE}"
        DONE_FLAG="${STEP_RESULTS}/.done_${TASK_SUITE}"

        EVAL_CMD="cd ${WORK_DIR}"
        EVAL_CMD="${EVAL_CMD} && source ${CONDA_BIN} libero"
        EVAL_CMD="${EVAL_CMD} && export LIBERO_HOME=${LIBERO_HOME}"
        EVAL_CMD="${EVAL_CMD} && export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
        EVAL_CMD="${EVAL_CMD} && export PYTHONPATH=\$(pwd):\${PYTHONPATH:-}:${LIBERO_HOME}"
        EVAL_CMD="${EVAL_CMD} && mkdir -p ${VIDEO_DIR}"
        EVAL_CMD="${EVAL_CMD} && ${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py"
        EVAL_CMD="${EVAL_CMD} --args.pretrained-path ${CKPT_PATH}"
        EVAL_CMD="${EVAL_CMD} --args.host 127.0.0.1"
        EVAL_CMD="${EVAL_CMD} --args.port ${PORT}"
        EVAL_CMD="${EVAL_CMD} --args.task-suite-name ${TASK_SUITE}"
        EVAL_CMD="${EVAL_CMD} --args.num-trials-per-task ${NUM_TRIALS}"
        EVAL_CMD="${EVAL_CMD} --args.video-out-path ${VIDEO_DIR}"
        EVAL_CMD="${EVAL_CMD} --args.results-dir ${STEP_RESULTS}"
        if [ -n "${USE_STATE_FLAG}" ]; then
            EVAL_CMD="${EVAL_CMD} ${USE_STATE_FLAG}"
        fi
        EVAL_CMD="${EVAL_CMD} 2>&1 | tee ${STEP_RESULTS}/${TASK_SUITE}_eval.log; touch ${DONE_FLAG}"

        tmux send-keys -t "${SESSION}:${WIN_NAME}" "$EVAL_CMD" Enter
    done

    # --- Wait for all 4 evals to finish ---
    echo "Waiting for all 4 task suites to finish for ${STEP_NAME}..."
    while true; do
        all_done=true
        for TASK_SUITE in "${TASK_SUITES[@]}"; do
            DONE_FLAG="${STEP_RESULTS}/.done_${TASK_SUITE}"
            if [ ! -f "$DONE_FLAG" ]; then
                all_done=false
                break
            fi
        done
        if $all_done; then
            break
        fi
        sleep 10
    done
    echo "All 4 suites finished for ${STEP_NAME}."

    # --- Print results ---
    for TASK_SUITE in "${TASK_SUITES[@]}"; do
        RESULTS_FILE="${STEP_RESULTS}/${TASK_SUITE}_results.txt"
        if [ -f "$RESULTS_FILE" ]; then
            echo ""
            cat "$RESULTS_FILE"
        fi
    done

    # --- Kill server ---
    echo ""
    echo "Stopping server..."
    tmux send-keys -t "${SESSION}:serve" C-c
    sleep 5
    echo "Server stopped."
done

echo ""
echo "======================================================"
echo "All sweeps finished. Collecting summary..."
echo "======================================================"

SUMMARY="${RESULTS_ROOT}/sweep_results.txt"
mkdir -p "$(dirname "$SUMMARY")"
{
    echo "Sweep: ${EXP_NAME}  trials=${NUM_TRIALS}  use_state=${USE_STATE}  action_chunk=${ACTION_CHUNK_SIZE:-auto}"
    echo "======================================================"
    printf "%-14s" "step"
    for ts in "${TASK_SUITES[@]}"; do
        printf "  %-18s" "$ts"
    done
    echo ""
    echo "----------------------------------------------------------------------"

    for STEP in "${STEPS[@]}"; do
        STEP_NAME="steps_${STEP}"
        printf "%-14s" "${STEP_NAME}"
        for TASK_SUITE in "${TASK_SUITES[@]}"; do
            RESULTS_FILE="${RESULTS_ROOT}/${STEP_NAME}/${TASK_SUITE}_results.txt"
            if [ -f "$RESULTS_FILE" ]; then
                rate=$(grep "Total Success Rate" "$RESULTS_FILE" | grep -oP '[0-9]+\.[0-9]+' | head -1)
                printf "  %-18s" "${rate:-N/A}"
            else
                printf "  %-18s" "N/A"
            fi
        done
        echo ""
    done
    echo "======================================================"
} | tee "$SUMMARY"

echo ""
echo "Summary saved to: ${SUMMARY}"
echo "Done! You can close this tmux session: tmux kill-session -t ${SESSION}"
ORCHESTRATOR_EOF

# === Replace placeholders in orchestrator script ===
sed -i "s|__WORK_DIR__|${WORK_DIR}|g" "$ORCH_SCRIPT"
sed -i "s|__GPU_ID__|${GPU_ID}|g" "$ORCH_SCRIPT"
sed -i "s|__PORT__|${PORT}|g" "$ORCH_SCRIPT"
sed -i "s|__NUM_TRIALS__|${NUM_TRIALS}|g" "$ORCH_SCRIPT"
sed -i "s|__USE_STATE__|${USE_STATE}|g" "$ORCH_SCRIPT"
sed -i "s|__ACTION_CHUNK_SIZE__|${ACTION_CHUNK_SIZE}|g" "$ORCH_SCRIPT"
sed -i "s|__SESSION__|${SESSION}|g" "$ORCH_SCRIPT"
sed -i "s|__STEPS__|${STEPS[*]}|g" "$ORCH_SCRIPT"
sed -i "s|__TASK_SUITES__|${TASK_SUITES[*]}|g" "$ORCH_SCRIPT"
sed -i "s|__EXP_NAME__|${EXP_NAME}|g" "$ORCH_SCRIPT"
sed -i "s|__CKPT_BASE__|${CKPT_BASE}|g" "$ORCH_SCRIPT"
sed -i "s|__RESULTS_ROOT__|${RESULTS_ROOT}|g" "$ORCH_SCRIPT"
sed -i "s|__STAR_VLA_PYTHON__|${star_vla_python}|g" "$ORCH_SCRIPT"
sed -i "s|__LIBERO_HOME__|${LIBERO_HOME}|g" "$ORCH_SCRIPT"
sed -i "s|__LIBERO_CONFIG_PATH__|${LIBERO_CONFIG_PATH}|g" "$ORCH_SCRIPT"
sed -i "s|__LIBERO_PYTHON__|${LIBERO_Python}|g" "$ORCH_SCRIPT"
sed -i "s|__CONDA_BIN__|${CONDA_BIN}|g" "$ORCH_SCRIPT"

chmod +x "$ORCH_SCRIPT"

# === Create orchestrator window and run ===
tmux new-window -t "$SESSION" -n "orchestrator" -c "$WORK_DIR"
tmux send-keys -t "${SESSION}:orchestrator" "bash ${ORCH_SCRIPT}" Enter

# Select the serve window by default
tmux select-window -t "${SESSION}:serve"

echo "=========================================="
echo "tmux session '${SESSION}' created with 6 windows:"
echo "  0: serve           - policy server"
echo "  1: libero_spatial   - eval"
echo "  2: libero_object    - eval"
echo "  3: libero_goal      - eval"
echo "  4: libero_10        - eval"
echo "  5: orchestrator     - sweep controller"
echo ""
echo "Config:"
echo "  GPU:        ${GPU_ID}"
echo "  Port:       ${PORT}"
echo "  Checkpoint: ${CKPT_BASE}"
echo "  Steps:      ${STEPS[*]}"
echo "  Use State:  ${USE_STATE}"
echo "  Act Chunk:  ${ACTION_CHUNK_SIZE:-auto}"
echo "  Results:    ${RESULTS_ROOT}"
echo ""
echo "Attach with: tmux attach -t ${SESSION}"
echo "=========================================="
