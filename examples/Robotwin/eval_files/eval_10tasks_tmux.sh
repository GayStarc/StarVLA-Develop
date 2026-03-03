#!/bin/bash

# 使用 tmux 并行评测 10 个任务，4-3-3 分批（同时最多 4 个任务）
# 4 个窗口，每窗依次运行分配到的任务，实现 4+3+3 的并发模式

SESSION_NAME="eval_10tasks"
EVAL_FILES_PATH=$(cd "$(dirname "$0")" && pwd)

task_config=${1:-demo_clean}
ckpt_setting=${2:-qwen3_4b_oft_easy_10tasks_joint_abs_2w}
seed=${3:-0}
gpu_id=${4:-0}  # 所有任务共用此 GPU

TASKS=(
    "adjust_bottle"      # 0 -> 窗1
    "beat_block_hammer"  # 1 -> 窗2
    "blocks_ranking_rgb" # 2 -> 窗3
    "click_alarmclock"   # 3 -> 窗4
    "dump_bin_bigbin"    # 4 -> 窗1
    "grab_roller"        # 5 -> 窗2
    "stack_blocks_three" # 6 -> 窗3
    "stack_blocks_two"   # 7 -> 窗1
    "stack_bowls_three"  # 8 -> 窗2
    "stack_bowls_two"    # 9 -> 窗3
)

# 每窗的任务索引: 窗1(0,4,7) 窗2(1,5,8) 窗3(2,6,9) 窗4(3)
# 第1批: 4个并行 | 第2批: 3个并行 | 第3批: 3个并行

# 若已存在同名会话则先关闭
tmux has-session -t $SESSION_NAME 2>/dev/null && tmux kill-session -t $SESSION_NAME

tmux new-session -d -s $SESSION_NAME -n "batch1" -c "$EVAL_FILES_PATH"
tmux send-keys -t $SESSION_NAME:0 "source /root/miniconda3/bin/activate robotwin && ./eval.sh ${TASKS[0]} $task_config $ckpt_setting $seed $gpu_id && ./eval.sh ${TASKS[4]} $task_config $ckpt_setting $seed $gpu_id && ./eval.sh ${TASKS[7]} $task_config $ckpt_setting $seed $gpu_id" Enter

tmux new-window -t $SESSION_NAME -n "batch2" -c "$EVAL_FILES_PATH"
tmux send-keys -t $SESSION_NAME:1 "source /root/miniconda3/bin/activate robotwin && ./eval.sh ${TASKS[1]} $task_config $ckpt_setting $seed $gpu_id && ./eval.sh ${TASKS[5]} $task_config $ckpt_setting $seed $gpu_id && ./eval.sh ${TASKS[8]} $task_config $ckpt_setting $seed $gpu_id" Enter

tmux new-window -t $SESSION_NAME -n "batch3" -c "$EVAL_FILES_PATH"
tmux send-keys -t $SESSION_NAME:2 "source /root/miniconda3/bin/activate robotwin && ./eval.sh ${TASKS[2]} $task_config $ckpt_setting $seed $gpu_id && ./eval.sh ${TASKS[6]} $task_config $ckpt_setting $seed $gpu_id && ./eval.sh ${TASKS[9]} $task_config $ckpt_setting $seed $gpu_id" Enter

tmux new-window -t $SESSION_NAME -n "batch4" -c "$EVAL_FILES_PATH"
tmux send-keys -t $SESSION_NAME:3 "source /root/miniconda3/bin/activate robotwin && ./eval.sh ${TASKS[3]} $task_config $ckpt_setting $seed $gpu_id" Enter

echo "tmux 会话已创建: $SESSION_NAME (4 窗口，4-3-3 模式)"
echo "  - 窗1: ${TASKS[0]} → ${TASKS[4]} → ${TASKS[7]}"
echo "  - 窗2: ${TASKS[1]} → ${TASKS[5]} → ${TASKS[8]}"
echo "  - 窗3: ${TASKS[2]} → ${TASKS[6]} → ${TASKS[9]}"
echo "  - 窗4: ${TASKS[3]}"
echo "  - 参数: task_config=$task_config, ckpt_setting=$ckpt_setting, seed=$seed, gpu_id=$gpu_id"
echo ""
echo "附加到会话: tmux attach -t $SESSION_NAME"
echo "切换窗口: Ctrl+b 然后按 0-3"
