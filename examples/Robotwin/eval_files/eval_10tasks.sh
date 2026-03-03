#!/bin/bash

# 基于 eval.sh 的批量评测脚本
# 评测 convert_test.sh 中的十个任务的成功率

ROBOTWIN_PATH=/mnt/cpfs/guchenyang/Code/RoboTwin

policy_name="model2robotwin_interface"
task_config=${1:-demo_clean}
ckpt_setting=${2:-qwen3_4b_oft_easy_10tasks_joint_abs_5w}
seed=${3:-0}
gpu_id=${4:-0}

# 十个评测任务（与 convert_test.sh 一致）
TASKS=(
    "adjust_bottle"
    "beat_block_hammer"
    "blocks_ranking_rgb"
    "click_alarmclock"
    "dump_bin_bigbin"
    "grab_roller"
    "stack_blocks_three"
    "stack_blocks_two"
    "stack_bowls_three"
    "stack_bowls_two"
)

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33m=== 批量评测 10 任务 ===\033[0m"
echo -e "\033[33mGPU: ${gpu_id} | task_config: ${task_config} | ckpt_setting: ${ckpt_setting} | seed: ${seed}\033[0m"
echo ""

EVAL_FILES_PATH=$(cd "$(dirname "$0")" && pwd)
STARVLA_PATH=$EVAL_FILES_PATH/../../..
DEPLOY_POLICY_PATH=$EVAL_FILES_PATH/deploy_policy.yml

export PYTHONPATH=$ROBOTWIN_PATH:$PYTHONPATH
export PYTHONPATH=$STARVLA_PATH:$PYTHONPATH
export PYTHONPATH=$EVAL_FILES_PATH:$PYTHONPATH

cd $ROBOTWIN_PATH

# 存储各任务成功率: task_name suc_num total
declare -A TASK_RESULTS
TOTAL_SUC=0
TOTAL_EPISODES=0
LOG_DIR=$EVAL_FILES_PATH/eval_10tasks_logs
mkdir -p $LOG_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/eval_${TIMESTAMP}.log

echo "评测日志: $LOG_FILE"
echo ""

for task_name in "${TASKS[@]}"; do
    echo -e "\033[36m----------------------------------------\033[0m"
    echo -e "\033[36m评测任务: ${task_name}\033[0m"
    echo -e "\033[36m----------------------------------------\033[0m"

    TASK_LOG=$LOG_DIR/${task_name}_${TIMESTAMP}.log

    PYTHONWARNINGS=ignore::UserWarning \
    python script/eval_policy.py --config $DEPLOY_POLICY_PATH \
        --overrides \
        --task_name ${task_name} \
        --task_config ${task_config} \
        --ckpt_setting ${ckpt_setting} \
        --seed ${seed} \
        --policy_name ${policy_name} 2>&1 | tee $TASK_LOG

    # 解析成功率: 从输出中提取最后的 "Success rate: X/Y => Z%"
    SUCCESS_LINE=$(grep "Success rate:" $TASK_LOG | tail -1)
    if [[ -n "$SUCCESS_LINE" ]]; then
        # 解析 X/Y 格式（兼容 ANSI 转义码），每个任务运行 100 个 episode
        RATE_PAIR=$(echo "$SUCCESS_LINE" | grep -oP '\d+/\d+' | tail -1)
        SUC_NUM=$(echo "$RATE_PAIR" | cut -d'/' -f1)
        TOTAL_NUM=$(echo "$RATE_PAIR" | cut -d'/' -f2)
        RATE_PCT=$(echo "$SUCCESS_LINE" | grep -oP '\d+\.?\d*%' | tail -1)

        if [[ -n "$SUC_NUM" && -n "$TOTAL_NUM" ]]; then
            TASK_RESULTS[$task_name]="${SUC_NUM}/${TOTAL_NUM} (${RATE_PCT})"
            TOTAL_SUC=$((TOTAL_SUC + SUC_NUM))
            TOTAL_EPISODES=$((TOTAL_EPISODES + TOTAL_NUM))
            echo -e "\033[32m${task_name}: ${SUC_NUM}/${TOTAL_NUM} = ${RATE_PCT}\033[0m"
        else
            TASK_RESULTS[$task_name]="解析失败"
            echo -e "\033[31m${task_name}: 解析失败\033[0m"
        fi
    else
        TASK_RESULTS[$task_name]="运行失败"
        echo -e "\033[31m${task_name}: 运行失败\033[0m"
    fi
    echo ""
done

# 打印汇总结果
echo -e "\033[33m========================================\033[0m"
echo -e "\033[33m           评测汇总结果\033[0m"
echo -e "\033[33m========================================\033[0m"

for task_name in "${TASKS[@]}"; do
    printf "  %-25s %s\n" "$task_name:" "${TASK_RESULTS[$task_name]}"
done

echo ""
if [[ $TOTAL_EPISODES -gt 0 ]]; then
    OVERALL_RATE=$(echo "scale=2; $TOTAL_SUC * 100 / $TOTAL_EPISODES" | bc 2>/dev/null || awk "BEGIN {printf \"%.1f\", $TOTAL_SUC*100/$TOTAL_EPISODES}")
    echo -e "\033[95m总体成功率: ${TOTAL_SUC}/${TOTAL_EPISODES} = ${OVERALL_RATE}%\033[0m"
else
    echo -e "\033[31m总体成功率: 无法计算\033[0m"
fi

echo ""
echo "评测完成. 详细日志保存在: $LOG_DIR"
