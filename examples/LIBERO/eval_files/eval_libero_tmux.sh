#!/bin/bash

###########################################################################################
# Script to run LIBERO evaluations for all 4 task suites in parallel using tmux
###########################################################################################

# Parse command-line arguments
if [ $# -eq 0 ]; then
    # Interactive mode: prompt for inputs
    read -p "Enter tmux session name [default: libero_eval]: " SESSION_NAME
    SESSION_NAME=${SESSION_NAME:-libero_eval}

    read -p "Enter base port [default: 5694]: " base_port_input
    base_port=${base_port_input:-5694}

    read -p "Enter checkpoint step (e.g., 20000): " ckpt_step
    if [ -z "$ckpt_step" ]; then
        echo "Error: checkpoint step is required"
        exit 1
    fi

    read -p "Enter experiment name [default: libero_eval]: " exp_name
    exp_name=${exp_name:-libero_eval}

    read -p "Use state? (y/n) [default: y]: " use_state_input
    use_state_input=${use_state_input:-y}
else
    # Command-line arguments mode
    SESSION_NAME=${1:-libero_eval}
    base_port=${2:-5694}
    ckpt_step=$3
    exp_name=${4:-libero_eval}
    use_state_input=${5:-y}

    if [ -z "$ckpt_step" ]; then
        echo "Usage: $0 [session_name] [base_port] <checkpoint_step> [exp_name] [use_state: y/n]"
        echo "Example: $0 libero_eval 5694 20000 my_experiment y"
        echo ""
        echo "Or run without arguments for interactive mode"
        exit 1
    fi
fi

# Build use_state flag
if [ "$use_state_input" = "y" ] || [ "$use_state_input" = "Y" ]; then
    use_state_flag="--args.use-state"
fi

# Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists. Please kill it first or attach to it."
    echo "To kill: tmux kill-session -t $SESSION_NAME"
    echo "To attach: tmux attach -t $SESSION_NAME"
    exit 1
fi

# Create new tmux session with first window
echo "Creating tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME -n "libero_spatial"

# Create additional windows
tmux new-window -t $SESSION_NAME -n "libero_object"
tmux new-window -t $SESSION_NAME -n "libero_goal"
tmux new-window -t $SESSION_NAME -n "libero_10"

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/mnt/nas/guchenyang/Code/vla-scratch/simulation/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/mnt/nas/guchenyang/miniconda3/envs/libero/bin/python

host="127.0.0.1"
# base_port is set from command-line arguments or interactive input above
your_ckpt="/mnt/nas/guchenyang/Code/starVLA/playground/LIBERO/0222_Qwen3_4B_PI_LIBERO_One_Shot/checkpoints/steps_${ckpt_step}_pytorch_model.pt"
num_trials_per_task=50
# === End of environment variable configuration ===
###########################################################################################

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"

# Results directory structure: results/<exp_name>/step_<ckpt_step>/<task_suite>/
results_base="results/${exp_name}/step_${ckpt_step}"

# Task suites to evaluate
task_suites=("libero_spatial" "libero_object" "libero_goal" "libero_10")

# Send commands to each window
for i in "${!task_suites[@]}"; do
    task_suite_name="${task_suites[$i]}"
    port=$((base_port))
    window_index=$i

    echo "Setting up window $window_index for task suite: $task_suite_name (port: $port)"

    # Send commands to the window
    tmux send-keys -t $SESSION_NAME:$window_index "cd /mnt/nas/guchenyang/Code/starVLA" C-m

    # Set up environment variables
    tmux send-keys -t $SESSION_NAME:$window_index "export LIBERO_HOME=$LIBERO_HOME" C-m
    tmux send-keys -t $SESSION_NAME:$window_index "export LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH" C-m
    tmux send-keys -t $SESSION_NAME:$window_index "export LIBERO_Python=$LIBERO_Python" C-m
    tmux send-keys -t $SESSION_NAME:$window_index "export PYTHONPATH=\$PYTHONPATH:\${LIBERO_HOME}" C-m
    tmux send-keys -t $SESSION_NAME:$window_index "export PYTHONPATH=\$(pwd):\${PYTHONPATH}" C-m

    # Activate conda environment
    tmux send-keys -t $SESSION_NAME:$window_index "source /mnt/nas/guchenyang/miniconda3/bin/activate libero" C-m

    # Create log directory
    tmux send-keys -t $SESSION_NAME:$window_index "mkdir -p $LOG_DIR" C-m

    # Set video output path: results/<exp_name>/step_<ckpt_step>/<task_suite>/
    video_out_path="${results_base}/${task_suite_name}"

    # Run evaluation command (as a single line)
    eval_cmd="\${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py --args.pretrained-path $your_ckpt --args.host $host --args.port $port --args.task-suite-name $task_suite_name --args.num-trials-per-task $num_trials_per_task --args.video-out-path $video_out_path --args.results-dir $results_base $use_state_flag"

    tmux send-keys -t $SESSION_NAME:$window_index "$eval_cmd" C-m
done

# Select the first window
tmux select-window -t $SESSION_NAME:0

echo ""
echo "=========================================="
echo "Tmux session '$SESSION_NAME' created successfully!"
echo "=========================================="
echo ""
echo "Four evaluation tasks are running in parallel:"
echo "  Window 0: libero_spatial (port: $base_port)"
echo "  Window 1: libero_object (port: $((base_port)))"
echo "  Window 2: libero_goal (port: $((base_port)))"
echo "  Window 3: libero_10 (port: $((base_port)))"
echo ""
echo "To attach to the session:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To switch between windows (inside tmux):"
echo "  Ctrl+b then 0/1/2/3"
echo ""
echo "To detach from session (inside tmux):"
echo "  Ctrl+b then d"
echo ""
echo "To kill the session:"
echo "  tmux kill-session -t $SESSION_NAME"
echo "=========================================="
