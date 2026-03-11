# Resolve script directory and repository root (two levels up)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

# LIBERO paths (override by exporting LIBERO_HOME/LIBERO_CONFIG_PATH)
export LIBERO_HOME=${LIBERO_HOME:-/path/to/your/LIBERO}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}

# Add LIBERO and repo to PYTHONPATH for imports
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=${repo_root}:${repo_root}/src:${PYTHONPATH}


# Checkpoint and pretrained VLM paths (override via env vars)
your_ckpt=${CKPT_PATH:-/path/to/your/Unifolm-VLA-Libero/checkpoints/pytorch_model.pt}
vlm_pretrained_path=${VLM_PRETRAINED_PATH:-/path/to/your/Unifolm-VLM-Base}
# Derive folder/step names from checkpoint path for output structure
folder_name=$(basename "$(dirname "$(dirname "${your_ckpt}")")")
step_name=$(basename "$(dirname "${your_ckpt}")")
# Task suite and evaluation settings
task_suite_name=${TASK_SUITE_NAME:-libero_spatial}   # libero_goal, libero_object, libero_10, libero_90
num_trials_per_task=${NUM_TRIALS_PER_TASK:-50}
window_size=${WINDOW_SIZE:-2}
unnorm_key=${UNNORM_KEY:-libero_spatial_no_noops}  # libero_goal_no_noops, libero_object_no_noops, libero_10_no_noops, libero_90_no_noops

# Output path for videos/results
video_out_path="results/${task_suite_name}/${folder_name}/${step_name}"

# GPU device id (CUDA_VISIBLE_DEVICES)
DEVICE=${DEVICE:-0}

# Run evaluation with the correct PYTHONPATH and GPU selection
PYTHONPATH="${repo_root}:${repo_root}/src:${LIBERO_HOME}:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=${DEVICE} python "${repo_root}/experiments/LIBERO/eval_libero.py" \
    --args.pretrained-path ${your_ckpt} \
    --args.vlm-pretrained-path ${vlm_pretrained_path} \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    --args.unnorm-key "$unnorm_key" \
    --args.window-size "$window_size"
