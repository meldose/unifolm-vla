# Resolve script directory and repository root (two levels up)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

# Checkpoint and VLM paths (override via env vars)
ckpt_path=${CKPT_PATH:-/path/to/your/Unifolm-VLA-Base/checkpoints/pytorch_model.pt}
vlm_pretrained_path=${VLM_PRETRAINED_PATH:-/path/to/your/Unifolm-VLM-Base}
# Server port and dataset unnormalization key
port=${PORT:-8777}
unnorm_key=${UNNORM_KEY:-g1_stack_block}

# Launch the real evaluation model server
python "${repo_root}/deployment/model_server/run_real_eval_server.py" \
    --ckpt_path "${ckpt_path}" \
    --port "${port}" \
    --unnorm_key "${unnorm_key}" \
    --vlm_pretrained_path "${vlm_pretrained_path}"
