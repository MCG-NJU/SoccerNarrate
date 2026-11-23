#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
REPO_ROOT=$(cd "${PROJECT_DIR}/.." && pwd)

HOSTS=${HOSTS:-"127.0.0.1"}
WORKDIR=${WORKDIR:-"${REPO_ROOT}"}
REMOTE_SETUP=${REMOTE_SETUP:-"source ${PROJECT_DIR}/venv/bin/activate"}
MULTI_HOST_LOG_DIR=${MULTI_HOST_LOG_DIR:-"${SCRIPT_DIR}/log"}
ANNOS_ROOT=${ANNOS_ROOT:-"/path/to/SoccerNet/asr/annos/test"}
VIDEO_ROOT=${VIDEO_ROOT:-"/path/to/SoccerNet/videos"}
MODEL_BASE=${MODEL_BASE:-"Qwen2_5"}

# Keep the same order: model_paths[i] maps to output_roots[i].
model_paths=(
  "${MODEL_PATH:-/path/to/soccernarrate/checkpoint}"
)

output_roots=(
  "${OUTPUT_ROOT:-${SCRIPT_DIR}/infer_test/soccernarrate}"
)

if [ "${#model_paths[@]}" -ne "${#output_roots[@]}" ]; then
  echo "Error: model_paths and output_roots must have the same length." >&2
  echo "model_paths=${#model_paths[@]}, output_roots=${#output_roots[@]}" >&2
  exit 1
fi

total="${#model_paths[@]}"
for i in "${!model_paths[@]}"; do
  model_path="${model_paths[$i]}"
  output_root="${output_roots[$i]}"

  echo "[$((i + 1))/$total] Start inference"
  echo "model_path:  $model_path"
  echo "output_root: $output_root"

  python "${SCRIPT_DIR}/baseline_infer.py" \
    --hosts "${HOSTS}" \
    --workdir "${WORKDIR}" \
    --remote_setup "${REMOTE_SETUP}" \
    --multi_host_log_dir "${MULTI_HOST_LOG_DIR}" \
    --annos_root "${ANNOS_ROOT}" \
    --video_root "${VIDEO_ROOT}" \
    --output_root "$output_root" \
    --model_path "$model_path" \
    --model_base "${MODEL_BASE}"
done
