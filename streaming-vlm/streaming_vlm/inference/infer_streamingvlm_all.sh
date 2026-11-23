#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
REPO_ROOT=$(cd "${PROJECT_DIR}/.." && pwd)

SPLITS=(${SPLITS:-test})
HOSTS=${HOSTS:-"127.0.0.1"}
WORKDIR=${WORKDIR:-"${REPO_ROOT}"}
REMOTE_SETUP=${REMOTE_SETUP:-"source ${PROJECT_DIR}/venv/bin/activate"}
MULTI_HOST_LOG_DIR=${MULTI_HOST_LOG_DIR:-"${SCRIPT_DIR}/log"}
ANNOS_ROOT=${ANNOS_ROOT:-"/path/to/SoccerNet/asr/annos"}
VIDEO_ROOT=${VIDEO_ROOT:-"/path/to/SoccerNet/videos"}
OUTPUT_ROOT_BASE=${OUTPUT_ROOT_BASE:-"${SCRIPT_DIR}/infer_test/streamingvlm_all"}
MODEL_PATH=${MODEL_PATH:-"mit-han-lab/StreamingVLM"}
MODEL_BASE=${MODEL_BASE:-"Qwen2_5"}

for split in "${SPLITS[@]}"; do
  python "${SCRIPT_DIR}/baseline_infer.py" \
    --hosts "${HOSTS}" \
    --workdir "${WORKDIR}" \
    --remote_setup "${REMOTE_SETUP}" \
    --multi_host_log_dir "${MULTI_HOST_LOG_DIR}" \
    --annos_root "${ANNOS_ROOT}/${split}" \
    --video_root "${VIDEO_ROOT}" \
    --output_root "${OUTPUT_ROOT_BASE}/${split}" \
    --model_path "${MODEL_PATH}" \
    --model_base "${MODEL_BASE}"
done

if [[ -n "${KEEP_GPUS_BUSY_SCRIPT:-}" && -f "${KEEP_GPUS_BUSY_SCRIPT}" ]]; then
  bash "${KEEP_GPUS_BUSY_SCRIPT}"
fi
