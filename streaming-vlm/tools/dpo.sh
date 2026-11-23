#!/bin/bash

set -e

#################### Environment and data config ####################
# qwen_vl_utils (>=0.0.14) uses per-video dict keys to control sampling/resize:
#   - frames: `fps` + `min_frames` / `max_frames`
#   - size:   `min_pixels` / `max_pixels` / `total_pixels`
# `streaming_vlm/data/lmm_dataset_dpo.py` reads the env vars below and forwards them into qwen_vl_utils.
export QWENVL_VIDEO_FPS=${QWENVL_VIDEO_FPS:-2.0}
export QWENVL_VIDEO_MIN_FRAMES=${QWENVL_VIDEO_MIN_FRAMES:-2}
export QWENVL_VIDEO_MAX_FRAMES=${QWENVL_VIDEO_MAX_FRAMES:-4}
export QWENVL_VIDEO_MIN_PIXELS=${QWENVL_VIDEO_MIN_PIXELS:-78400}     # 280*280 -> 100 visual tokens (28*28 per token)
export QWENVL_VIDEO_MAX_PIXELS=${QWENVL_VIDEO_MAX_PIXELS:-921600}     # will be clamped by qwen_vl_utils internal limit
export QWENVL_VIDEO_TOTAL_PIXELS=${QWENVL_VIDEO_TOTAL_PIXELS:-7745600}

# Legacy aliases (some older code paths / scripts still read these names).
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-${QWENVL_VIDEO_MAX_FRAMES}}
export VIDEO_MIN_PIXELS=${VIDEO_MIN_PIXELS:-${QWENVL_VIDEO_MIN_PIXELS}}
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-${QWENVL_VIDEO_MAX_PIXELS}}
export VIDEO_TOTAL_PIXELS=${VIDEO_TOTAL_PIXELS:-${QWENVL_VIDEO_TOTAL_PIXELS}}

text_sink=${TEXT_SINK:-512}
TEXT_SLIDING_WINDOW=${TEXT_SLIDING_WINDOW:-512}

export LMM_DPO_VISION_DTYPE=bf16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export MODEL_NAME=${MODEL_NAME:-"qwen2_5_vl"}
export LEARNING_RATE=${LEARNING_RATE:-1e-5}   # can be overridden externally
export BETA=${BETA:-0.1}
export LENGTH_NORM=${LENGTH_NORM:-False}
export INCLUDE_PREFIX_VIDEO=${INCLUDE_PREFIX_VIDEO:-True}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${PROJECT_DIR}/.." && pwd)
export DATASET_PATH=${DATASET_PATH:-"${REPO_ROOT}/data"}
cd "${PROJECT_DIR}"

########################################################
epoch_num=1
gradient_accumulation_steps=16
model_name=${PRETRAINED_MODEL_NAME_OR_PATH:-"mit-han-lab/StreamingVLM"}
########################################################

timestamp=$(date +%Y%m%d_%H%M%S)
export RUN_NAME="${MODEL_NAME}_dpo_beta${BETA}_lr${LEARNING_RATE}_${timestamp}"
export OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints"} # no / at the end

read -r -a TRAIN_DATASET_NAMES <<< "${TRAIN_DATASET_NAMES:-dpo_cross_event_15s.jsonl}"

TRAIN_FILES=("${TRAIN_DATASET_NAMES[@]/#/$DATASET_PATH/}")

#################### Distributed config for torchrun ####################
# These variables can be injected by multi-node scripts or use local defaults.
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29501}
export NNODES=${NNODES:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-2}
export NODE_RANK=${NODE_RANK:-0}

echo "=========================================="
echo "Torchrun Training Config"
echo "------------------------------------------"
echo "MASTER_ADDR:      ${MASTER_ADDR}"
echo "MASTER_PORT:      ${MASTER_PORT}"
echo "NNODES:           ${NNODES}"
echo "GPUS_PER_NODE:    ${GPUS_PER_NODE}"
echo "NODE_RANK:        ${NODE_RANK}"
echo "RUN_NAME:         ${RUN_NAME}"
echo "OUTPUT_DIR:       ${OUTPUT_DIR}"
echo "LEARNING_RATE:    ${LEARNING_RATE}"
echo "BETA:             ${BETA}"
echo "LENGTH_NORM:      ${LENGTH_NORM}"
echo "=========================================="
echo ""

extra_dpo_args=(
    --beta "${BETA}"
    --length_norm "${LENGTH_NORM}"
    --include_prefix_video "${INCLUDE_PREFIX_VIDEO}"
)
if [[ -n "${REF_MODEL_NAME_OR_PATH:-}" ]]; then
    extra_dpo_args+=(--ref_model_name_or_path "${REF_MODEL_NAME_OR_PATH}")
fi

#################### Launch with torchrun ####################
TOKENIZERS_PARALLELISM=false torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train_dpo.py \
    --deepspeed "${PROJECT_DIR}/scripts/zero3.json" \
    --overwrite_output_dir True  \
    --output_dir "${OUTPUT_DIR}/${RUN_NAME}" \
    --run_name "${RUN_NAME}" \
    --save_on_each_node False \
    --do_train True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio 0.03 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --num_train_epochs ${epoch_num} \
    --logging_steps 1 \
    --bf16 True \
    --gradient_checkpointing True \
    --pretrained_model_name_or_path "${model_name}" \
    --train_annotation_paths "${TRAIN_FILES[@]}" \
    --dataloader_num_workers 4 \
    --use_liger_kernel True \
    --report_to none \
    --ignore_data_skip False \
    --remove_unused_columns False \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 1 \
    --load_best_model_at_end False \
    --greater_is_better False \
    --prediction_loss_only true \
    --text_sink ${text_sink} \
    --text_sliding_window ${TEXT_SLIDING_WINDOW} \
    "${extra_dpo_args[@]}" \
    --eval_strategy no
