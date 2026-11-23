#!/bin/bash

set -e

#################### Environment and data config ####################
export VIDEO_MIN_PIXELS=78400 # 280*280. the minimum visual frame tokens sent to llm is 100
export FPS_MAX_FRAMES=64      # maximum number of frames for each video (480/60/2 = 4min)
export VIDEO_MAX_PIXELS=921600 # 1280*720=921600. the maximum overall video tokens sent to llm is 24k (leave 8k for language)
export VIDEO_TOTAL_PIXELS=10745600 # 128*640*360=29491200

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${PROJECT_DIR}/.." && pwd)

text_sink=${TEXT_SINK:-512}
TEXT_SLIDING_WINDOW=${TEXT_SLIDING_WINDOW:-512}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export DATASET_PATH=${DATASET_PATH:-"${REPO_ROOT}/data"}
export MODEL_NAME=${MODEL_NAME:-"qwen2_5_vl"}
export LEARNING_RATE=${LEARNING_RATE:-1e-5}   # can be overridden externally

cd "${PROJECT_DIR}"

########################################################
epoch_num=${EPOCH_NUM:-1}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-16}
model_name=${PRETRAINED_MODEL_NAME_OR_PATH:-"mit-han-lab/StreamingVLM"}
########################################################

timestamp=$(date +%Y%m%d_%H%M%S)
export RUN_NAME="${MODEL_NAME}_lr${LEARNING_RATE}_${timestamp}"
export OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints"} # no / at the end

read -r -a TRAIN_DATASET_NAMES <<< "${TRAIN_DATASET_NAMES:-sft.jsonl}"
read -r -a VALID_DATASET_NAMES <<< "${VALID_DATASET_NAMES:-sft.jsonl}"

TRAIN_FILES=("${TRAIN_DATASET_NAMES[@]/#/$DATASET_PATH/}")
VALID_FILES=("${VALID_DATASET_NAMES[@]/#/$DATASET_PATH/}")

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
echo "=========================================="
echo ""

#################### Launch with torchrun ####################
TOKENIZERS_PARALLELISM=false torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
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
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 1 \
    --load_best_model_at_end False \
    --greater_is_better False \
    --prediction_loss_only true \
    --eval_steps 300 \
    --metric_for_best_model eval_loss \
    --eval_strategy steps \
    --per_device_eval_batch_size 1 \
    --eval_annotation_paths "${VALID_FILES[@]}" \
    --text_sink ${text_sink} \
    --text_sliding_window ${TEXT_SLIDING_WINDOW}
