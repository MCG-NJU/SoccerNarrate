#!/bin/bash
set -e

# Enable offline mode to avoid blocking network requests
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export VIDEO_MIN_PIXELS=78400
export FPS_MAX_FRAMES=240
export VIDEO_MAX_PIXELS=9437184
export TORCH_CUDA_ARCH_LIST=9.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_NAME="qwen3vl"
export DATASET_TYPE="12"
export MIN_CLIP_SEC=16
export MAX_CLIP_SEC=64
export MAX_SILENCE_SEC=3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."

TOKENIZERS_PARALLELISM=false torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${GPUS} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER} \
  --master_port=${MASTER_PORT} \
  train.py \
  --deepspeed ./scripts/deepspeed_zero3.json \
  --output_dir checkpoints/${MODEL_NAME}/asr${DATASET_TYPE}_lr${LEARNING_RATE}_nodes${NNODES}_gpus${GPUS} \
  --overwrite_output_dir True \
  --run_name ${MODEL_NAME}_lr${LEARNING_RATE}_${MIN_CLIP_SEC}_${MAX_CLIP_SEC}_${MAX_SILENCE_SEC} \
  --do_train True \
  --eval_strategy no \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate ${LEARNING_RATE} \
  --warmup_ratio 0.02 \
  --optim adamw_torch \
  --lr_scheduler_type cosine \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --save_steps 20 \
  --save_total_limit 1 \
  --bf16 True \
  --gradient_checkpointing True \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-8B-Instruct}" \
  --annotation_paths "${TRAIN_ANNOTATION_PATH:-/path/to/soccernet_train_with_seeks.jsonl}" \
  --dataloader_num_workers 2 \
  --freeze_modules visual \
  --use_liger_kernel True \
  --report_to none \
  --save_only_model True
