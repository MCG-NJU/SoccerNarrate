export VIDEO_MIN_PIXELS=78400 # 100*28*28. the minimum visual frame tokens sent to llm is 100
export FPS_MAX_FRAMES=240 # maximum number of frames for each video (480/60/2 = 4min)
export VIDEO_MAX_PIXELS=9437184 # 9437184 # 24576*28*28. the maximum overall video tokens sent to llm is 24k (leave 8k for language)
text_sink=512
TEXT_SLIDING_WINDOW=512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=9.0
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
export DATASET_PATH=${DATASET_PATH:-"/path/to/SoccerNet/asr/annos"}
export MODEL_NAME=${MODEL_NAME:-"qwen3vl"}
cd "${PROJECT_DIR}"
########################################################
epoch_num=1
gradient_accumulation_steps=1
model_name=${PRETRAINED_MODEL_NAME_OR_PATH:-"Qwen/Qwen3-VL-8B-Instruct"}
########################################################
timestamp=$(date +%Y%m%d_%H%M%S)
export RUN_NAME="${MODEL_NAME}_lr${LEARNING_RATE}_${timestamp}"
export OUTPUT_DIR="./checkpoints" # wo a / at the end

TRAIN_DATASET_NAMES=(
    "soccernet_train_with_seeks.jsonl"
)
VALID_DATASET_NAMES=(
    "soccernet_test_with_seeks.jsonl"
)

TRAIN_FILES=("${TRAIN_DATASET_NAMES[@]/#/$DATASET_PATH/}")
VALID_FILES=("${VALID_DATASET_NAMES[@]/#/$DATASET_PATH/}")

TOKENIZERS_PARALLELISM=false torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${GPUS} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER} \
  --master_port=${MASTER_PORT} \
 train.py \
    --deepspeed ./tools/zero3.json \
    --overwrite_output_dir True  \
    --output_dir "${OUTPUT_DIR}/${RUN_NAME}_${timestamp}" \
    --run_name $RUN_NAME \
    --save_on_each_node False \
    --do_train True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio 0.03 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --num_train_epochs $epoch_num \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --pretrained_model_name_or_path $model_name \
    --train_annotation_paths "${TRAIN_FILES[@]}"  \
    --dataloader_num_workers 4 \
    --use_liger_kernel True \
    --report_to none \
    --ignore_data_skip False \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 2 \
    --load_best_model_at_end False \
    --greater_is_better False \
    --prediction_loss_only true \
    --eval_steps 300 \
    --metric_for_best_model eval_loss \
    --eval_strategy steps \
    --per_device_eval_batch_size 1 \
    --eval_annotation_paths "${VALID_FILES[@]}" \
    --text_sink $text_sink \
    --text_sliding_window $TEXT_SLIDING_WINDOW

