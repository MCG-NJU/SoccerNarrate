export VIDEO_MIN_PIXELS=78400 # 100*28*28. the minimum visual frame tokens sent to llm is 100
export FPS_MAX_FRAMES=128 # maximum number of frames for each video (480/60/2 = 4min)
export VIDEO_MAX_PIXELS=4718592 # 24576*28*28. the maximum overall video tokens sent to llm is 24k (leave 8k for language)
export VIDEO_TOTAL_PIXELS=14745600 # 50*768*28*28
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GPUS=8

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."

learning_rate=2e-5 # sft uses 2e-5 lr
run_name="scnet_ac12_sft_fromlivecc_24k480x100_lr${learning_rate}_gpus${GPUS}"
TOKENIZERS_PARALLELISM=false torchrun --standalone --nproc_per_node=${GPUS} train.py \
  --deepspeed ./scripts/deepspeed_zero3.json \
  --output_dir checkpoints/$run_name \
  --overwrite_output_dir True \
  --run_name $run_name \
  --save_on_each_node False \
  --do_train True \
  --eval_strategy no \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate $learning_rate \
  --warmup_ratio 0.03 \
  --optim adamw_torch \
  --lr_scheduler_type cosine \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --save_steps 50 \
  --bf16 True \
  --tf32 True \
  --gradient_checkpointing True \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-8B-Instruct}" \
  --annotation_paths "${TRAIN_ANNOTATION_PATH:-/path/to/soccernet_train_with_seeks.jsonl}" \
  --dataloader_num_workers 1 \
  --freeze_modules visual \
  --use_liger_kernel True \
  --report_to none \
  --save_only_model True
