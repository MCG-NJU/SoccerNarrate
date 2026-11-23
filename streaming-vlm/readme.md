# SoccerNarrate with StreamingVLM

This directory contains the StreamingVLM-based training and inference code used by the SoccerNarrate release. See the repository-level `README.md` for the full four-step workflow.

## Environment

```bash
bash scripts/env_infer.sh
bash scripts/env_sft.sh
```

Use `streamingvlm-infer` for inference and `streamingvlm-sft` for SFT/DPO training.

## SFT Training

```bash
export DATASET_PATH="$(pwd)/../data"
export PRETRAINED_MODEL_NAME_OR_PATH="mit-han-lab/StreamingVLM"
bash tools/multi_nodes.sh
```

The launcher calls `tools/sft_stage_1.sh`. Edit `IP_LIST` in `tools/multi_nodes.sh` for multi-node runs.

## DPO Training

```bash
export DATASET_PATH="$(pwd)/../data"
export PRETRAINED_MODEL_NAME_OR_PATH="/path/to/sft/checkpoint"
export REF_MODEL_NAME_OR_PATH="/path/to/reference/checkpoint"
bash tools/multi_nodes_dpo.sh
```

The launcher calls `tools/dpo.sh`, which runs `train_dpo.py` on the sample DPO JSONL format in `../data`.

## SoccerNarrate-Eval Inference

See [`streaming_vlm/inference/README.md`](streaming_vlm/inference/README.md) for detailed single-host and multi-host streaming inference instructions.

## SoccerNarrate-Eval Metrics

```bash
cd ..
export QWEN_API_URL="http://localhost:8000/v1/chat/completions"
export QWEN_API_KEY="your_api_key"

python eval/llm_as_a_judge/soccernet_vtt_event_mentions.py \
  --infer-root streaming-vlm/streaming_vlm/inference/infer_test/streamingvlm_all/test \
  --labels-root /path/to/SoccerNet-label \
  --num-workers 64 \
  --out-dir eval/llm_as_a_judge/results
```

The evaluator reports event recall, precision, and F1 in the summary JSON.
