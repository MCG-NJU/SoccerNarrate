# DPO Training Notes

This directory includes a lightweight DPO training path for StreamingVLM-style streaming commentary. It does not depend on `trl`; `train_dpo.py` computes the DPO objective directly from chosen/rejected JSONL samples.

## Data Format

The DPO dataset reader is `streaming_vlm/data/lmm_dataset_dpo.py`. Each sample provides a shared history plus two candidate continuations:

- `prefix_text_tokens`: previous commentary context.
- `prefix_video_secs`: optional video seconds before the preference window.
- `chunk_video_secs`: video seconds for the supervised preference window.
- `chosen_steps` and `rejected_steps`: word lists for the preferred and dispreferred continuations.
- `meta.video_path`: path to the source match video.

Only assistant tokens inside the current chunk are used for the DPO loss; prefix tokens are masked.

## Run

Single-node and multi-node runs use the same launcher:

```bash
cd streaming-vlm
export DATASET_PATH="$(pwd)/../data"
export PRETRAINED_MODEL_NAME_OR_PATH="/path/to/sft/checkpoint"
export REF_MODEL_NAME_OR_PATH="/path/to/reference/checkpoint"  # optional
export BETA=0.1
export LENGTH_NORM=True
bash tools/multi_nodes_dpo.sh
```

The launcher calls `tools/dpo.sh`, which runs `train_dpo.py` with DeepSpeed ZeRO-3 by default. For multi-node training, edit the `IP_LIST` array in `tools/multi_nodes_dpo.sh`.

## Useful Options

- `BETA`: DPO temperature coefficient.
- `LENGTH_NORM`: normalize log probabilities by supervised token count.
- `INCLUDE_PREFIX_VIDEO`: include or skip prefix video seconds in the visual context.
- `TEXT_SINK` and `TEXT_SLIDING_WINDOW`: control previous-text truncation.
