# Streaming Inference Guide

This directory contains the real-time streaming inference pipeline for **SoccerNarrate** (built on StreamingVLM with bounded KV-cache reuse at 2 FPS).

## 1. Environment

Activate the inference environment created by `streaming-vlm/scripts/env_infer.sh`:

```bash
conda activate streamingvlm-infer
```

## 2. Batch Full-Match Inference (`infer_streamingvlm_all.sh`)

Run batched streaming inference over SoccerNet-style annotations to generate per-half WebVTT (`.vtt`) commentary files:

```bash
cd streaming-vlm

export MODEL_PATH="/path/to/soccernarrate/checkpoint"
export MODEL_BASE="Qwen2_5"                               # Qwen2_5 or Qwen2
export HOSTS="127.0.0.1"                                  # comma-separated SSH hosts for multi-host runs
export REMOTE_SETUP="conda activate streamingvlm-infer"   # environment activation command on remote hosts
export ANNOS_ROOT="/path/to/SoccerNet/asr/annos"
export VIDEO_ROOT="/path/to/SoccerNet/videos"
export OUTPUT_ROOT_BASE="$(pwd)/streaming_vlm/inference/infer_test/streamingvlm_all"
export SPLITS="test"

bash streaming_vlm/inference/infer_streamingvlm_all.sh
```

The script invokes `baseline_infer.py` and writes per-half `.vtt` commentary outputs mirroring `ANNOS_ROOT` under:

```text
${OUTPUT_ROOT_BASE}/${split}/<league_season>/<match>/1_720p.vtt
${OUTPUT_ROOT_BASE}/${split}/<league_season>/<match>/2_720p.vtt
```

## 3. Evaluating Multiple Checkpoints (`infer.sh`)

To run inference across multiple checkpoints sequentially, edit the `model_paths` and `output_roots` arrays in `infer.sh` (keeping the same order) and run:

```bash
cd streaming-vlm
export HOSTS="127.0.0.1"
export ANNOS_ROOT="/path/to/SoccerNet/asr/annos/test"
export VIDEO_ROOT="/path/to/SoccerNet/videos"
bash streaming_vlm/inference/infer.sh
```

## 4. Direct Single-Host CLI (`baseline_infer.py`)

You can also invoke `baseline_infer.py` directly (it automatically distributes jobs across all visible GPUs on the local machine unless `--disable_auto_gpus` is set):

```bash
python streaming-vlm/streaming_vlm/inference/baseline_infer.py \
  --annos_root /path/to/SoccerNet/asr/annos/test \
  --video_root /path/to/SoccerNet/videos \
  --output_root streaming-vlm/streaming_vlm/inference/infer_test/soccernarrate/test \
  --model_path /path/to/soccernarrate/checkpoint \
  --model_base Qwen2_5
```

### Key Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--model_path` | — | Local checkpoint path or HuggingFace model ID |
| `--model_base` | `Qwen2_5` | Backbone family (`Qwen2_5` or `Qwen2`) |
| `--annos_root` | — | Directory tree of match annotations to mirror |
| `--video_root` | `$SOCCERNET_VIDEOS_ROOT` | Root directory containing SoccerNet `<league>/<season>/<match>/[1|2]_720p.*` videos |
| `--output_root` | — | Output directory for generated `.vtt` commentary files |
| `--window_size` | `16` | Visual sliding window size in seconds (32 frames at 2 FPS) |
| `--text_sink` / `--text_sliding_window` | `512` / `512` | Attention sink and recent text KV-cache budget |
| `--temperature` / `--repetition_penalty` | `0.9` / `1.05` | Sampling decoding parameters |
| `--hosts` | `""` | Comma-separated SSH targets for multi-host sharded inference (e.g., `host1,host2`) |
| `--write_sentence_txt` | `False` | Optionally write sentence-merged `.txt` transcripts alongside `.vtt` |
