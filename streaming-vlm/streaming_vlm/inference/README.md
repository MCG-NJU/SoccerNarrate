# Streaming Inference Guide

This directory contains the single-process real-time streaming inference pipeline for **SoccerNarrate** (built on StreamingVLM with bounded KV-cache reuse at 2 FPS).

## 1. Environment

Activate the inference environment created by `streaming-vlm/scripts/env_infer.sh`:

```bash
conda activate streamingvlm-infer
```

## 2. Single-Video Inference (`inference.py`)

Run real-time streaming commentary on any local soccer video clip or full match half (`.mp4` / `.mkv`) and save timestamped subtitles as a `.vtt` file:

```bash
python streaming-vlm/streaming_vlm/inference/inference.py \
  --model_path /path/to/soccernarrate/checkpoint \
  --model_base Qwen2_5 \
  --video_path demo/Goal.mp4 \
  --output_dir output/Goal.vtt
```

## 3. Single-Process Evaluation Inference (`baseline_infer.py`)

To run single-process inference across the **SoccerNarrate-Eval** test split (45 full matches / 90 halves) and write `.vtt` outputs matching the layout expected by `eval/llm_as_a_judge/soccernet_vtt_event_mentions.py`:

```bash
python streaming-vlm/streaming_vlm/inference/baseline_infer.py \
  --model_path /path/to/soccernarrate/checkpoint \
  --model_base Qwen2_5 \
  --annos_root /path/to/SoccerNet/asr/annos/test \
  --video_root /path/to/SoccerNet/videos \
  --output_root streaming-vlm/streaming_vlm/inference/infer_test/soccernarrate/test
```

The script loads the model once in a single process, iterates sequentially over the test halves, and writes:

```text
<output_root>/<league>_<season>/<match_name>/1_720p.vtt
<output_root>/<league>_<season>/<match_name>/2_720p.vtt
```

*(If `--annos_root` is omitted, `baseline_infer.py` directly scans all `<league>/<season>/<match_name>/{1,2}_720p.{mkv,mp4}` videos under `--video_root`.)*

### Key Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--model_path` | — | Local checkpoint path or HuggingFace model ID |
| `--model_base` | `Qwen2_5` | Backbone family (`Qwen2_5` or `Qwen2`) |
| `--video_root` | `$SOCCERNET_VIDEOS_ROOT` | Root directory containing SoccerNet `<league>/<season>/<match>/{1,2}_720p.*` videos |
| `--annos_root` | `None` | Optional split directory (e.g., `SoccerNet/asr/annos/test`) to select the evaluation split |
| `--output_root` | — | Output directory for generated `.vtt` commentary files |
| `--window_size` | `16` | Visual sliding window size in seconds (32 frames at 2 FPS) |
| `--text_sink` / `--text_sliding_window` | `512` / `512` | Attention sink and recent text KV-cache budget |
| `--temperature` / `--repetition_penalty` | `0.9` / `1.05` | Sampling decoding parameters |
| `--overwrite` | `False` | Re-run and overwrite existing `.vtt` outputs |

<details>
<summary>Optional: manual multi-GPU sharding</summary>

If you have multiple GPUs on a machine and want to split the 90 test halves across GPUs manually, run separate single-process commands with `--num_shards` and `--shard_id`:

```bash
CUDA_VISIBLE_DEVICES=0 python streaming-vlm/streaming_vlm/inference/baseline_infer.py \
  --model_path /path/to/soccernarrate/checkpoint \
  --annos_root /path/to/SoccerNet/asr/annos/test \
  --video_root /path/to/SoccerNet/videos \
  --output_root streaming-vlm/streaming_vlm/inference/infer_test/soccernarrate/test \
  --num_shards 2 --shard_id 0
```

</details>
