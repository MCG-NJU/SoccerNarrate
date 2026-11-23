# LLM-as-a-Judge Event Evaluation

This folder contains scripts for evaluating whether generated soccer commentary mentions SoccerNet key events at the right time.

## Main Script

Use `soccernet_vtt_event_mentions.py` to evaluate VTT outputs from SoccerNarrate-Eval inference:

```bash
export QWEN_API_URL="http://localhost:8000/v1/chat/completions"
export QWEN_API_KEY="your_api_key"

python eval/llm_as_a_judge/soccernet_vtt_event_mentions.py \
  --infer-root streaming-vlm/streaming_vlm/inference/infer_test/streamingvlm_all/test \
  --labels-root /path/to/SoccerNet-label \
  --num-workers 64 \
  --out-dir eval/llm_as_a_judge/results
```

## Method

The evaluator scans each VTT file with sliding windows. For windows that contain SoccerNet key events, it asks a judge model whether the generated commentary mentions each event. For negative windows, it checks whether the commentary incorrectly claims that selected events happened.

## Outputs

Each run writes:

- `sliding_window_event_mentions.jsonl`: per-window judge records.
- `sliding_window_event_mentions_summary.json`: aggregate metrics and configuration.
- `summary_index.json`: index file when evaluating multiple inference roots.

The summary includes event recall, faithfulness precision, temporal precision, and temporal F1. The default matching tolerance is controlled by `--delta-seconds`.
