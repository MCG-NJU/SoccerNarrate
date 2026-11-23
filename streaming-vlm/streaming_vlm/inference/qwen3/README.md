# Qwen3-VL Streaming Patch

This folder contains the Qwen3-VL streaming adaptation used by the inference code.

- `pos_emb.py`: RoPE position handling for streaming video inputs.
- `language_forward.py`: language-model forward pass with `StreamingArgs` and `StreamingCache`.
- `vision_forward.py`: vision forward pass with precomputed position embeddings.
- `model_forward.py`: wrapper that injects visual features and updates the streaming cache.
- `patch_model.py`: `convert_qwen3_to_streaming(model)` monkey-patches a loaded Qwen3-VL model.

Basic usage:

```python
from streaming_vlm.inference.qwen3.patch_model import convert_qwen3_to_streaming

model = convert_qwen3_to_streaming(model)
```

When upgrading Transformers, check the signatures of Qwen3-VL causal mask, decoder layer, vision attention, and RoPE helper functions.
