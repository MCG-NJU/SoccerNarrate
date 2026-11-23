from types import MethodType

from transformers import Qwen3VLForConditionalGeneration

from streaming_vlm.inference.generate.prepare_generation import prepare_multiturn_multimodal_inputs_for_generation
from streaming_vlm.inference.generate.streaming_generate_qwen import _sample, streaming_generate
from streaming_vlm.inference.qwen3.language_forward import (
    streaming_language_model_forward,
    streaming_text_attn_forward,
    streaming_text_decoder_layer_forward,
)
from streaming_vlm.inference.qwen3.model_forward import model_forward, qwen3_vl_forward
from streaming_vlm.inference.qwen3.pos_emb import get_rope_index
from streaming_vlm.inference.qwen3.vision_forward import (
    streaming_visual_attention_forward,
    streaming_visual_block_forward,
    streaming_visual_encoder_forward,
)

# Defaults mirrored from the Qwen2.5 wrapper; kept here for parity/configuration.
DURATION_SEC = 4
WINDOW_SIZE = 4  # Ensure WINDOW_SIZE is divisible by DURATION_SEC
OUT_FPS = 1
MAX_TOKEN_PER_DURATION = 30
MAX_PIXELS = 360 * 640


def convert_qwen3_to_streaming(model: Qwen3VLForConditionalGeneration):
    # Patch generation helpers
    model.generate = MethodType(streaming_generate, model)
    model.prepare_inputs_for_generation = MethodType(prepare_multiturn_multimodal_inputs_for_generation, model)
    model._sample = MethodType(_sample, model)

    # Patch core forward paths
    model.forward = MethodType(qwen3_vl_forward, model)
    model.model.forward = MethodType(model_forward, model.model)
    model.model.language_model.forward = MethodType(streaming_language_model_forward, model.model.language_model)
    for layer in model.model.language_model.layers:
        layer.forward = MethodType(streaming_text_decoder_layer_forward, layer)
        layer.self_attn.forward = MethodType(streaming_text_attn_forward, layer.self_attn)

    # Patch vision tower to keep consistent entry points
    model.model.visual.forward = MethodType(streaming_visual_encoder_forward, model.model.visual)
    for blk in model.model.visual.blocks:
        blk.forward = MethodType(streaming_visual_block_forward, blk)
        blk.attn.forward = MethodType(streaming_visual_attention_forward, blk.attn)

    # Patch RoPE index computation
    model.model.get_rope_index = MethodType(get_rope_index, model.model)
    return model
