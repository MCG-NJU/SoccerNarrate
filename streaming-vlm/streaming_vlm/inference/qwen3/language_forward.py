from typing import List, Optional, Tuple, Union

import torch
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    Cache,
    Qwen3VLTextDecoderLayer,
    apply_rotary_pos_emb,
    rotate_half,
    create_causal_mask,
    eager_attention_forward,
)
from transformers.utils import logging

from streaming_vlm.inference.generate.streaming_cache import StreamingCache
from streaming_vlm.inference.streaming_args import StreamingArgs

logger = logging.get_logger(__name__)


def streaming_text_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    streaming_args: StreamingArgs = None,
    **kwargs,
):
    pos_mode = streaming_args.pos_mode if streaming_args is not None else "append"
    bsz, q_len, _ = hidden_states.size()
    hidden_shape = (bsz, q_len, -1, self.head_dim)

    def _align_rope(cos: torch.Tensor, sin: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure RoPE tables match the current sequence length (may differ when cache slices input_ids).
        if cos.size(-2) != seq_len:
            cos = cos[..., -seq_len:, :]
            sin = sin[..., -seq_len:, :]
        return cos, sin

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    if pos_mode == "append":
        cos, sin = _align_rope(cos, sin, q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if pos_mode == "shrink":
        # Key cache now includes past + new tokens; rotate with full length and slice tail for queries.
        def _apply_rope(x: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            c = c.unsqueeze(1)
            s = s.unsqueeze(1)
            return (x * c) + (rotate_half(x) * s)

        key_len = key_states.shape[-2]
        cos_k, sin_k = _align_rope(cos, sin, key_len)
        if key_len != q_len:
            cos_q = cos_k[..., -q_len:, :]
            sin_q = sin_k[..., -q_len:, :]
        else:
            cos_q, sin_q = cos_k, sin_k

        query_states = _apply_rope(query_states, cos_q, sin_q)
        key_states = _apply_rope(key_states, cos_k, sin_k)

    # Cast back to original dtype if norms upcasted to fp32
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            "The input hidden states seems to be silently casted in float32; casting back to %s.", target_dtype
        )
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    # Some attention kernels may return flattened outputs; infer actual seq length from the tensor to avoid shape drift.
    hidden_size = self.q_proj.out_features  # num_attention_heads * head_dim
    seq_len = attn_output.numel() // (bsz * hidden_size)
    attn_output = attn_output.reshape(bsz, seq_len, hidden_size).contiguous()
    if seq_len != q_len:
        attn_output = attn_output[:, -q_len:, :]
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights


def streaming_text_decoder_layer_forward(
    self: Qwen3VLTextDecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    streaming_args: StreamingArgs = None,
    **kwargs,
):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    attn_output, attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        streaming_args=streaming_args,
        output_attentions=output_attentions,
        **kwargs,
    )
    hidden_states = residual + attn_output

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return (hidden_states, attn_weights) if output_attentions else (hidden_states,)


def streaming_language_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    visual_pos_masks: Optional[torch.Tensor] = None,
    deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    streaming_args: StreamingArgs = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = StreamingCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            streaming_args=streaming_args,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = past_key_values if use_cache else None

    if streaming_args is not None and streaming_args.pos_mode == "shrink":
        streaming_args.input_ids = F.pad(streaming_args.input_ids, (0, 1), "constant", 0)

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
