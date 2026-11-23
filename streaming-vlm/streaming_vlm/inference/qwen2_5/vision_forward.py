from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from typing import Optional, Tuple
from torch.nn import functional as F
from types import MethodType
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    logger,
)

def streaming_visual_attention_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )

        if position_embeddings is None:
            logger.warning_once("Missing `position_embeddings`; computing cos/sin from `rotary_pos_emb`.")
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = eager_attention_forward
        if getattr(self.config, "_attn_implementation", "eager") != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if getattr(self.config, "_attn_implementation", "eager") == "flash_attention_2":
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output

def streaming_visual_block_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

    return hidden_states


def streaming_visual_encoder_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """Vision backbone: patch-embed → window reordering → multi-layer local/global attention → merge"""

    # 1. Patch → Token
    hidden_states = self.patch_embed(hidden_states)

    # 2. Rotary position encoding
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    # 3. window indexing
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens, device=hidden_states.device, dtype=torch.int32
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)  # Remove duplicate window start points

    # 4. Rearrange tokens within the same window to adjacent memory blocks
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit,
                                          self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index]
    hidden_states = hidden_states.reshape(seq_len, -1)

    # Position encoding is also reordered
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit,
                                            self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index].reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    # 5. Global cu_seqlens
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                         grid_thw[:, 0]).cumsum(dim=0).to(torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0))

    # 6. Swin blocks
    for idx, blk in enumerate(self.blocks):
        cu_now = cu_seqlens if idx in self.fullatt_block_indexes else cu_window_seqlens
        hidden_states = blk(hidden_states, cu_seqlens=cu_now, position_embeddings=position_embeddings)

    # 7. Merge + restore token order
    hidden_states = self.merger(hidden_states)
    hidden_states = hidden_states[torch.argsort(window_index)]

    return hidden_states
