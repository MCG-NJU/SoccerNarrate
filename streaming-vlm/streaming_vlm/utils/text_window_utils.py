from __future__ import annotations

from typing import Any, Dict, Optional


def _as_token_list(input_ids) -> list[int]:
    if input_ids is None:
        return []
    if hasattr(input_ids, "flatten"):
        try:
            return input_ids.flatten().tolist()
        except Exception:
            pass
    if isinstance(input_ids, (list, tuple)):
        return list(input_ids)
    return []


def _decode_window(tokenizer, token_ids) -> str:
    if tokenizer is None or not token_ids:
        return ""
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception:
        return ""


def _starts_at_safe_boundary(text: str) -> bool:
    if not text:
        return False
    first_char = text[0]
    if first_char.isspace():
        return True
    return not first_char.isalnum()


def compute_middle_truncation(
    *,
    input_ids,
    body_start: int,
    body_end: int,
    keep_left: int,
    keep_right: int,
    tokenizer=None,
    align_right_boundary: bool = True,
    max_right_shift: int = 32,
    preview_tokens: int = 8,
) -> Dict[str, Any]:
    ids = _as_token_list(input_ids)
    keep_left = max(0, int(keep_left or 0))
    keep_right = max(0, int(keep_right or 0))

    info: Dict[str, Any] = {
        "cut_start": None,
        "cut_end": None,
        "body_start": int(body_start),
        "body_end": int(body_end),
        "keep_left": keep_left,
        "keep_right": keep_right,
        "initial_right_start": None,
        "adjusted_right_start": None,
        "right_shift": 0,
        "left_preview": "",
        "right_preview_before": "",
        "right_preview_after": "",
    }

    if not ids or body_start > body_end:
        return info

    body_len = body_end - body_start + 1
    if body_len <= 0 or keep_left + keep_right >= body_len:
        return info

    cut_start = body_start + keep_left
    initial_right_start = body_end - keep_right + 1 if keep_right > 0 else body_end + 1
    adjusted_right_start = initial_right_start

    if align_right_boundary and tokenizer is not None and keep_right > 0:
        max_candidate = min(body_end, initial_right_start + max(0, int(max_right_shift)))
        while adjusted_right_start <= max_candidate:
            head_ids = ids[adjusted_right_start : min(body_end + 1, adjusted_right_start + preview_tokens)]
            head_text = _decode_window(tokenizer, head_ids)
            if _starts_at_safe_boundary(head_text):
                break
            adjusted_right_start += 1
        if adjusted_right_start > body_end:
            adjusted_right_start = initial_right_start

    cut_end = adjusted_right_start - 1
    if cut_start > cut_end:
        return info

    left_preview = _decode_window(tokenizer, ids[max(body_start, cut_start - preview_tokens) : cut_start])
    right_preview_before = _decode_window(
        tokenizer,
        ids[initial_right_start : min(body_end + 1, initial_right_start + preview_tokens)],
    )
    right_preview_after = _decode_window(
        tokenizer,
        ids[adjusted_right_start : min(body_end + 1, adjusted_right_start + preview_tokens)],
    )

    info.update(
        {
            "cut_start": int(cut_start),
            "cut_end": int(cut_end),
            "initial_right_start": int(initial_right_start),
            "adjusted_right_start": int(adjusted_right_start),
            "right_shift": int(adjusted_right_start - initial_right_start),
            "left_preview": left_preview,
            "right_preview_before": right_preview_before,
            "right_preview_after": right_preview_after,
        }
    )
    return info
