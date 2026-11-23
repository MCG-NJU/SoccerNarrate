SYSTEM_PROMPT_OFFSET = 58
TOKEN_IDS_DEFAULT = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "user": 872,
    "assistant": 77091,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|video_pad|>": 151656,
    "\n": 198,
    "Time": 1462,
}

# Mutable runtime copy so callers can inject tokenizer-specific ids (Qwen3 may differ slightly).
TOKEN_IDS = dict(TOKEN_IDS_DEFAULT)


def _token_id(tokenizer, text: str) -> int | None:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None


def set_token_id_map(tokenizer_or_map):
    """
    Override TOKEN_IDS with tokenizer-specific ids (needed when IDs differ across Qwen generations).

    Accepts either a tokenizer or a prebuilt dict of token ids.
    """
    global TOKEN_IDS
    if hasattr(tokenizer_or_map, "__call__"):
        tok = tokenizer_or_map
        maybe_ids = {
            "<|im_start|>": _token_id(tok, "<|im_start|>"),
            "<|im_end|>": _token_id(tok, "<|im_end|>"),
            "user": _token_id(tok, "user"),
            "assistant": _token_id(tok, "assistant"),
            "<|vision_start|>": _token_id(tok, "<|vision_start|>"),
            "<|vision_end|>": _token_id(tok, "<|vision_end|>"),
            "<|video_pad|>": _token_id(tok, "<|video_pad|>"),
            "\n": _token_id(tok, "\n"),
            "Time": _token_id(tok, "Time"),
        }
        # Keep defaults for tokens that could not be resolved as single ids.
        TOKEN_IDS = {k: maybe_ids.get(k) or TOKEN_IDS_DEFAULT[k] for k in TOKEN_IDS_DEFAULT}
    elif isinstance(tokenizer_or_map, dict):
        TOKEN_IDS = {**TOKEN_IDS_DEFAULT, **tokenizer_or_map}
    else:
        raise ValueError("set_token_id_map expects a tokenizer or a dict of token ids")


def get_qwen_range(input_ids, label: str, index, contain_lf: bool = True, strict: bool = True):
    """
    Parameters
    -------
    input_ids : List[int]       ID sequence of length [seqlen]
    label     : str             Values in {'user','assistant','vision','vision_block','user_text','previous text'}
    index     : int             The index-th segment (starting from 0)
    strict    : bool            When False, return None instead of raising if the segment is missing

    Returns
    -------
    start_idx, end_idx : Tuple[int, int]
        (start position, end position) of the index-th segment corresponding to label
    """
    # Find all start_indexes and end_indexes according to a pattern, with ... being free content in between
    # <|im_start|>user\n ...<|im_end|>
    # <|im_start|>assistant\n ...<|im_end|>
    # <|vision_start|><|video_pad|> ...<|vision_end|>  (here .. is <|video_pad|>)
    # <|vision_end|> ...<|im_end|>

    assert label in ['user', 'assistant', 'vision', 'vision_block', 'user_text', 'previous text']
    input_ids = input_ids.flatten().tolist()

    # Prepare token-id level patterns for each label type
    if label == 'user':
        start_pat = [
            TOKEN_IDS["<|im_start|>"],
            TOKEN_IDS["user"],
        ]
        end_pat = [TOKEN_IDS["<|im_end|>"]]
    elif label == 'user_text':
        start_pat = [TOKEN_IDS["Time"]]
        end_pat = [TOKEN_IDS["<|vision_start|>"]]
    elif label == 'assistant':
        start_pat = [
            TOKEN_IDS["<|im_start|>"],
            TOKEN_IDS["assistant"],
        ]
        end_pat = [TOKEN_IDS["<|im_end|>"]]
    elif label == 'vision' or label == 'vision_block':
        start_pat = [TOKEN_IDS["<|vision_start|>"]]
        end_pat = [TOKEN_IDS["<|vision_end|>"]]
    elif label == 'previous text':
        # Used as a leading context block; match the first <|im_start|> ... <|im_end|> segment.
        start_pat = [TOKEN_IDS["<|im_start|>"]]
        end_pat = [TOKEN_IDS["<|im_end|>"]]

    n = len(input_ids)
    segments = []

    i = 0
    while i <= n - len(start_pat):
        # Check if start pattern matches
        if input_ids[i : i + len(start_pat)] == start_pat:
            start_idx = i
            j = i + len(start_pat)
            # Continue looking right for end pattern
            while j <= n - len(end_pat):
                if input_ids[j : j + len(end_pat)] == end_pat:
                    if j + len(end_pat) < n and input_ids[j + len(end_pat)] == TOKEN_IDS["\n"] and contain_lf: # In non-final rounds, there is a \n ending
                        segments.append((start_idx, j+len(end_pat)))
                    else:
                        segments.append((start_idx, j+len(end_pat)-1))
                    i = j + len(end_pat)  # Skip this segment, continue scanning forward
                    break
                j += 1
            else:
                break
        else:
            i += 1

    # Qwen3 expands a video into multiple per-frame vision segments. For pruning we often want the
    # whole video block, so group consecutive vision segments until the next <|im_end|>.
    if label == "vision_block" and segments:
        blocks = []
        cur_start, cur_end = segments[0]
        for seg_start, seg_end in segments[1:]:
            # If an <|im_end|> occurs between segments, start a new block.
            if TOKEN_IDS["<|im_end|>"] in input_ids[cur_end:seg_start]:
                blocks.append((cur_start, cur_end))
                cur_start, cur_end = seg_start, seg_end
            else:
                cur_end = seg_end
        blocks.append((cur_start, cur_end))
        segments = blocks

    if index >= len(segments):
        if strict:
            raise IndexError(f"label={label} index={index} out of {len(segments)} segments")
        return None

    if label == 'user_text':
        return segments[index][0], segments[index][1] - 1  # Because text uses the following vision start as marker
    return segments[index]
