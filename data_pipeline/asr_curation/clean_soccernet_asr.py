#!/usr/bin/env python3
"""Clean SoccerNet ASR annotation JSONs.

What it does
- Keeps top-level metadata as-is (all keys), but replaces `segments` with cleaned segments.
- For each segment:
  - Normalizes whitespace in `original_text` (strip + collapse spaces).
    - Drops fields: `numbered_text`, `corrected_text`, `corrected_text_numbered`, `issues`.
- Merges short segments into neighbors (to avoid overly short utterances).
- Removes obvious noise-only segments.

Outputs
- Writes cleaned files to an output root while preserving relative directory layout.

Example
  python clean_soccernet_asr.py \
    --input-root /path/to/SoccerNet/asr/annos/test \
    --output-root /path/to/SoccerNet/asr/annos_cleaned/test

Notes
- This script does not split long segments (only merge short + remove noise).
- If you want different thresholds, see the CLI args.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_ONLY_RE = re.compile(r"^[^A-Za-z0-9]+$")
_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")
_SOFT_END_RE = re.compile(r"[,;:\u2014\-]\s*$")
_STARTS_WITH_WORD_RE = re.compile(r"^[A-Za-z]+")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def count_words(text: str) -> int:
    text = normalize_text(text)
    if not text:
        return 0
    return len([w for w in text.split(" ") if w])


def is_noise_text(text: str, *, allow_single_word: bool = False) -> bool:
    t = normalize_text(text)
    if not t:
        return True
    if _PUNCT_ONLY_RE.match(t):
        return True

    # Very short fillers. Be conservative: don't remove common commentary tokens like "oh".
    words = t.lower().split(" ")
    if len(words) == 1:
        if words[0] in {"mm", "mmm", "um", "uh", "er", "ah"}:
            return True
        if not allow_single_word and len(t) <= 2:
            # e.g. ".." already caught; this catches extremely tiny leftovers
            return True
    return False


def segment_duration(seg: Dict[str, Any]) -> Optional[float]:
    start = seg.get("start")
    end = seg.get("end")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
        return float(end - start)
    return None


def segment_gap(prev: Dict[str, Any], cur: Dict[str, Any]) -> Optional[float]:
    pe = prev.get("end")
    cs = cur.get("start")
    if isinstance(pe, (int, float)) and isinstance(cs, (int, float)):
        return float(cs - pe)
    return None


@dataclass
class MergeConfig:
    merge_gap_s: float
    short_words: int
    short_chars: int
    min_duration_s: float
    low_density_wps: float
    low_density_min_duration_s: float
    merge_continuations: bool
    continuation_max_prev_words: int
    medium_short_words: int
    medium_short_duration_s: float
    max_merged_duration_s: float
    # Post-processing filter
    post_filter_max_words: int  # Delete segments with words <= this
    post_filter_max_duration_s: float  # ... and duration < this
    post_filter_single_word_max_duration_s: float  # Delete single-word segments with duration < this


def is_low_density(seg: Dict[str, Any], cfg: MergeConfig) -> bool:
    """Check if segment is low-density (long duration but few words) - candidate for deletion."""
    text = normalize_text(str(seg.get("original_text") or ""))
    w = count_words(text)
    d = segment_duration(seg)
    if d is not None and d > 0:
        wps = w / d
        if d >= cfg.low_density_min_duration_s and wps <= cfg.low_density_wps:
            return True
    return False


def should_post_filter(seg: Dict[str, Any], cfg: MergeConfig) -> bool:
    """Check if segment should be filtered out after merging (still too short)."""
    text = normalize_text(str(seg.get("original_text") or ""))
    w = count_words(text)
    d = segment_duration(seg)
    
    if d is None or d <= 0:
        return False
    
    # Single word and very short duration
    if w == 1 and d < cfg.post_filter_single_word_max_duration_s:
        return True
    
    # Few words and short duration
    if w <= cfg.post_filter_max_words and d < cfg.post_filter_max_duration_s:
        return True
    
    return False


def should_force_merge(seg: Dict[str, Any], cfg: MergeConfig, prev: Optional[Dict[str, Any]] = None) -> bool:
    text = normalize_text(str(seg.get("original_text") or ""))
    w = count_words(text)
    d = segment_duration(seg)
    if d is not None and d > 0:
        if d <= cfg.min_duration_s:
            return True
    if w <= cfg.short_words:
        return True
    if len(text) <= cfg.short_chars:
        return True
    # Medium-short segments: also tend to merge, but not if prev is already long
    if w <= cfg.medium_short_words and (d is None or d <= cfg.medium_short_duration_s):
        if prev is not None:
            prev_text = normalize_text(str(prev.get("original_text") or ""))
            prev_words = count_words(prev_text)
            # Don't apply medium-short merge if prev is already approaching the limit
            if prev_words > 20:
                return False
        return True
    return False


def is_continuation(prev: Dict[str, Any], cur: Dict[str, Any], cfg: MergeConfig) -> bool:
    if not cfg.merge_continuations:
        return False

    prev_text = normalize_text(str(prev.get("original_text") or ""))
    cur_text = normalize_text(str(cur.get("original_text") or ""))
    if not prev_text or not cur_text:
        return False

    # If previous segment clearly ends a sentence, do not merge.
    if _SENTENCE_END_RE.search(prev_text):
        return False

    # Avoid blindly merging after extremely long segments.
    if count_words(prev_text) > cfg.continuation_max_prev_words:
        return False

    # If current starts with a lowercase word or connective, it's often a continuation.
    first_word_m = _STARTS_WITH_WORD_RE.match(cur_text)
    first_word = first_word_m.group(0).lower() if first_word_m else ""

    connectives = {
        "and",
        "but",
        "because",
        "so",
        "which",
        "that",
        "then",
        "as",
        "if",
        "when",
        "while",
        "though",
        "although",
        "however",
        "also",
    }

    starts_lower = bool(cur_text) and cur_text[0].isalpha() and cur_text[0].islower()
    starts_connective = first_word in connectives
    prev_soft_end = bool(_SOFT_END_RE.search(prev_text))

    return starts_connective or (starts_lower and (prev_soft_end or len(prev_text) >= 8))


def pick_category(a: Dict[str, Any], b: Dict[str, Any]) -> Any:
    # Prefer the category from the segment that carries more words; fall back to longer duration.
    ta = normalize_text(str(a.get("original_text") or ""))
    tb = normalize_text(str(b.get("original_text") or ""))
    wa, wb = count_words(ta), count_words(tb)
    if wa != wb:
        return a.get("category") if wa > wb else b.get("category")
    da, db = segment_duration(a) or 0.0, segment_duration(b) or 0.0
    return a.get("category") if da >= db else b.get("category")


def merge_two(prev: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(prev)

    # Time
    if isinstance(cur.get("end"), (int, float)):
        merged["end"] = cur["end"]

    # Text
    t1 = normalize_text(str(prev.get("original_text") or ""))
    t2 = normalize_text(str(cur.get("original_text") or ""))
    merged_text = (t1 + " " + t2).strip() if t1 and t2 else (t1 or t2)
    merged["original_text"] = normalize_text(merged_text)

    # Category
    merged["category"] = pick_category(prev, cur)

    # Drop fields we don't want (in case prev had them)
    for k in ("numbered_text", "corrected_text", "corrected_text_numbered", "issues"):
        merged.pop(k, None)

    return merged


def clean_segments(segments: List[Dict[str, Any]], cfg: MergeConfig) -> List[Dict[str, Any]]:
    # Sort by start time if present.
    segments_sorted = sorted(
        (dict(s) for s in segments),
        key=lambda s: (s.get("start") if isinstance(s.get("start"), (int, float)) else float("inf"), s.get("segment_id", 0)),
    )

    cleaned: List[Dict[str, Any]] = []

    for seg in segments_sorted:
        # Normalize text and drop unwanted fields
        seg = dict(seg)
        seg["original_text"] = normalize_text(str(seg.get("original_text") or ""))
        for k in ("numbered_text", "corrected_text", "corrected_text_numbered", "issues"):
            seg.pop(k, None)

        if is_noise_text(seg.get("original_text") or ""):
            continue

        # Delete low-density segments before merging
        if is_low_density(seg, cfg):
            continue

        if not cleaned:
            cleaned.append(seg)
            continue

        prev = cleaned[-1]
        gap = segment_gap(prev, seg)
        can_merge = gap is not None and gap <= cfg.merge_gap_s and gap >= 0

        # Check if merge would exceed max length
        would_merge = can_merge and (
            should_force_merge(prev, cfg, None)
            or should_force_merge(seg, cfg, prev)
            or is_continuation(prev, seg, cfg)
        )
        
        if would_merge:
            # Prevent over-long merges by checking duration
            prev_start = prev.get("start")
            seg_end = seg.get("end")
            if isinstance(prev_start, (int, float)) and isinstance(seg_end, (int, float)):
                merged_duration = seg_end - prev_start
                if merged_duration <= cfg.max_merged_duration_s:
                    cleaned[-1] = merge_two(prev, seg)
                else:
                    cleaned.append(seg)
            else:
                # If we can't calculate duration, merge anyway (fallback)
                cleaned[-1] = merge_two(prev, seg)
        else:
            cleaned.append(seg)

    # Post-processing: filter out segments that are still too short after merging
    filtered: List[Dict[str, Any]] = []
    for seg in cleaned:
        if not should_post_filter(seg, cfg):
            filtered.append(seg)

    # Reassign segment_id sequentially for cleanliness.
    for i, s in enumerate(filtered):
        s["segment_id"] = i
    return filtered


def iter_json_files(root: Path, pattern: str) -> Iterable[Path]:
    return root.rglob(pattern)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_output_path(input_root: Path, output_root: Path, input_file: Path, suffix: str) -> Path:
    rel = input_file.relative_to(input_root)
    if suffix:
        out_name = rel.stem + suffix + rel.suffix
        rel = rel.with_name(out_name)
    return output_root / rel


def process_file(input_path: Path, input_root: Path, output_root: Path, cfg: MergeConfig, suffix: str) -> Tuple[int, int]:
    data = load_json(input_path)
    segs = data.get("segments")
    if not isinstance(segs, list):
        return (0, 0)

    cleaned = clean_segments(segs, cfg)
    data["segments"] = cleaned

    out_path = make_output_path(input_root, output_root, input_path, suffix)
    dump_json(out_path, data)

    return (len(segs), len(cleaned))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean SoccerNet ASR processed JSON annotations")
    p.add_argument("--input-root", type=Path, default="/path/to/SoccerNet/asr/annos/test", help="Root folder to scan for JSONs")
    p.add_argument("--output-root", type=Path, default="/path/to/SoccerNet/asr/annos/test_clean", help="Root folder to write cleaned JSONs")
    p.add_argument("--pattern", type=str, default="*_processed.json", help="Glob pattern for files (default: *_processed.json)")
    p.add_argument("--suffix", type=str, default="_clean", help="Suffix added before .json in output filenames")

    p.add_argument("--merge-gap-s", type=float, default=1, help="Max gap between segments to allow merging")
    p.add_argument("--short-words", type=int, default=2, help="Treat segments with <= this many words as short")
    p.add_argument("--short-chars", type=int, default=12, help="Treat segments with <= this many chars as short")
    p.add_argument("--min-duration-s", type=float, default=0.5, help="Treat segments shorter than this duration as merge candidates")
    p.add_argument("--low-density-wps", type=float, default=0.5, help="Words-per-second threshold for low-density segments")
    p.add_argument("--low-density-min-duration-s", type=float, default=5.0, help="Minimum duration for low-density rule")
    p.add_argument(
        "--merge-continuations",
        action="store_true",
        default=True,
        help="Also merge continuation-like segments (e.g., next starts lowercase/connective and prev doesn't end sentence).",
    )
    p.add_argument(
        "--no-merge-continuations",
        action="store_false",
        dest="merge_continuations",
        help="Disable continuation-like merging.",
    )
    p.add_argument(
        "--continuation-max-prev-words",
        type=int,
        default=30,
        help="Do not apply continuation merge if previous segment has more than this many words.",
    )
    p.add_argument(
        "--medium-short-words",
        type=int,
        default=8,
        help="Also merge segments with <= this many words if duration is short (medium-short threshold).",
    )
    p.add_argument(
        "--medium-short-duration-s",
        type=float,
        default=5.0,
        help="Max duration for medium-short segments to be merged.",
    )
    p.add_argument(
        "--max-merged-duration-s",
        type=float,
        default=64.0,
        help="Do not merge if result would exceed this duration in seconds.",
    )
    p.add_argument(
        "--post-filter-max-words",
        type=int,
        default=2,
        help="Post-merge: delete segments with <= this many words and short duration.",
    )
    p.add_argument(
        "--post-filter-max-duration-s",
        type=float,
        default=2.0,
        help="Post-merge: delete segments with few words and duration < this.",
    )
    p.add_argument(
        "--post-filter-single-word-max-duration-s",
        type=float,
        default=1.5,
        help="Post-merge: delete single-word segments with duration < this.",
    )

    p.add_argument("--limit", type=int, default=0, help="Process at most N files (0 = no limit)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = MergeConfig(
        merge_gap_s=args.merge_gap_s,
        short_words=args.short_words,
        short_chars=args.short_chars,
        min_duration_s=args.min_duration_s,
        low_density_wps=args.low_density_wps,
        low_density_min_duration_s=args.low_density_min_duration_s,
        merge_continuations=args.merge_continuations,
        continuation_max_prev_words=args.continuation_max_prev_words,
        medium_short_words=args.medium_short_words,
        medium_short_duration_s=args.medium_short_duration_s,
        max_merged_duration_s=args.max_merged_duration_s,
        post_filter_max_words=args.post_filter_max_words,
        post_filter_max_duration_s=args.post_filter_max_duration_s,
        post_filter_single_word_max_duration_s=args.post_filter_single_word_max_duration_s,
    )

    input_root: Path = args.input_root
    output_root: Path = args.output_root

    files = list(iter_json_files(input_root, args.pattern))
    files.sort()
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    total_in = 0
    total_out = 0
    processed = 0

    for f in files:
        in_n, out_n = process_file(f, input_root, output_root, cfg, args.suffix)
        if in_n == 0 and out_n == 0:
            continue
        total_in += in_n
        total_out += out_n
        processed += 1

    print(
        json.dumps(
            {
                "files_processed": processed,
                "segments_before": total_in,
                "segments_after": total_out,
                "removed_or_merged": total_in - total_out,
                "input_root": str(input_root),
                "output_root": str(output_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
