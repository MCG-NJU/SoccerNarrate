#!/usr/bin/env python3
"""Build DPO chosen/rejected pairs for SoccerNet-ASR event ablation.

- chosen: 人类 GT 解说（来自 SoccerNet-ASR 标注，位于目标事件窗内）
- rejected: 人类 “错误” 解说（来自同数据集的其它关键事件窗的 GT 解说）

输入：event_mentions*.jsonl（包含 judge/window/event 字段）
输出：JSONL（StreamingVLM DPO 数据格式）：
  - match_id, t0, prefix_text_tokens, prefix_video_secs, chunk_video_secs
  - chosen_steps, rejected_steps
  - query（可选）与 meta（含 rejected_other_event 信息）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

_WS_RE = re.compile(r"\s+")
_WORD_CHAR_RE = re.compile(r"\w", re.UNICODE)
_ELLIPSIS_ONLY_RE = re.compile(r"^[.…]+$")


SOCCER_COMMENTARY_QUERIES: List[str] = [
    "Commentate on this match.",
    "Describe what's happening in this video.",
    "Provide live commentary for this clip.",
    "What's happening in this football match?",
    "Comment on the action in this video.",
    "Give a commentary of this play.",
    "Describe the events in this clip.",
    "What's going on in this match?",
    "Provide commentary for this moment.",
    "Tell me what's happening here.",
    "Commentate on this action.",
    "Describe the play in this video.",
    "What can you see in this clip?",
    "Give me a live commentary.",
    "What's the action in this video?",
    "Comment on this football moment.",
    "Describe the match events.",
    "What's happening on the pitch?",
    "Provide match commentary.",
    "Commentate on this scene.",
]


def _select_query_for_sample(*, sampling_seed: int, unique_key: str) -> str:
    if not SOCCER_COMMENTARY_QUERIES:
        return "Commentate on this match."
    h = hashlib.md5(f"{int(sampling_seed)}::{unique_key}".encode("utf-8"), usedforsecurity=False).digest()
    idx = int.from_bytes(h[:4], "big", signed=False) % len(SOCCER_COMMENTARY_QUERIES)
    return str(SOCCER_COMMENTARY_QUERIES[idx])


def _stable_choice_index(*, sampling_seed: int, unique_key: str, salt: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    h = hashlib.md5(
        f"{int(sampling_seed)}::{salt}::{unique_key}".encode("utf-8"),
        usedforsecurity=False,
    ).digest()
    return int.from_bytes(h[:8], "big", signed=False) % int(modulo)


def _pick_other_event_candidate(
    *,
    c: "CandidateWindow",
    pool: List["CandidateWindow"],
    pool_indices_by_match_half: Dict[Tuple[str, str, str, int], List[int]],
    pool_indices_by_match: Dict[Tuple[str, str, str], List[int]],
    pool_indices_global: List[int],
    sampling_seed: int,
    unique_key: str,
) -> Tuple[Optional["CandidateWindow"], str]:
    def _pick_from(indices: List[int], *, salt: str) -> Optional["CandidateWindow"]:
        if len(indices) <= 1:
            return None
        start = _stable_choice_index(
            sampling_seed=int(sampling_seed),
            unique_key=str(unique_key),
            salt=str(salt),
            modulo=len(indices),
        )
        for offset in range(len(indices)):
            idx = indices[(start + offset) % len(indices)]
            cand = pool[idx]
            if str(getattr(cand, "event_key", "")) and str(getattr(cand, "event_key", "")) == str(c.event_key):
                continue
            return cand
        return None

    key_mh = (str(c.split), str(c.league_season), str(c.match), int(c.half))
    picked = _pick_from(pool_indices_by_match_half.get(key_mh, []), salt="other_event::same_match_half")
    if picked is not None:
        return picked, "same_match_half"

    key_m = (str(c.split), str(c.league_season), str(c.match))
    picked = _pick_from(pool_indices_by_match.get(key_m, []), salt="other_event::same_match")
    if picked is not None:
        return picked, "same_match"

    picked = _pick_from(pool_indices_global, salt="other_event::global")
    if picked is not None:
        return picked, "global"

    return None, "none"


def _canonicalize_flag(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


_LABEL_CANON: Dict[str, str] = {
    "goal": "Goal",
    "penalty": "Penalty",
    "red card": "Red card",
    "yellow card": "Yellow card",
    "yellow->red card": "Yellow->red card",
    "yellow->red": "Yellow->red card",
    "yellow to red card": "Yellow->red card",
    "yellow to red": "Yellow->red card",
    "second yellow": "Yellow->red card",
    "2nd yellow": "Yellow->red card",
    "shots on target": "Shots on target",
    "shot on target": "Shots on target",
    "shots off target": "Shots off target",
    "shot off target": "Shots off target",
    "corner": "Corner",
    "substitution": "Substitution",
    "foul": "Foul",
    "offside": "Offside",
    "direct free-kick": "Direct free-kick",
    "direct free kick": "Direct free-kick",
    "indirect free-kick": "Indirect free-kick",
    "indirect free kick": "Indirect free-kick",
    "ball out of play": "Ball out of play",
    "throw-in": "Throw-in",
    "throw in": "Throw-in",
    "clearance": "Clearance",
    "kick-off": "Kick-off",
    "kick off": "Kick-off",
}

_TIER_BY_LABEL: Dict[str, str] = {
    "Goal": "T1",
    "Penalty": "T1",
    "Red card": "T1",
    "Yellow->red card": "T1",
    "Yellow card": "T1",
    "Shots on target": "T1",
    "Shots off target": "T1",
    "Corner": "T1",
    "Substitution": "T1",
    "Foul": "T2",
    "Offside": "T2",
    "Direct free-kick": "T2",
    "Indirect free-kick": "T2",
    "Ball out of play": "T3",
    "Throw-in": "T3",
    "Clearance": "T3",
    "Kick-off": "T3",
}

_ANCHOR_PRIORITY_GROUPS: List[List[str]] = [
    ["Goal"],
    ["Penalty"],
    ["Red card", "Yellow->red card"],
    ["Shots on target", "Shots off target"],
    ["Yellow card"],
    ["Corner", "Substitution", "Offside", "Direct free-kick", "Indirect free-kick"],
    ["Foul"],
    ["Ball out of play", "Throw-in", "Clearance", "Kick-off"],
]
_ANCHOR_PRIORITY: Dict[str, int] = {
    label: rank for rank, labels in enumerate(_ANCHOR_PRIORITY_GROUPS) for label in labels
}


def _canonicalize_label(label: Any) -> str:
    if not isinstance(label, str):
        return ""
    text = _WS_RE.sub(" ", label.replace("→", "->")).strip()
    if not text:
        return ""
    key = text.lower()
    key = key.replace("free kick", "free-kick")
    key = key.replace("throw in", "throw-in")
    key = key.replace("kick off", "kick-off")
    return _LABEL_CANON.get(key, text)


def _label_tier(label: str) -> str:
    label = _canonicalize_label(label)
    return _TIER_BY_LABEL.get(label, "T2")


def _label_priority(label: str) -> int:
    label = _canonicalize_label(label)
    return int(_ANCHOR_PRIORITY.get(label, 10_000))


def _choose_anchor_label(*, window_event_labels: Iterable[str], fallback_label: str) -> Tuple[str, str]:
    candidates: List[str] = []
    for raw in window_event_labels:
        canon = _canonicalize_label(raw)
        if canon:
            candidates.append(canon)
    if not candidates:
        fallback = _canonicalize_label(fallback_label)
        return (fallback or str(fallback_label or ""), "event_label_fallback")
    anchor = min(candidates, key=lambda x: (_label_priority(x), x))
    return (anchor, "labels_in_window")


def _parse_league_season(league_season: str) -> Tuple[str, str]:
    if "_" not in league_season:
        return league_season, ""
    league, season = league_season.rsplit("_", 1)
    return league, season


def _normalize_video_root(video_root: Path) -> Path:
    if (video_root / "videos").is_dir():
        return video_root / "videos"
    return video_root


def _normalize_label_root(label_root: Path) -> Path:
    if (label_root / "annos" / "raw").is_dir():
        return label_root / "annos" / "raw"
    if (label_root / "raw").is_dir() and label_root.name != "raw":
        return label_root / "raw"
    return label_root


def _find_half_video(match_video_dir: Path, half: int, exts: List[str]) -> Optional[Path]:
    for ext in exts:
        ext = ext.lstrip(".")
        candidate = match_video_dir / f"{half}_720p.{ext}"
        if candidate.exists():
            return candidate
    return None


def _word_list(text: str) -> List[str]:
    text = _WS_RE.sub(" ", (text or "")).strip()
    if not text:
        return []
    return re.findall(r"\S+", text)


def _count_wordlike_tokens(tokens: Iterable[str]) -> int:
    return sum(1 for t in tokens if _WORD_CHAR_RE.search(t or "") is not None)


def _strip_trailing_ellipsis_from_steps(steps: List[List[str]]) -> List[List[str]]:
    out: List[List[str]] = []
    for step in steps:
        if not isinstance(step, list):
            tokens = _word_list(str(step))
        else:
            tokens = [str(t) for t in step if str(t or "").strip()]
        while tokens and _ELLIPSIS_ONLY_RE.fullmatch(str(tokens[-1]).strip()):
            tokens.pop()
        out.append(tokens)
    return out


@dataclass(frozen=True)
class WordTimeline:
    starts: List[float]
    ends: List[float]
    words: List[str]

    def words_before(self, t: float) -> List[str]:
        idx = bisect_right(self.ends, float(t))
        return self.words[:idx]

    def words_in_interval(self, start: float, end: float) -> List[str]:
        start = float(start)
        end = float(end)
        if end <= start:
            return []
        i = bisect_left(self.ends, start)
        out: List[str] = []
        for j in range(i, len(self.words)):
            ws = self.starts[j]
            we = self.ends[j]
            if ws >= end:
                break
            if we <= start:
                continue
            out.append(self.words[j])
        return out

    def words_bucketed_by_second(
        self,
        *,
        start_sec: int,
        num_steps: int,
        chunk_duration: float,
    ) -> List[List[str]]:
        start_sec = int(start_sec)
        num_steps = int(num_steps)
        chunk_duration = float(chunk_duration)
        if num_steps <= 0:
            return []
        if chunk_duration <= 0:
            return [[] for _ in range(num_steps)]

        window_start = float(start_sec)
        window_end = float(start_sec + num_steps - 1) + chunk_duration

        out: List[List[str]] = [[] for _ in range(num_steps)]
        i = bisect_left(self.ends, window_start)
        for j in range(i, len(self.words)):
            ws = float(self.starts[j])
            we = float(self.ends[j])
            if ws >= window_end:
                break
            if we <= window_start:
                continue
            mid = 0.5 * (ws + we)
            bucket_sec = int(math.floor(mid))
            idx = bucket_sec - start_sec
            if idx < 0:
                idx = 0
            elif idx >= num_steps:
                idx = num_steps - 1
            out[idx].append(self.words[j])
        return out

    def words_evenly_bucketed_by_second(
        self,
        *,
        start_sec: int,
        num_steps: int,
        chunk_duration: float,
    ) -> List[List[str]]:
        start_sec = int(start_sec)
        num_steps = int(num_steps)
        chunk_duration = float(chunk_duration)
        if num_steps <= 0:
            return []
        if chunk_duration <= 0:
            return [[] for _ in range(num_steps)]

        window_start = float(start_sec)
        window_end = float(start_sec + num_steps - 1) + chunk_duration
        window_duration = float(window_end - window_start)
        if window_duration <= 0:
            return [[] for _ in range(num_steps)]

        window_words = self.words_in_interval(window_start, window_end)
        n = len(window_words)
        out: List[List[str]] = [[] for _ in range(num_steps)]
        if n <= 0:
            return out

        for idx_word, word in enumerate(window_words):
            mid = (float(idx_word) + 0.5) * (window_duration / float(n))
            idx = int(math.floor(mid))
            if idx < 0:
                idx = 0
            elif idx >= num_steps:
                idx = num_steps - 1
            out[idx].append(word)
        return out


@dataclass
class CandidateWindow:
    event_key: str
    sample_key: str
    split: str
    league_season: str
    match: str
    half: int
    position_sec: float
    event_label: str
    pre_seconds: float
    post_seconds: float
    window_seconds: Optional[float]
    window_start_s: float
    window_end_s: float
    t0_sec: int
    chunk_seconds: int
    asr_json: Path
    video_path: Path
    label_json: Optional[Path]
    events_in_window: List[Dict[str, Any]]
    anchor_label: str
    anchor_tier: str
    anchor_source: str


def _load_asr_segments(
    asr_json: Path,
    *,
    only_llm_keep: bool,
) -> List[Dict[str, Any]]:
    data = json.loads(asr_json.read_text(encoding="utf-8"))
    segments = data.get("segments", []) or []
    out: List[Dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        if only_llm_keep and seg.get("llm_keep") is False:
            continue
        out.append(seg)
    return out


def _build_word_timeline(
    segments: List[Dict[str, Any]],
    *,
    text_field: str,
    fallback_field: str,
) -> WordTimeline:
    starts: List[float] = []
    ends: List[float] = []
    words: List[str] = []
    for seg in segments:
        try:
            seg_start = float(seg.get("start"))
            seg_end = float(seg.get("end"))
        except Exception:
            continue
        if not (seg_end > seg_start):
            continue
        raw = seg.get(text_field)
        if not isinstance(raw, str) or not raw.strip():
            raw = seg.get(fallback_field)
        if not isinstance(raw, str) or not raw.strip():
            continue
        w = _word_list(raw)
        if not w:
            continue
        dur = seg_end - seg_start
        step = dur / float(len(w))
        for idx, token in enumerate(w):
            ws = seg_start + idx * step
            we = seg_start + (idx + 1) * step
            starts.append(float(ws))
            ends.append(float(min(we, seg_end)))
            words.append(token)
    return WordTimeline(starts=starts, ends=ends, words=words)


def _compute_inference_window(
    *,
    position_sec: float,
    pre_seconds: float,
    post_seconds: float,
    window_seconds: Optional[float],
    window_mode: str,
) -> Tuple[float, float, int, int]:
    if window_mode == "full_window":
        window_start_s = float(position_sec) - float(pre_seconds)
        window_end_s = float(position_sec) + float(post_seconds)
        t0_sec = int(math.floor(window_start_s))
        if t0_sec < 0:
            t0_sec = 0
        desired_window_seconds = window_seconds
        if desired_window_seconds is None:
            desired_window_seconds = float(pre_seconds) + float(post_seconds)
        desired_window_seconds = max(1.0, float(desired_window_seconds))
        desired_window_seconds = min(
            desired_window_seconds,
            max(1.0, window_end_s - max(0.0, window_start_s)),
        )
        chunk_seconds = max(1, int(round(desired_window_seconds)))
        return window_start_s, window_end_s, t0_sec, chunk_seconds

    window_start_s = float(position_sec)
    window_end_s = float(position_sec) + float(post_seconds)
    t0_sec = int(math.floor(window_start_s))
    chunk_seconds = max(1, int(round(post_seconds))) if post_seconds > 0 else 1
    return window_start_s, window_end_s, t0_sec, chunk_seconds


def _resolve_label_json(label_root: Path, *, league: str, season: str, match: str) -> Optional[Path]:
    if not label_root.is_dir():
        return None
    label_dir = label_root / league / season / match
    for name in ("Labels-v2.json", "Labels.json"):
        cand = label_dir / name
        if cand.exists():
            return cand
    return None


def _load_label_events_by_half(label_json: Path) -> Dict[int, List[Dict[str, Any]]]:
    data = json.loads(label_json.read_text(encoding="utf-8"))
    grouped: Dict[int, List[Dict[str, Any]]] = {1: [], 2: []}
    for ann in data.get("annotations", []) or []:
        if not isinstance(ann, dict):
            continue
        game_time = str(ann.get("gameTime") or "")
        if "-" not in game_time:
            continue
        left, right = [p.strip() for p in game_time.split("-", 1)]
        try:
            half = int(left)
        except Exception:
            continue
        if half not in (1, 2):
            continue

        t = None
        pos = ann.get("position", None)
        if pos is not None:
            try:
                t = float(pos) / 1000.0
            except Exception:
                t = None
        if t is None:
            if ":" in right:
                mm, ss = right.split(":", 1)
                try:
                    t = float(int(mm) * 60 + int(ss))
                except Exception:
                    t = None
        if t is None:
            continue

        label = ann.get("label", None)
        if not label:
            continue
        ev: Dict[str, Any] = {"t": float(t), "label": str(label)}
        team = ann.get("team", None)
        if team and team != "not applicable":
            ev["team"] = str(team)
        visibility = ann.get("visibility", None)
        if visibility:
            ev["visibility"] = str(visibility)
        grouped[half].append(ev)

    grouped[1].sort(key=lambda x: x.get("t", 0.0))
    grouped[2].sort(key=lambda x: x.get("t", 0.0))
    return grouped


def _events_in_window(events: List[Dict[str, Any]], start_s: float, end_s: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in events or []:
        try:
            t = float(e.get("t"))
        except Exception:
            continue
        if start_s <= t <= end_s:
            out.append(e)
    return out


def _gt_word_rate(
    timeline: WordTimeline,
    *,
    start_sec: int,
    duration_sec: int,
) -> Tuple[int, float]:
    duration_sec = int(duration_sec)
    if duration_sec <= 0:
        return 0, 0.0
    words = timeline.words_in_interval(float(start_sec), float(start_sec + duration_sec))
    count = _count_wordlike_tokens(words)
    return count, float(count) / float(duration_sec)


def _unique_sample_key(*, event_key: str, sample_key: str) -> str:
    sample_key = str(sample_key or "")
    if sample_key:
        return sample_key
    return str(event_key or "")


def _iter_event_mentions(jsonl_path: Path) -> Iterator[Dict[str, Any]]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _load_done_keys(out_jsonl: Path) -> set[str]:
    if not out_jsonl.exists():
        return set()
    done: set[str] = set()
    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            meta = obj.get("meta", {})
            if not isinstance(meta, dict):
                continue
            sample_key = meta.get("sample_key")
            event_key = meta.get("event_key")
            if isinstance(sample_key, str) and sample_key:
                done.add(sample_key)
            elif isinstance(event_key, str) and event_key:
                done.add(event_key)
    return done


def _stable_shard_for_key(key: str, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(h, 16) % int(num_shards)


def _candidate_from_row(
    row: Dict[str, Any],
    *,
    args: argparse.Namespace,
    video_root: Path,
    label_root: Path,
    video_exts: List[str],
    label_cache: Dict[Path, Dict[int, List[Dict[str, Any]]]],
    skip_reasons: Optional[Counter[str]] = None,
    note_reasons: Optional[Counter[str]] = None,
) -> Optional[CandidateWindow]:
    def _skip(reason: str) -> None:
        if skip_reasons is not None:
            skip_reasons[reason] += 1

    def _note(reason: str) -> None:
        if note_reasons is not None:
            note_reasons[reason] += 1

    event_key = str(row.get("event_key") or "")
    if not event_key:
        _skip("missing_event_key")
        return None

    judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
    if _canonicalize_flag(judge.get("happened")) != "happened":
        _skip("judge_happened!=happened")
        return None
    if "mentioned" in judge and _canonicalize_flag(judge.get("mentioned")) != "mentioned":
        _skip("judge_mentioned!=mentioned")
        return None

    event = row.get("event") if isinstance(row.get("event"), dict) else {}
    split = str(event.get("split") or "")
    excluded_splits = getattr(args, "exclude_splits_set", set())
    if excluded_splits:
        split_key = split.strip().lower()
        if split_key in excluded_splits:
            _skip(f"split_excluded:{split_key}")
            return None
    league_season = str(event.get("league_season") or "")
    match = str(event.get("match") or "")
    try:
        half = int(event.get("half"))
    except Exception:
        _skip("bad_half")
        return None
    try:
        position_sec = float(event.get("position_sec"))
    except Exception:
        _skip("missing_or_bad_position_sec")
        return None
    event_label = str(event.get("event_label") or row.get("event_label") or "")

    window = row.get("window") if isinstance(row.get("window"), dict) else {}
    window_seconds_val: Optional[float]
    try:
        window_seconds_val = float(window.get("window_seconds"))
    except Exception:
        window_seconds_val = None
    if args.require_window_seconds and args.require_window_seconds > 0:
        if window_seconds_val is None or int(round(window_seconds_val)) != int(args.require_window_seconds):
            _skip("window_seconds_mismatch")
            return None
    try:
        pre_seconds = float(window.get("pre_seconds", 0.0))
    except Exception:
        pre_seconds = 0.0
    try:
        post_seconds = float(window.get("post_seconds", 0.0))
    except Exception:
        post_seconds = 0.0

    if not split or not league_season or not match or half not in (1, 2):
        _skip("missing_event_fields")
        return None

    asr_json = args.asr_root / split / league_season / match / f"{half}-224p_processed_clean.json"
    if not asr_json.exists():
        _skip("missing_asr_json")
        return None

    league, season = _parse_league_season(league_season)
    match_video_dir = video_root / league / season / match
    video_path = _find_half_video(match_video_dir, half, video_exts)
    if video_path is None:
        ext = (video_exts[0] if video_exts else "mkv").lstrip(".")
        _note("missing_video_path")
        video_path = match_video_dir / f"{half}_720p.{ext}"

    window_start_s, window_end_s, t0_sec, chunk_seconds = _compute_inference_window(
        position_sec=position_sec,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        window_seconds=window_seconds_val,
        window_mode=str(args.window_mode),
    )

    events_in_window: List[Dict[str, Any]] = []
    label_json = _resolve_label_json(label_root, league=league, season=season, match=match)
    if label_json is None:
        _note("missing_label_json")
    if label_json is not None:
        if label_json not in label_cache:
            try:
                label_cache[label_json] = _load_label_events_by_half(label_json)
            except Exception:
                label_cache[label_json] = {1: [], 2: []}
        evs = label_cache[label_json].get(half, [])
        events_in_window = _events_in_window(evs, float(t0_sec), float(t0_sec + chunk_seconds))

    anchor_label, anchor_source = _choose_anchor_label(
        window_event_labels=[str(e.get("label") or "") for e in events_in_window],
        fallback_label=event_label,
    )
    anchor_tier = _label_tier(anchor_label)

    return CandidateWindow(
        event_key=event_key,
        sample_key=str(row.get("sample_key") or ""),
        split=split,
        league_season=league_season,
        match=match,
        half=half,
        position_sec=position_sec,
        event_label=event_label,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        window_seconds=window_seconds_val,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        t0_sec=t0_sec,
        chunk_seconds=chunk_seconds,
        asr_json=asr_json,
        video_path=video_path,
        label_json=label_json,
        events_in_window=events_in_window,
        anchor_label=anchor_label,
        anchor_tier=anchor_tier,
        anchor_source=anchor_source,
    )


def main() -> int:
    require_window_seconds = 15
    parser = argparse.ArgumentParser(description="Build SoccerNet-ASR DPO pairs for event ablation (rejected=other_event_gt).")
    parser.add_argument(
        "--event_mentions_jsonl",
        type=Path,
        default=Path("/path/to/SoccerNet/asr/event_mentions/event_mentions.jsonl"),
        help="event_mentions*.jsonl file path.",
    )
    parser.add_argument(
        "--out_jsonl",
        type=Path,
        default=Path(f"/path/to/SoccerNet/asr/dpo_annos/dpo_cross_event_{require_window_seconds}s.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--asr_root",
        type=Path,
        default=Path("/path/to/SoccerNet/asr/annos"),
        help="SoccerNet-ASR annos root (contains train/valid/test trees).",
    )
    parser.add_argument(
        "--exclude_splits",
        type=str,
        default="test",
        help="Comma-separated dataset splits to exclude (default: test). Set empty to disable.",
    )
    parser.add_argument(
        "--video_root",
        type=Path,
        default=Path(os.environ.get("SOCCERNET_VIDEOS_ROOT", "/path/to/SoccerNet/videos")),
        help="SoccerNet videos root (contains <league>/<season>/<match>/[1|2]_720p.*).",
    )
    parser.add_argument(
        "--label_root",
        type=Path,
        default=Path(os.environ.get("SOCCERNET_LABEL_DIR", "/path/to/SoccerNet/SoccerNet-label/raw_annos")),
        help="SoccerNet-label root (raw or its parent). Used only to fill meta.events_in_window/anchor_label.",
    )
    parser.add_argument("--video_exts", type=str, default="mkv,mp4", help="Video extensions to try, comma-separated.")

    parser.add_argument(
        "--model_path",
        type=str,
        default="mit-han-lab/StreamingVLM",
        help="HF model path (used only to load tokenizer for prefix_text_tokens).",
    )
    parser.add_argument("--window_size", type=int, default=16, help="Prefix video seconds length.")
    parser.add_argument("--chunk_duration", type=int, default=1, help="Per-step chunk duration in seconds.")
    parser.add_argument(
        "--chosen_steps_mode",
        type=str,
        choices=["even", "timed_midpoint"],
        default="even",
        help="How to bucket GT words into per-second steps.",
    )
    parser.add_argument("--text_sliding_window", type=int, default=512, help="Keep last N prefix tokens (0 => store empty).")
    parser.add_argument("--max_prefix_words", type=int, default=6000, help="Cap prefix words before tokenization.")
    parser.add_argument(
        "--window_mode",
        type=str,
        choices=["full_window", "post_only"],
        default="full_window",
        help="How to define the supervised (chosen/rejected) video window.",
    )
    parser.add_argument(
        "--require_window_seconds",
        type=int,
        default=require_window_seconds,
        help="Keep only rows where window.window_seconds == this value (set <=0 to disable).",
    )
    parser.add_argument(
        "--min_gt_words_per_sec",
        type=float,
        default=1.5,
        help="Filter positives by GT word rate (punctuation excluded). Set <=0 to disable.",
    )
    parser.add_argument("--sampling_seed", type=int, default=12345, help="Deterministic seed (query/neg selection).")

    parser.add_argument("--asr_text_field", type=str, default="original_text")
    parser.add_argument("--asr_fallback_field", type=str, default="llm_polished_text")
    parser.add_argument("--include_not_llm_keep", action="store_true", default=False)

    parser.add_argument("--max_samples", type=int, default=0, help="If >0, stop after writing N samples.")
    parser.add_argument("--resume", action="store_true", default=False, help="Skip sample_key already in out_jsonl.")
    parser.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="Optional extra JSONL to load done keys from (effective only when --resume is set).",
    )
    parser.add_argument("--dry_run", action="store_true", default=False, help="Do not load tokenizer; write empty prefix_text_tokens.")
    parser.add_argument("--num_shards", type=int, default=1, help="Split positives into N shards for parallel runs.")
    parser.add_argument("--shard_id", type=int, default=0, help="Which shard index to run (0..num_shards-1).")
    args = parser.parse_args()

    if args.chunk_duration <= 0:
        raise ValueError("--chunk_duration must be > 0")
    if args.window_size <= 0:
        raise ValueError("--window_size must be > 0")
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be > 0")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must be in [0, num_shards)")

    args.exclude_splits_set = {
        s.strip().lower() for s in str(getattr(args, "exclude_splits", "")).split(",") if s.strip()
    }

    only_llm_keep = not args.include_not_llm_keep

    tokenizer = None
    if not args.dry_run:
        try:
            from transformers import AutoProcessor
        except Exception as e:  # pragma: no cover
            raise RuntimeError("transformers is required to build prefix_text_tokens (or run with --dry_run).") from e
        try:
            processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
            tokenizer = processor.tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from --model_path={args.model_path!r}.") from e

    video_root = _normalize_video_root(args.video_root)
    label_root = _normalize_label_root(args.label_root)
    video_exts = [e.strip() for e in str(args.video_exts).split(",") if e.strip()]

    done_keys: set[str] = set()
    if args.resume:
        done_keys |= _load_done_keys(args.out_jsonl)
        if args.resume_from is not None:
            done_keys |= _load_done_keys(args.resume_from)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    label_cache: Dict[Path, Dict[int, List[Dict[str, Any]]]] = {}
    timeline_cache: Dict[Path, WordTimeline] = {}

    rows_seen = 0
    skipped_resume = 0
    skipped_shard = 0
    skip_reasons: Counter[str] = Counter()
    note_reasons: Counter[str] = Counter()

    pool_candidates: List[CandidateWindow] = []
    candidates_to_process: List[CandidateWindow] = []

    for row in _iter_event_mentions(args.event_mentions_jsonl):
        rows_seen += 1
        event_key = str(row.get("event_key") or "")
        sample_key = str(row.get("sample_key") or "")
        unique_key = _unique_sample_key(event_key=event_key, sample_key=sample_key)
        if not event_key:
            continue
        if args.num_shards > 1 and _stable_shard_for_key(unique_key, args.num_shards) != args.shard_id:
            skipped_shard += 1
            continue

        already_done = bool(done_keys and unique_key in done_keys)
        if already_done:
            skipped_resume += 1

        cand = _candidate_from_row(
            row,
            args=args,
            video_root=video_root,
            label_root=label_root,
            video_exts=video_exts,
            label_cache=label_cache,
            skip_reasons=None if already_done else skip_reasons,
            note_reasons=None if already_done else note_reasons,
        )
        if cand is None:
            continue

        if args.min_gt_words_per_sec and float(args.min_gt_words_per_sec) > 0.0:
            if cand.asr_json not in timeline_cache:
                segments = _load_asr_segments(cand.asr_json, only_llm_keep=only_llm_keep)
                timeline_cache[cand.asr_json] = _build_word_timeline(
                    segments, text_field=args.asr_text_field, fallback_field=args.asr_fallback_field
                )
            timeline = timeline_cache[cand.asr_json]
            _, rate = _gt_word_rate(timeline, start_sec=cand.t0_sec, duration_sec=cand.chunk_seconds)
            if rate < float(args.min_gt_words_per_sec):
                if not already_done:
                    skip_reasons["gt_word_rate_too_low"] += 1
                continue

        pool_candidates.append(cand)
        if already_done:
            continue
        candidates_to_process.append(cand)

    if not pool_candidates or not candidates_to_process:
        shard_info = f" shard={args.shard_id}/{args.num_shards}" if args.num_shards > 1 else ""
        print(f"done:{shard_info} written=0 out={args.out_jsonl}")
        if rows_seen:
            print(
                f"[stats] rows_seen={rows_seen} skipped_shard={skipped_shard} skipped_resume={skipped_resume} "
                f"rejected={sum(skip_reasons.values())}"
            )
            for reason, n in skip_reasons.most_common():
                print(f"[stats] skip_reason {reason}\t{n}")
            for reason, n in note_reasons.most_common():
                print(f"[stats] note {reason}\t{n}")
        return 0

    pool_indices_global = list(range(len(pool_candidates)))
    pool_indices_by_match_half: Dict[Tuple[str, str, str, int], List[int]] = {}
    pool_indices_by_match: Dict[Tuple[str, str, str], List[int]] = {}
    for idx, pc in enumerate(pool_candidates):
        key_mh = (str(pc.split), str(pc.league_season), str(pc.match), int(pc.half))
        pool_indices_by_match_half.setdefault(key_mh, []).append(idx)
        key_m = (str(pc.split), str(pc.league_season), str(pc.match))
        pool_indices_by_match.setdefault(key_m, []).append(idx)

    written = 0
    with args.out_jsonl.open("a", encoding="utf-8") as out_f:
        for c in candidates_to_process:
            unique_key = _unique_sample_key(event_key=c.event_key, sample_key=c.sample_key)
            query_for_sample = _select_query_for_sample(sampling_seed=int(args.sampling_seed), unique_key=unique_key)

            if c.asr_json not in timeline_cache:
                segments = _load_asr_segments(c.asr_json, only_llm_keep=only_llm_keep)
                timeline_cache[c.asr_json] = _build_word_timeline(
                    segments, text_field=args.asr_text_field, fallback_field=args.asr_fallback_field
                )
            timeline = timeline_cache[c.asr_json]

            prefix_video_len = int(args.window_size)
            prefix_start = max(0, int(c.t0_sec) - prefix_video_len)
            prefix_video_secs = list(range(prefix_start, int(c.t0_sec)))
            chunk_video_secs = list(range(int(c.t0_sec), int(c.t0_sec) + int(c.chunk_seconds)))

            prefix_words = timeline.words_before(float(c.t0_sec))
            if args.max_prefix_words > 0 and len(prefix_words) > args.max_prefix_words:
                prefix_words = prefix_words[-args.max_prefix_words :]
            prefix_text_raw = " ".join(prefix_words).strip()

            if args.dry_run:
                prefix_token_ids: List[int] = []
            else:
                assert tokenizer is not None
                full_ids = tokenizer.encode(prefix_text_raw, add_special_tokens=False)
                keep = int(args.text_sliding_window) if args.text_sliding_window and args.text_sliding_window > 0 else 0
                prefix_token_ids = full_ids[-keep:] if keep else []

            if str(args.chosen_steps_mode) == "timed_midpoint":
                chosen_steps = timeline.words_bucketed_by_second(
                    start_sec=int(c.t0_sec),
                    num_steps=len(chunk_video_secs),
                    chunk_duration=float(args.chunk_duration),
                )
            else:
                chosen_steps = timeline.words_evenly_bucketed_by_second(
                    start_sec=int(c.t0_sec),
                    num_steps=len(chunk_video_secs),
                    chunk_duration=float(args.chunk_duration),
                )
            gt_word_count = sum(_count_wordlike_tokens(step) for step in chosen_steps)
            gt_words_per_sec = float(gt_word_count) / float(len(chunk_video_secs) or 1)

            other, scope = _pick_other_event_candidate(
                c=c,
                pool=pool_candidates,
                pool_indices_by_match_half=pool_indices_by_match_half,
                pool_indices_by_match=pool_indices_by_match,
                pool_indices_global=pool_indices_global,
                sampling_seed=int(args.sampling_seed),
                unique_key=unique_key,
            )
            rejected_other_event_meta: Dict[str, Any] = {"scope": scope, "picked": bool(other is not None)}
            if other is None:
                rejected_steps = [[] for _ in chunk_video_secs]
            else:
                if other.asr_json not in timeline_cache:
                    segments = _load_asr_segments(other.asr_json, only_llm_keep=only_llm_keep)
                    timeline_cache[other.asr_json] = _build_word_timeline(
                        segments, text_field=args.asr_text_field, fallback_field=args.asr_fallback_field
                    )
                other_timeline = timeline_cache[other.asr_json]
                if str(args.chosen_steps_mode) == "timed_midpoint":
                    rejected_steps = other_timeline.words_bucketed_by_second(
                        start_sec=int(other.t0_sec),
                        num_steps=len(chunk_video_secs),
                        chunk_duration=float(args.chunk_duration),
                    )
                else:
                    rejected_steps = other_timeline.words_evenly_bucketed_by_second(
                        start_sec=int(other.t0_sec),
                        num_steps=len(chunk_video_secs),
                        chunk_duration=float(args.chunk_duration),
                    )
                rejected_other_event_meta.update(
                    {
                        "event_key": str(other.event_key),
                        "sample_key": str(other.sample_key),
                        "event_label": str(other.event_label),
                        "anchor_label": str(other.anchor_label),
                        "anchor_tier": str(other.anchor_tier),
                        "position_sec": float(other.position_sec),
                        "t0": float(other.t0_sec),
                        "source_chunk_seconds": int(other.chunk_seconds),
                        "used_chunk_seconds": int(len(chunk_video_secs)),
                        "match_id": f"{other.league_season}/{other.match}/half{other.half}",
                        "asr_path": str(other.asr_json),
                    }
                )

            chosen_steps = _strip_trailing_ellipsis_from_steps(chosen_steps)
            rejected_steps = _strip_trailing_ellipsis_from_steps(rejected_steps)

            match_id = f"{c.league_season}/{c.match}/half{c.half}"
            sample = {
                "match_id": match_id,
                "t0": float(c.t0_sec),
                "prefix_text_tokens": prefix_token_ids,
                "prefix_video_secs": prefix_video_secs,
                "chunk_video_secs": chunk_video_secs,
                "chosen_steps": chosen_steps,
                "rejected_steps": rejected_steps,
                "query": str(query_for_sample),
                "meta": {
                    "event_key": c.event_key,
                    "sample_key": c.sample_key,
                    "event_label": c.event_label,
                    "anchor_label": c.anchor_label,
                    "anchor_tier": c.anchor_tier,
                    "anchor_source": c.anchor_source,
                    "position_sec": float(c.position_sec),
                    "half": int(c.half),
                    "split": c.split,
                    "league_season": c.league_season,
                    "match": c.match,
                    "chosen_steps_mode": str(args.chosen_steps_mode),
                    "window": {
                        "window_seconds": float(c.window_seconds) if c.window_seconds is not None else None,
                        "pre_seconds": float(c.pre_seconds),
                        "post_seconds": float(c.post_seconds),
                    },
                    "window_mode": str(args.window_mode),
                    "inference_window": {"start_s": float(c.window_start_s), "end_s": float(c.window_end_s)},
                    "asr_path": str(c.asr_json),
                    "video_path": str(c.video_path),
                    "label_path": str(c.label_json) if c.label_json is not None else "",
                    "events_in_window": c.events_in_window,
                    "rejected_source": "other_event_gt",
                    "rejected_other_event": rejected_other_event_meta,
                    "gt_word_count": int(gt_word_count),
                    "gt_words_per_sec": float(gt_words_per_sec),
                    "hyperparams": {
                        "query": str(query_for_sample),
                        "window_size": int(args.window_size),
                        "chunk_duration": int(args.chunk_duration),
                        "text_sliding_window": int(args.text_sliding_window),
                    },
                },
            }

            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
            if args.max_samples and args.max_samples > 0 and written >= int(args.max_samples):
                break

    shard_info = f" shard={args.shard_id}/{args.num_shards}" if args.num_shards > 1 else ""
    print(f"done:{shard_info} written={written} out={args.out_jsonl}")
    if rows_seen:
        print(
            f"[stats] rows_seen={rows_seen} skipped_shard={skipped_shard} skipped_resume={skipped_resume} "
            f"rejected={sum(skip_reasons.values())}"
        )
        for reason, n in skip_reasons.most_common():
            print(f"[stats] skip_reason {reason}\t{n}")
        for reason, n in note_reasons.most_common():
            print(f"[stats] note {reason}\t{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
