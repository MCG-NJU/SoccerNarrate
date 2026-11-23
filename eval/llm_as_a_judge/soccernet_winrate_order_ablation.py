#!/usr/bin/env python3
"""
Order-ablation experiment for LLM-as-a-judge win-rate on SoccerNet commentary.

Goal:
  - Fix one GT event window: [event_time - 5s, event_time + 25s)
  - Build:
      A = event-correct model commentary (same event window)
      B = event-mismatched model commentary
  - Compare four prompt orders:
      1) GT, A, B
      2) GT, B, A
      3) A, B, GT
      4) B, A, GT
  - Test three mismatch types:
      - same_type_wrong_time
      - different_type_wrong_event
      - background_mismatch

Data:
  - model commentary:
      streaming-vlm/streaming_vlm/inference/results/baseline_new
  - GT ASR commentary:
      /path/to/SoccerNet/asr/annos/test
  - SoccerNet event labels:
      /path/to/SoccerNet/soccernet-label/annos/raw

Judge:
  - Use the same prompt as:
      streaming-vlm/streaming_vlm/eval/model_compete/judge.py
  - Send requests in the same HTTP style as:
      eval/llm_as_a_judge/soccernet_vtt_event_mentions.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import local
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _import_shared() -> Any:
    try:
        import soccernet_vtt_event_mentions as shared  # type: ignore

        return shared
    except Exception:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import soccernet_vtt_event_mentions as shared  # type: ignore

        return shared


def _import_human() -> Any:
    try:
        import soccernet_human_asr_event_mentions as human  # type: ignore

        return human
    except Exception:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import soccernet_human_asr_event_mentions as human  # type: ignore

        return human


shared = _import_shared()
human = _import_human()

_REQUESTS_LOCAL = local()

SYS_PROMPT = (
    """
    You are a strict A/B evaluator for sports video commentary.

    Rules:
    1) You MUST output exactly one uppercase letter: "A" or "B".

    2) Evaluate in this order:

    Broadcast tone & pacing: Vary the emotion; deliver live play-by-play or brief, in-the-moment analysis. Not just scene description. Use a real-time broadcast voice that puts the audience courtside.

    Consistency & accuracy: Every detail must stay synchronized with the visuals and never contradict them.
    """
)

ORDER_TEMPLATES: Dict[str, str] = {
    "gt_a_b": (
        "Reference transcript (Provide a reference for the tone and on-screen information):\n{reference}\n\n"
        "Candidate A:\n{A}\n\n"
        "Candidate B:\n{B}\n\n"
    ),
    "gt_b_a": (
        "Reference transcript (Provide a reference for the tone and on-screen information):\n{reference}\n\n"
        "Candidate B:\n{B}\n\n"
        "Candidate A:\n{A}\n\n"
    ),
    "a_b_gt": (
        "Candidate A:\n{A}\n\n"
        "Candidate B:\n{B}\n\n"
        "Reference transcript (Provide a reference for the tone and on-screen information):\n{reference}\n\n"
    ),
    "b_a_gt": (
        "Candidate B:\n{B}\n\n"
        "Candidate A:\n{A}\n\n"
        "Reference transcript (Provide a reference for the tone and on-screen information):\n{reference}\n\n"
    ),
}

MISMATCH_TYPES: Tuple[str, ...] = (
    "same_type_wrong_time",
    "different_type_wrong_event",
    "background_mismatch",
)

DEFAULT_TARGET_LABELS: Tuple[str, ...] = (
    "Goal",
    "Red card",
    "Yellow->red card",
    "Penalty",
    "Substitution",
    "Yellow card",
    "Shots on target",
    "Shots off target",
    "Offside",
    "Corner",
    "Foul",
    "Direct free-kick",
    "Indirect free-kick",
    "Clearance",
)

LABEL_PRIORITY: Dict[str, int] = {
    "Goal": 0,
    "Red card": 1,
    "Yellow->red card": 2,
    "Penalty": 3,
    "Substitution": 4,
    "Yellow card": 5,
    "Shots on target": 6,
    "Shots off target": 7,
    "Offside": 8,
    "Corner": 9,
    "Foul": 10,
    "Direct free-kick": 11,
    "Indirect free-kick": 12,
    "Clearance": 13,
    "Ball out of play": 20,
    "Throw-in": 21,
    "Kick-off": 22,
}

NEAR_GT_WINNER_BY_ORDER: Dict[str, str] = {
    "gt_a_b": "A",
    "gt_b_a": "B",
    "a_b_gt": "B",
    "b_a_gt": "A",
}

_WS_RE = re.compile(r"\s+")
_TRAILING_ELLIPSIS_RE = re.compile(r"(?:\s*(?:\.\.\.|…)+)\s*$")
_WORD_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", flags=re.UNICODE)


@dataclass(frozen=True)
class CandidateWindow:
    candidate_id: str
    source_kind: str
    league_season: str
    match: str
    half: int
    label: str
    event_key: str
    time_sec: float
    window_start_sec: float
    window_end_sec: float
    text: str
    word_count: int
    char_count: int
    event_count_in_window: int
    same_match: bool = True


@dataclass
class HalfBundle:
    infer_path: Path
    asr_path: Path
    label_path: Path
    league_season: str
    match: str
    half: int
    model_segments: List[Tuple[float, float, str]]
    gt_segments: List[Tuple[float, float, str]]
    events: List[Any]
    max_time_sec: float
    event_candidate_cache: Dict[str, Optional[CandidateWindow]] = field(default_factory=dict)
    background_candidates: Optional[List[CandidateWindow]] = None


@dataclass(frozen=True)
class AblationSample:
    sample_id: str
    mismatch_type: str
    league_season: str
    match: str
    target_half: int
    target_label: str
    target_time_sec: float
    gt_window_start_sec: float
    gt_window_end_sec: float
    gt_text: str
    gt_word_count: int
    a: CandidateWindow
    b: CandidateWindow
    b_same_half_as_target: bool
    b_time_gap_sec_same_half: Optional[float]


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return ""
    text = _TRAILING_ELLIPSIS_RE.sub("", text).strip()
    text = _WS_RE.sub(" ", text).strip()
    return text


def _word_tokens(text: str) -> List[str]:
    return [tok.lower() for tok in _WORD_RE.findall(text or "")]


def _word_count(text: str) -> int:
    return len(_word_tokens(text))


def _unique_ratio(text: str) -> float:
    toks = _word_tokens(text)
    if not toks:
        return 0.0
    return float(len(set(toks))) / float(len(toks))


def _is_fluent_enough(
    text: str,
    *,
    min_words: int,
    min_unique_ratio: float,
    min_chars: int,
) -> bool:
    text = _clean_text(text)
    if not text:
        return False
    if len(text) < int(min_chars):
        return False
    wc = _word_count(text)
    if wc < int(min_words):
        return False
    if _unique_ratio(text) < float(min_unique_ratio):
        return False
    return True


def _window_bounds(
    center_sec: float,
    *,
    pre_seconds: float,
    post_seconds: float,
) -> Tuple[float, float]:
    window_seconds = float(pre_seconds) + float(post_seconds)
    start_sec = max(0.0, float(center_sec) - float(pre_seconds))
    end_sec = start_sec + window_seconds
    return start_sec, end_sec


def _candidate_id(
    *,
    source_kind: str,
    infer_path: Path,
    label: str,
    event_key: str,
    start_sec: float,
    end_sec: float,
) -> str:
    raw = (
        f"{source_kind}|{infer_path}|{label}|{event_key}|"
        f"{float(start_sec):.3f}|{float(end_sec):.3f}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _sample_id(
    *,
    mismatch_type: str,
    league_season: str,
    match: str,
    target_half: int,
    target_label: str,
    target_time_sec: float,
    a_id: str,
    b_id: str,
) -> str:
    raw = (
        f"{mismatch_type}|{league_season}|{match}|{target_half}|"
        f"{target_label}|{float(target_time_sec):.3f}|{a_id}|{b_id}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _normalize_api_url(url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return "http://localhost:8000/v1/chat/completions"
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url


def _http_post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    api_key: Optional[str],
    timeout_s: float,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        import requests  # type: ignore

        session = getattr(_REQUESTS_LOCAL, "session", None)
        if session is None:
            session = requests.Session()
            _REQUESTS_LOCAL.session = session
        resp = session.post(url, json=payload, headers=headers, timeout=float(timeout_s))
        resp.raise_for_status()
        return resp.json()
    except ModuleNotFoundError:
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as f:
            return json.loads(f.read().decode("utf-8"))


def _parse_winner(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty judge response.")
    upper = text.upper()
    for line in upper.splitlines():
        line = line.strip()
        if line in {"A", "B"}:
            return line
        m = re.match(r"^(?:ANSWER|FINAL|VOTE)\s*[:：-]?\s*([AB])\b", line)
        if m:
            return m.group(1)
    m = re.search(r"\b([AB])\b", upper)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse judge winner from response: {raw_text!r}")


def _call_ab_judge(
    *,
    api_url: str,
    api_key: Optional[str],
    model: str,
    user_prompt: str,
    max_retries: int,
    timeout_s: float,
) -> Tuple[str, str]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 32,
    }

    last_err: Optional[Exception] = None
    for attempt in range(int(max_retries)):
        try:
            data = _http_post_json(
                api_url,
                payload,
                api_key=api_key,
                timeout_s=float(timeout_s),
            )
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            winner = _parse_winner(str(content))
            return winner, str(content)
        except Exception as exc:
            last_err = exc
            if attempt < int(max_retries) - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Judge call failed: {last_err}")


def _build_prompt(
    *,
    order_name: str,
    gt_text: str,
    a_text: str,
    b_text: str,
    max_chars: int,
) -> str:
    if order_name not in ORDER_TEMPLATES:
        raise KeyError(f"Unknown order_name: {order_name}")
    template = ORDER_TEMPLATES[order_name]
    return template.format(
        reference=_clean_text(gt_text)[: int(max_chars)],
        A=_clean_text(a_text)[: int(max_chars)],
        B=_clean_text(b_text)[: int(max_chars)],
    )


def _load_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str) and sample_id:
                out[sample_id] = row
    return out


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _asr_path_for_infer(infer_path: Path, infer_root: Path, asr_root: Path) -> Optional[Path]:
    try:
        rel = infer_path.relative_to(infer_root)
    except ValueError:
        return None
    if len(rel.parts) < 3:
        return None
    league_season = rel.parts[0]
    match_dir = rel.parts[1]
    half = shared.half_from_filename(infer_path.name)
    if half is None:
        return None
    out = asr_root / league_season / match_dir / f"{half}-224p_processed_clean.json"
    return out


def _event_count_in_window(events: Sequence[Any], *, start_sec: float, end_sec: float) -> int:
    return len(shared.events_in_interval(events, start=float(start_sec), end=float(end_sec)))


def _nearest_event_distance(events: Sequence[Any], *, start_sec: float, end_sec: float) -> float:
    if not events:
        return math.inf
    best = math.inf
    for ev in events:
        d = shared.interval_distance_sec(
            t=float(ev.position_sec),
            start=float(start_sec),
            end=float(end_sec),
        )
        if d < best:
            best = d
    return float(best)


def _get_event_candidate(
    bundle: HalfBundle,
    ev: Any,
    *,
    pre_seconds: float,
    post_seconds: float,
    min_model_words: int,
    min_unique_ratio: float,
    min_model_chars: int,
    max_events_per_window: int,
) -> Optional[CandidateWindow]:
    cached = bundle.event_candidate_cache.get(ev.key)
    if ev.key in bundle.event_candidate_cache:
        return cached

    start_sec, end_sec = _window_bounds(
        float(ev.position_sec),
        pre_seconds=float(pre_seconds),
        post_seconds=float(post_seconds),
    )
    text = _clean_text(
        shared.collect_interval_text(
            bundle.model_segments,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
        )
    )
    if not _is_fluent_enough(
        text,
        min_words=int(min_model_words),
        min_unique_ratio=float(min_unique_ratio),
        min_chars=int(min_model_chars),
    ):
        bundle.event_candidate_cache[ev.key] = None
        return None

    event_count = _event_count_in_window(bundle.events, start_sec=start_sec, end_sec=end_sec)
    if int(max_events_per_window) > 0 and event_count > int(max_events_per_window):
        bundle.event_candidate_cache[ev.key] = None
        return None

    cand = CandidateWindow(
        candidate_id=_candidate_id(
            source_kind="event",
            infer_path=bundle.infer_path,
            label=str(ev.label_raw),
            event_key=str(ev.key),
            start_sec=start_sec,
            end_sec=end_sec,
        ),
        source_kind="event",
        league_season=bundle.league_season,
        match=bundle.match,
        half=int(bundle.half),
        label=str(ev.label_raw),
        event_key=str(ev.key),
        time_sec=float(ev.position_sec),
        window_start_sec=float(start_sec),
        window_end_sec=float(end_sec),
        text=text,
        word_count=_word_count(text),
        char_count=len(text),
        event_count_in_window=int(event_count),
        same_match=True,
    )
    bundle.event_candidate_cache[ev.key] = cand
    return cand


def _build_background_candidates(
    bundle: HalfBundle,
    *,
    window_seconds: float,
    stride_seconds: float,
    min_background_words: int,
    min_unique_ratio: float,
    min_background_chars: int,
    background_min_event_gap_sec: float,
) -> List[CandidateWindow]:
    if bundle.background_candidates is not None:
        return bundle.background_candidates

    out: List[CandidateWindow] = []
    seen_texts: set[str] = set()

    for _idx, start_sec, end_sec in shared.iter_sliding_windows(
        max_time_sec=float(bundle.max_time_sec),
        window_seconds=float(window_seconds),
        stride_seconds=float(stride_seconds),
    ):
        if shared.events_in_interval(bundle.events, start=float(start_sec), end=float(end_sec)):
            continue
        nearest_gap = _nearest_event_distance(bundle.events, start_sec=start_sec, end_sec=end_sec)
        if nearest_gap < float(background_min_event_gap_sec):
            continue

        text = _clean_text(
            shared.collect_interval_text(
                bundle.model_segments,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
            )
        )
        if not _is_fluent_enough(
            text,
            min_words=int(min_background_words),
            min_unique_ratio=float(min_unique_ratio),
            min_chars=int(min_background_chars),
        ):
            continue
        if text in seen_texts:
            continue
        seen_texts.add(text)

        out.append(
            CandidateWindow(
                candidate_id=_candidate_id(
                    source_kind="background",
                    infer_path=bundle.infer_path,
                    label="Background",
                    event_key="",
                    start_sec=start_sec,
                    end_sec=end_sec,
                ),
                source_kind="background",
                league_season=bundle.league_season,
                match=bundle.match,
                half=int(bundle.half),
                label="Background",
                event_key="",
                time_sec=float(start_sec + end_sec) / 2.0,
                window_start_sec=float(start_sec),
                window_end_sec=float(end_sec),
                text=text,
                word_count=_word_count(text),
                char_count=len(text),
                event_count_in_window=0,
                same_match=True,
            )
        )

    bundle.background_candidates = out
    return out


def _length_ratio_ok(
    a_words: int,
    b_words: int,
    *,
    min_ratio: float,
    max_ratio: float,
) -> bool:
    if a_words <= 0 or b_words <= 0:
        return False
    ratio = float(b_words) / float(a_words)
    return float(min_ratio) <= ratio <= float(max_ratio)


def _candidate_score(
    target: CandidateWindow,
    candidate: CandidateWindow,
    *,
    target_half: int,
) -> Tuple[int, int, int, int, int]:
    return (
        0 if int(candidate.half) == int(target_half) else 1,
        int(candidate.event_count_in_window),
        abs(int(candidate.word_count) - int(target.word_count)),
        abs(int(candidate.char_count) - int(target.char_count)),
        abs(int(round(float(candidate.time_sec) * 10.0)) - int(round(float(target.time_sec) * 10.0))),
    )


def _selection_priority(sample: AblationSample) -> Tuple[int, int, int, float, str]:
    return (
        int(LABEL_PRIORITY.get(sample.target_label, 999)),
        int(LABEL_PRIORITY.get(sample.b.label, 999)),
        0 if sample.b_same_half_as_target else 1,
        float(sample.b_time_gap_sec_same_half if sample.b_time_gap_sec_same_half is not None else 1e9),
        sample.sample_id,
    )


def _pick_best_candidate(
    target: CandidateWindow,
    pool: Sequence[CandidateWindow],
    *,
    target_half: int,
    min_length_ratio: float,
    max_length_ratio: float,
) -> Optional[CandidateWindow]:
    best: Optional[CandidateWindow] = None
    best_score: Optional[Tuple[int, int, int, int, int]] = None
    for cand in pool:
        if cand.candidate_id == target.candidate_id:
            continue
        if _clean_text(cand.text) == _clean_text(target.text):
            continue
        if not _length_ratio_ok(
            target.word_count,
            cand.word_count,
            min_ratio=float(min_length_ratio),
            max_ratio=float(max_length_ratio),
        ):
            continue
        score = _candidate_score(target, cand, target_half=target_half)
        if best_score is None or score < best_score:
            best = cand
            best_score = score
    return best


def _same_half_time_gap(ev_a: Any, ev_b: CandidateWindow) -> Optional[float]:
    if int(ev_b.half) != int(ev_a.half):
        return None
    return abs(float(ev_b.time_sec) - float(ev_a.position_sec))


def _keep_target_event(
    target_ev: Any,
    target_gt_text: str,
    *,
    target_event_count: int,
    min_gt_words: int,
    max_events_per_window: int,
    target_labels: Optional[set[str]],
) -> bool:
    if target_labels is not None and str(target_ev.label_raw) not in target_labels:
        return False
    if _word_count(target_gt_text) < int(min_gt_words):
        return False
    if int(max_events_per_window) > 0 and int(target_event_count) > int(max_events_per_window):
        return False
    return True


def _discover_halves(
    *,
    infer_root: Path,
    asr_root: Path,
    labels_root: Path,
    text_field: str,
    fallback_field: str,
    only_llm_keep: bool,
    max_files: Optional[int],
) -> List[HalfBundle]:
    infer_files = shared.iter_vtt_files(infer_root, max_files=max_files)
    out: List[HalfBundle] = []

    for infer_file in infer_files:
        half = shared.half_from_filename(infer_file.name)
        if half is None:
            continue
        label_path = shared.label_file_for_infer(infer_file, infer_root, labels_root)
        if not label_path.exists():
            continue
        asr_path = _asr_path_for_infer(infer_file, infer_root, asr_root)
        if asr_path is None or not asr_path.exists():
            continue

        league_season, match_dir = shared.infer_meta_from_path(infer_file, infer_root)
        model_segments = shared.load_commentary_segments_from_vtt(infer_file)
        gt_segments = human.load_commentary_segments_from_asr_json(
            asr_path,
            text_field=text_field,
            fallback_field=fallback_field,
            only_llm_keep=bool(only_llm_keep),
        )
        events = shared.key_events_only(
            shared.load_label_events(
                label_path,
                half=int(half),
                league_season=league_season,
                match=match_dir,
                visible_only=True,
                include_background=False,
                include_unknown_labels=False,
            )
        )
        if not model_segments or not gt_segments or not events:
            continue

        max_seg_end = max((end for _start, end, _text in model_segments), default=0.0)
        max_gt_end = max((end for _start, end, _text in gt_segments), default=0.0)
        max_event_t = max((float(ev.position_sec) for ev in events), default=0.0)
        max_time = max(float(max_seg_end), float(max_gt_end), float(max_event_t))

        out.append(
            HalfBundle(
                infer_path=infer_file,
                asr_path=asr_path,
                label_path=label_path,
                league_season=league_season,
                match=match_dir,
                half=int(half),
                model_segments=model_segments,
                gt_segments=gt_segments,
                events=list(events),
                max_time_sec=float(max_time),
            )
        )

    return out


def _build_samples(
    *,
    halves: Sequence[HalfBundle],
    pre_seconds: float,
    post_seconds: float,
    background_stride_seconds: float,
    min_model_words: int,
    min_background_words: int,
    min_gt_words: int,
    min_unique_ratio: float,
    min_model_chars: int,
    min_background_chars: int,
    max_events_per_window: int,
    min_event_gap_sec: float,
    background_min_event_gap_sec: float,
    min_length_ratio: float,
    max_length_ratio: float,
    target_labels: Optional[Sequence[str]],
    max_samples_per_type: Optional[int],
    seed: int,
) -> Tuple[List[AblationSample], Dict[str, Any]]:
    target_label_set = set(target_labels) if target_labels else None
    match_groups: Dict[Tuple[str, str], List[HalfBundle]] = defaultdict(list)
    for half in halves:
        match_groups[(half.league_season, half.match)].append(half)
    for group in match_groups.values():
        group.sort(key=lambda x: x.half)

    built_by_type: Dict[str, List[AblationSample]] = {k: [] for k in MISMATCH_TYPES}
    per_type_built_labels: Dict[str, Counter[str]] = {k: Counter() for k in MISMATCH_TYPES}

    for (_league_season, _match), group in match_groups.items():
        event_candidates: List[Tuple[HalfBundle, Any, CandidateWindow]] = []
        for bundle in group:
            for ev in bundle.events:
                cand = _get_event_candidate(
                    bundle,
                    ev,
                    pre_seconds=float(pre_seconds),
                    post_seconds=float(post_seconds),
                    min_model_words=int(min_model_words),
                    min_unique_ratio=float(min_unique_ratio),
                    min_model_chars=int(min_model_chars),
                    max_events_per_window=int(max_events_per_window),
                )
                if cand is not None:
                    event_candidates.append((bundle, ev, cand))

        background_by_half: Dict[int, List[CandidateWindow]] = {}
        all_match_backgrounds: List[CandidateWindow] = []
        for bundle in group:
            backgrounds = _build_background_candidates(
                bundle,
                window_seconds=float(pre_seconds) + float(post_seconds),
                stride_seconds=float(background_stride_seconds),
                min_background_words=int(min_background_words),
                min_unique_ratio=float(min_unique_ratio),
                min_background_chars=int(min_background_chars),
                background_min_event_gap_sec=float(background_min_event_gap_sec),
            )
            background_by_half[int(bundle.half)] = backgrounds
            all_match_backgrounds.extend(backgrounds)

        for bundle, target_ev, a_cand in event_candidates:
            gt_start_sec, gt_end_sec = _window_bounds(
                float(target_ev.position_sec),
                pre_seconds=float(pre_seconds),
                post_seconds=float(post_seconds),
            )
            gt_text = _clean_text(
                shared.collect_interval_text(
                    bundle.gt_segments,
                    start_sec=float(gt_start_sec),
                    end_sec=float(gt_end_sec),
                )
            )
            target_event_count = _event_count_in_window(
                bundle.events,
                start_sec=float(gt_start_sec),
                end_sec=float(gt_end_sec),
            )
            if not _keep_target_event(
                target_ev,
                gt_text,
                target_event_count=int(target_event_count),
                min_gt_words=int(min_gt_words),
                max_events_per_window=int(max_events_per_window),
                target_labels=target_label_set,
            ):
                continue

            same_type_pool: List[CandidateWindow] = []
            diff_type_pool: List[CandidateWindow] = []
            for other_bundle, other_ev, other_cand in event_candidates:
                if other_ev.key == target_ev.key:
                    continue
                if int(other_bundle.half) == int(bundle.half):
                    if abs(float(other_ev.position_sec) - float(target_ev.position_sec)) < float(min_event_gap_sec):
                        continue
                if str(other_ev.label_raw) == str(target_ev.label_raw):
                    same_type_pool.append(other_cand)
                else:
                    diff_type_pool.append(other_cand)

            local_backgrounds = background_by_half.get(int(bundle.half), [])
            background_pool = local_backgrounds or all_match_backgrounds

            chosen_same_type = _pick_best_candidate(
                a_cand,
                same_type_pool,
                target_half=int(bundle.half),
                min_length_ratio=float(min_length_ratio),
                max_length_ratio=float(max_length_ratio),
            )
            chosen_diff_type = _pick_best_candidate(
                a_cand,
                diff_type_pool,
                target_half=int(bundle.half),
                min_length_ratio=float(min_length_ratio),
                max_length_ratio=float(max_length_ratio),
            )
            chosen_background = _pick_best_candidate(
                a_cand,
                background_pool,
                target_half=int(bundle.half),
                min_length_ratio=float(min_length_ratio),
                max_length_ratio=float(max_length_ratio),
            )

            candidates_by_type: Dict[str, Optional[CandidateWindow]] = {
                "same_type_wrong_time": chosen_same_type,
                "different_type_wrong_event": chosen_diff_type,
                "background_mismatch": chosen_background,
            }

            gt_word_count = _word_count(gt_text)
            for mismatch_type, b_cand in candidates_by_type.items():
                if b_cand is None:
                    continue
                same_half_gap = (
                    abs(float(b_cand.time_sec) - float(target_ev.position_sec))
                    if int(b_cand.half) == int(bundle.half)
                    else None
                )
                sample = AblationSample(
                    sample_id=_sample_id(
                        mismatch_type=mismatch_type,
                        league_season=bundle.league_season,
                        match=bundle.match,
                        target_half=int(bundle.half),
                        target_label=str(target_ev.label_raw),
                        target_time_sec=float(target_ev.position_sec),
                        a_id=a_cand.candidate_id,
                        b_id=b_cand.candidate_id,
                    ),
                    mismatch_type=mismatch_type,
                    league_season=bundle.league_season,
                    match=bundle.match,
                    target_half=int(bundle.half),
                    target_label=str(target_ev.label_raw),
                    target_time_sec=float(target_ev.position_sec),
                    gt_window_start_sec=float(gt_start_sec),
                    gt_window_end_sec=float(gt_end_sec),
                    gt_text=gt_text,
                    gt_word_count=int(gt_word_count),
                    a=a_cand,
                    b=b_cand,
                    b_same_half_as_target=bool(int(b_cand.half) == int(bundle.half)),
                    b_time_gap_sec_same_half=None if same_half_gap is None else float(same_half_gap),
                )
                built_by_type[mismatch_type].append(sample)
                per_type_built_labels[mismatch_type][sample.target_label] += 1

    rng = random.Random(int(seed))
    selected: List[AblationSample] = []
    selection_manifest: Dict[str, Any] = {
        "built_counts": {},
        "selected_counts": {},
        "built_target_label_counts": {},
        "selected_target_label_counts": {},
    }
    for mismatch_type in MISMATCH_TYPES:
        items = list({sample.sample_id: sample for sample in built_by_type[mismatch_type]}.values())
        items.sort(key=lambda s: s.sample_id)
        rng.shuffle(items)
        items.sort(key=_selection_priority)
        if max_samples_per_type is not None and int(max_samples_per_type) > 0:
            items = items[: int(max_samples_per_type)]
        selected.extend(items)
        selection_manifest["built_counts"][mismatch_type] = len(built_by_type[mismatch_type])
        selection_manifest["selected_counts"][mismatch_type] = len(items)
        selection_manifest["built_target_label_counts"][mismatch_type] = dict(
            sorted(per_type_built_labels[mismatch_type].items())
        )
        selection_manifest["selected_target_label_counts"][mismatch_type] = dict(
            sorted(Counter(sample.target_label for sample in items).items())
        )

    selected.sort(key=lambda s: (s.mismatch_type, s.league_season, s.match, s.target_half, s.target_time_sec))
    return selected, selection_manifest


def _evaluate_one_sample(
    sample: AblationSample,
    *,
    api_url: str,
    api_key: Optional[str],
    judge_model: str,
    max_prompt_chars: int,
    max_retries: int,
    timeout_s: float,
    sleep_between_requests_s: float,
) -> Dict[str, Any]:
    winner_by_order: Dict[str, str] = {}
    raw_by_order: Dict[str, str] = {}

    for order_name in ORDER_TEMPLATES:
        prompt = _build_prompt(
            order_name=order_name,
            gt_text=sample.gt_text,
            a_text=sample.a.text,
            b_text=sample.b.text,
            max_chars=int(max_prompt_chars),
        )
        winner, raw = _call_ab_judge(
            api_url=api_url,
            api_key=api_key,
            model=judge_model,
            user_prompt=prompt,
            max_retries=int(max_retries),
            timeout_s=float(timeout_s),
        )
        winner_by_order[order_name] = winner
        raw_by_order[order_name] = raw
        if float(sleep_between_requests_s) > 0:
            time.sleep(float(sleep_between_requests_s))

    row: Dict[str, Any] = asdict(sample)
    row.update(
        {
            "winner_by_order": winner_by_order,
            "raw_response_by_order": raw_by_order,
            "judge_model": judge_model,
            "api_url": api_url,
            "completed": True,
            "timestamp": int(time.time()),
        }
    )
    return row


def _safe_rate(num: int, den: int) -> Optional[float]:
    if int(den) <= 0:
        return None
    return float(num) / float(den)


def _pair_flip_rate(records: Sequence[Dict[str, Any]], left: str, right: str) -> Optional[float]:
    valid = 0
    flips = 0
    for row in records:
        winners = row.get("winner_by_order") or {}
        wl = winners.get(left)
        wr = winners.get(right)
        if wl not in {"A", "B"} or wr not in {"A", "B"}:
            continue
        valid += 1
        if wl != wr:
            flips += 1
    return _safe_rate(flips, valid)


def _summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [
        row
        for row in records
        if row.get("completed") is True and isinstance(row.get("winner_by_order"), dict)
    ]

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in completed:
        by_type[str(row.get("mismatch_type") or "unknown")].append(row)

    summary: Dict[str, Any] = {
        "total_completed_samples": len(completed),
        "orders": list(ORDER_TEMPLATES.keys()),
        "mismatch_types": {},
    }

    for mismatch_type in MISMATCH_TYPES:
        rows = by_type.get(mismatch_type, [])
        order_stats: Dict[str, Any] = {}
        adjacent_valid = 0
        adjacent_wins = 0
        correct_valid = 0
        correct_wins = 0
        a_adjacent_valid = 0
        a_adjacent_a_wins = 0
        b_adjacent_valid = 0
        b_adjacent_a_wins = 0
        label_counter = Counter()

        for row in rows:
            label_counter[str(row.get("target_label") or "")] += 1
            winners = row.get("winner_by_order") or {}
            for order_name in ORDER_TEMPLATES:
                winner = winners.get(order_name)
                if winner not in {"A", "B"}:
                    continue
                correct_valid += 1
                if winner == "A":
                    correct_wins += 1
                adjacent_valid += 1
                if winner == NEAR_GT_WINNER_BY_ORDER[order_name]:
                    adjacent_wins += 1
                if order_name in {"gt_a_b", "b_a_gt"}:
                    a_adjacent_valid += 1
                    if winner == "A":
                        a_adjacent_a_wins += 1
                if order_name in {"gt_b_a", "a_b_gt"}:
                    b_adjacent_valid += 1
                    if winner == "A":
                        b_adjacent_a_wins += 1

        for order_name in ORDER_TEMPLATES:
            counts = Counter()
            for row in rows:
                winner = (row.get("winner_by_order") or {}).get(order_name)
                if winner in {"A", "B"}:
                    counts[winner] += 1
            total = counts["A"] + counts["B"]
            order_stats[order_name] = {
                "num_valid": total,
                "A_wins": counts["A"],
                "B_wins": counts["B"],
                "A_win_rate": _safe_rate(counts["A"], total),
                "B_win_rate": _safe_rate(counts["B"], total),
                "near_gt_candidate": NEAR_GT_WINNER_BY_ORDER[order_name],
            }

        any_flip_valid = 0
        any_flip_count = 0
        for row in rows:
            winners = row.get("winner_by_order") or {}
            votes = [winners.get(name) for name in ORDER_TEMPLATES]
            votes = [v for v in votes if v in {"A", "B"}]
            if len(votes) < 2:
                continue
            any_flip_valid += 1
            if len(set(votes)) > 1:
                any_flip_count += 1

        summary["mismatch_types"][mismatch_type] = {
            "num_samples": len(rows),
            "target_label_counts": dict(sorted(label_counter.items())),
            "order_stats": order_stats,
            "correct_candidate_A_win_rate_all_orders": _safe_rate(correct_wins, correct_valid),
            "near_gt_candidate_win_rate_all_orders": _safe_rate(adjacent_wins, adjacent_valid),
            "A_win_rate_when_A_adjacent_to_GT": _safe_rate(a_adjacent_a_wins, a_adjacent_valid),
            "A_win_rate_when_B_adjacent_to_GT": _safe_rate(b_adjacent_a_wins, b_adjacent_valid),
            "adjacency_bias_gap_for_A": (
                None
                if _safe_rate(a_adjacent_a_wins, a_adjacent_valid) is None
                or _safe_rate(b_adjacent_a_wins, b_adjacent_valid) is None
                else _safe_rate(a_adjacent_a_wins, a_adjacent_valid)
                - _safe_rate(b_adjacent_a_wins, b_adjacent_valid)
            ),
            "flip_rate_gt_first_pair": _pair_flip_rate(rows, "gt_a_b", "gt_b_a"),
            "flip_rate_gt_last_pair": _pair_flip_rate(rows, "a_b_gt", "b_a_gt"),
            "flip_rate_any_across_four_orders": _safe_rate(any_flip_count, any_flip_valid),
        }

    overall_rows = completed
    overall_label_counter = Counter(str(row.get("target_label") or "") for row in overall_rows)
    overall_order_stats: Dict[str, Any] = {}
    for order_name in ORDER_TEMPLATES:
        counts = Counter()
        for row in overall_rows:
            winner = (row.get("winner_by_order") or {}).get(order_name)
            if winner in {"A", "B"}:
                counts[winner] += 1
        total = counts["A"] + counts["B"]
        overall_order_stats[order_name] = {
            "num_valid": total,
            "A_wins": counts["A"],
            "B_wins": counts["B"],
            "A_win_rate": _safe_rate(counts["A"], total),
            "B_win_rate": _safe_rate(counts["B"], total),
            "near_gt_candidate": NEAR_GT_WINNER_BY_ORDER[order_name],
        }

    summary["overall"] = {
        "num_samples": len(overall_rows),
        "target_label_counts": dict(sorted(overall_label_counter.items())),
        "order_stats": overall_order_stats,
        "flip_rate_gt_first_pair": _pair_flip_rate(overall_rows, "gt_a_b", "gt_b_a"),
        "flip_rate_gt_last_pair": _pair_flip_rate(overall_rows, "a_b_gt", "b_a_gt"),
        "flip_rate_any_across_four_orders": _safe_rate(
            sum(
                1
                for row in overall_rows
                if len(
                    {
                        v
                        for v in [
                            (row.get("winner_by_order") or {}).get("gt_a_b"),
                            (row.get("winner_by_order") or {}).get("gt_b_a"),
                            (row.get("winner_by_order") or {}).get("a_b_gt"),
                            (row.get("winner_by_order") or {}).get("b_a_gt"),
                        ]
                        if v in {"A", "B"}
                    }
                )
                > 1
            ),
            sum(
                1
                for row in overall_rows
                if len(
                    [
                        v
                        for v in [
                            (row.get("winner_by_order") or {}).get("gt_a_b"),
                            (row.get("winner_by_order") or {}).get("gt_b_a"),
                            (row.get("winner_by_order") or {}).get("a_b_gt"),
                            (row.get("winner_by_order") or {}).get("b_a_gt"),
                        ]
                        if v in {"A", "B"}
                    ]
                )
                >= 2
            ),
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Order-ablation win-rate experiment with GT/A/B prompt permutations.")
    parser.add_argument(
        "--infer-root",
        type=Path,
        default=Path("streaming-vlm/streaming_vlm/inference/results/baseline_new"),
    )
    parser.add_argument(
        "--asr-root",
        type=Path,
        default=Path("/path/to/SoccerNet/asr/annos/test"),
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("/path/to/SoccerNet/soccernet-label/annos/raw"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval/llm_as_a_judge/results_order_ablation"),
    )
    parser.add_argument("--run-name", type=str, default="baseline_new_order_ablation")
    parser.add_argument("--text-field", type=str, default="corrected_name_text")
    parser.add_argument("--fallback-field", type=str, default="llm_polished_text")
    parser.add_argument("--only-llm-keep", dest="only_llm_keep", action="store_true")
    parser.add_argument("--include-not-llm-keep", dest="only_llm_keep", action="store_false")
    parser.set_defaults(only_llm_keep=True)
    parser.add_argument("--window-pre-seconds", type=float, default=5.0)
    parser.add_argument("--window-post-seconds", type=float, default=25.0)
    parser.add_argument("--background-stride-seconds", type=float, default=15.0)
    parser.add_argument("--min-model-words", type=int, default=18)
    parser.add_argument("--min-background-words", type=int, default=18)
    parser.add_argument("--min-gt-words", type=int, default=8)
    parser.add_argument("--min-unique-ratio", type=float, default=0.35)
    parser.add_argument("--min-model-chars", type=int, default=60)
    parser.add_argument("--min-background-chars", type=int, default=60)
    parser.add_argument("--max-events-per-window", type=int, default=2)
    parser.add_argument("--min-event-gap-sec", type=float, default=60.0)
    parser.add_argument("--background-min-event-gap-sec", type=float, default=45.0)
    parser.add_argument("--min-length-ratio", type=float, default=0.6)
    parser.add_argument("--max-length-ratio", type=float, default=1.8)
    parser.add_argument("--target-labels", nargs="*", default=list(DEFAULT_TARGET_LABELS))
    parser.add_argument("--max-samples-per-type", type=int, default=50)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.environ.get("QWEN_API_URL", "http://localhost:8000/v1/chat/completions"),
    )
    parser.add_argument("--api-key", type=str, default=os.environ.get("QWEN_API_KEY", None))
    parser.add_argument(
        "--judge-model",
        type=str,
        default=os.environ.get("QWEN_MODEL", "Qwen3-235B-A22B-Instruct-2507"),
    )
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-prompt-chars", type=int, default=8000)
    parser.add_argument("--sleep-between-requests-s", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    infer_root = Path(args.infer_root).expanduser().resolve()
    asr_root = Path(args.asr_root).expanduser().resolve()
    labels_root = Path(args.labels_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() / str(args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_url = _normalize_api_url(args.api_url)
    results_jsonl = out_dir / "order_ablation_results.jsonl"
    summary_json = out_dir / "order_ablation_summary.json"
    manifest_json = out_dir / "order_ablation_sample_manifest.json"

    halves = _discover_halves(
        infer_root=infer_root,
        asr_root=asr_root,
        labels_root=labels_root,
        text_field=str(args.text_field),
        fallback_field=str(args.fallback_field),
        only_llm_keep=bool(args.only_llm_keep),
        max_files=args.max_files,
    )
    if not halves:
        raise SystemExit("No valid half bundles found.")

    samples, manifest = _build_samples(
        halves=halves,
        pre_seconds=float(args.window_pre_seconds),
        post_seconds=float(args.window_post_seconds),
        background_stride_seconds=float(args.background_stride_seconds),
        min_model_words=int(args.min_model_words),
        min_background_words=int(args.min_background_words),
        min_gt_words=int(args.min_gt_words),
        min_unique_ratio=float(args.min_unique_ratio),
        min_model_chars=int(args.min_model_chars),
        min_background_chars=int(args.min_background_chars),
        max_events_per_window=int(args.max_events_per_window),
        min_event_gap_sec=float(args.min_event_gap_sec),
        background_min_event_gap_sec=float(args.background_min_event_gap_sec),
        min_length_ratio=float(args.min_length_ratio),
        max_length_ratio=float(args.max_length_ratio),
        target_labels=args.target_labels,
        max_samples_per_type=args.max_samples_per_type,
        seed=int(args.seed),
    )

    manifest_payload = {
        "infer_root": str(infer_root),
        "asr_root": str(asr_root),
        "labels_root": str(labels_root),
        "run_name": str(args.run_name),
        "judge_model": str(args.judge_model),
        "api_url": str(api_url),
        "selection": manifest,
        "samples": [asdict(sample) for sample in samples],
    }
    manifest_json.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        dry_summary = {
            "dry_run": True,
            "num_half_bundles": len(halves),
            "num_selected_samples": len(samples),
            "selection": manifest,
            "manifest_json": str(manifest_json),
        }
        summary_json.write_text(json.dumps(dry_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(dry_summary, ensure_ascii=False, indent=2))
        return

    existing = _load_existing_results(results_jsonl)
    todo = [sample for sample in samples if sample.sample_id not in existing]

    iterator: Iterable[AblationSample]
    if tqdm is not None:
        iterator = tqdm(todo, desc="order-ablation", total=len(todo))
    else:
        iterator = todo

    for sample in iterator:
        row = _evaluate_one_sample(
            sample,
            api_url=api_url,
            api_key=args.api_key,
            judge_model=str(args.judge_model),
            max_prompt_chars=int(args.max_prompt_chars),
            max_retries=int(args.max_retries),
            timeout_s=float(args.request_timeout_s),
            sleep_between_requests_s=float(args.sleep_between_requests_s),
        )
        _append_jsonl(results_jsonl, row)

    final_rows = list(_load_existing_results(results_jsonl).values())
    summary = {
        "infer_root": str(infer_root),
        "asr_root": str(asr_root),
        "labels_root": str(labels_root),
        "run_name": str(args.run_name),
        "judge_model": str(args.judge_model),
        "api_url": str(api_url),
        "num_half_bundles": len(halves),
        "num_selected_samples": len(samples),
        "num_completed_rows": len(final_rows),
        "selection": manifest,
        "metrics": _summarize(final_rows),
        "manifest_json": str(manifest_json),
        "results_jsonl": str(results_jsonl),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
