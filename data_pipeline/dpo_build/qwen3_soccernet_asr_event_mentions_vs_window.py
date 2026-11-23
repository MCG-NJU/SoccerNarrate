#!/usr/bin/env python3
"""
Evaluate whether SoccerNet-ASR mentions visible+mentioned SoccerNet events under
multiple time-window sizes, using Qwen3-235B as an LLM judge.

What this script does:
1) Reads SoccerNet labels with `mentioned` / `visibility` fields.
2) Keeps key events with `visibility == visible` and `mentioned == true`.
3) Matches each event to SoccerNet-ASR annotations under `asr/annos`.
4) Extracts ASR text around the event time for multiple window sizes.
5) Sends each event-window pair to Qwen3-235B to judge whether the event is
   mentioned in the commentary.
6) Writes resumable JSONL results and plots:
   - overall mention_rate vs window_seconds
   - per-label mention_rate vs window_seconds

Default window mode is `forward`:
  window = [event_time, event_time + window_seconds]

Examples:
  python qwen3_soccernet_asr_event_mentions_vs_window.py \
    --run-name full_center \
    --num-workers 64

  python qwen3_soccernet_asr_event_mentions_vs_window.py \
    --run-name debug_goal \
    --labels Goal \
    --max-events 20 \
    --dry-run

  python qwen3_soccernet_asr_event_mentions_vs_window.py \
    --run-name full_center \
    --plot-only

Env:
  - QWEN_API_URL
  - QWEN_API_KEY
  - QWEN_MODEL
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from threading import local
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_LABELS_ROOT = Path("/path/to/SoccerNet/SoccerNet-label/annos_with_mentioned_happened")
DEFAULT_ASR_ROOT = Path("/path/to/SoccerNet/asr/annos")
DEFAULT_OUT_DIR = "/path/to/SoccerNet/asr/qwen3_soccernet_asr_event_mentions_vs_window"
DEFAULT_WINDOWS: Tuple[float, ...] = (
    1.0,
    2.0,
    3.0,
    5.0,
    8.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
)

SCHEMA_VERSION = "qwen3_asr_event_mentions_vs_window_v1"

KEY_EVENT_LABELS: List[str] = [
    "Ball out of play",
    "Clearance",
    "Corner",
    "Direct free-kick",
    "Foul",
    "Goal",
    "Indirect free-kick",
    "Kick-off",
    "Offside",
    "Penalty",
    "Red card",
    "Shots off target",
    "Shots on target",
    "Substitution",
    "Throw-in",
    "Yellow card",
    "Yellow->red card",
]
KEY_EVENT_LABEL_SET: set[str] = set(KEY_EVENT_LABELS)

_REQUESTS_LOCAL = local()

SYSTEM_PROMPT = """
You are a careful soccer commentary analysis assistant.

You will be given a short ASR commentary window around a labeled soccer event.

Decide whether the commentary mentions that event.

Rules:
- Use ONLY the given commentary text.
- Be conservative.
- Return "mentioned" only if the text explicitly names the event, or gives strong
  indirect evidence that clearly implies the event.
- Examples of strong indirect evidence:
  - "flag is up" for Offside
  - "points to the spot" for Penalty
  - "booked" / "shown yellow" for Yellow card
  - "sent off" / "shown red" for Red card
  - "behind for a corner" for Corner
- Vague nearby play, buildup, or weak hints are NOT enough.

Return strict JSON only.
""".strip()

MENTION_USER_PROMPT = """
Below is a short window of soccer ASR commentary around a labeled event.

Event label: "{{EVENT_LABEL}}"
Event game clock: "{{GAME_TIME}}"
Half: {{HALF}}
Window seconds: {{WINDOW_SECONDS}}
Window start/end (seconds in half): {{START_SEC}} to {{END_SEC}}

Commentary text:
{{COMMENTARY_TEXT}}

Question:
Does this commentary mention the event "{{EVENT_LABEL}}"?

Return strict JSON only:
{"mentioned":"mentioned|not_mentioned","support_type":"explicit|indirect|none","evidence":"<short quote or rationale>"}
""".strip()

_WS_RE = re.compile(r"\s+")
_TRAILING_ELLIPSIS_RE = re.compile(r"(?:\s*(?:\.\.\.|…)+)\s*$")
_LEAGUE_SEASON_RE = re.compile(r"^(?P<league>.+)_(?P<season>\d{4}-\d{4})$")

_EVENT_LABEL_CANON: Dict[str, str] = {
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


@dataclass(frozen=True)
class EventRef:
    league_season: str
    match: str
    half: int
    label_path: str
    label_raw: str
    position_sec: float
    position_ms: int
    game_time: str
    visibility: str
    mentioned: bool

    @property
    def key(self) -> str:
        raw = (
            f"{self.label_path}|{self.half}|{self.game_time}|{self.label_raw}|"
            f"{self.position_ms}|{self.league_season}|{self.match}"
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()


@dataclass
class JudgeResult:
    mentioned: str
    support_type: str
    evidence: str
    raw_response: str
    postprocessed: bool = False
    postprocess_reason: str = ""


def _norm_token(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def coerce_bool(value: Any) -> Optional[bool]:
    norm = _norm_token(value)
    if norm in {"true", "1", "yes", "y", "mentioned", "happened"}:
        return True
    if norm in {"false", "0", "no", "n", "not_mentioned", "not_happened"}:
        return False
    return None


def canonicalize_mentioned(value: Any) -> Optional[str]:
    norm = _norm_token(value)
    if norm in {"mentioned", "yes", "y", "true", "mention"}:
        return "mentioned"
    if norm in {
        "not_mentioned",
        "not mentioned",
        "unmentioned",
        "notmention",
        "no",
        "n",
        "false",
    }:
        return "not_mentioned"
    return None


def canonicalize_support_type(value: Any) -> Optional[str]:
    norm = _norm_token(value)
    if norm in {"explicit", "direct"}:
        return "explicit"
    if norm in {"indirect", "implicit"}:
        return "indirect"
    if norm in {"none", "no", ""}:
        return "none"
    return None


def canonicalize_event_label(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    text = _WS_RE.sub(" ", value.replace("→", "->")).strip()
    if not text:
        return ""
    key = text.lower()
    key = key.replace("free kick", "free-kick")
    key = key.replace("throw in", "throw-in")
    key = key.replace("kick off", "kick-off")
    if key in _EVENT_LABEL_CANON:
        return _EVENT_LABEL_CANON[key]
    for label in KEY_EVENT_LABELS:
        if key == label.lower():
            return label
    return text


def normalize_visibility(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    return s


def is_visible_visibility(value: Any) -> Optional[bool]:
    norm = normalize_visibility(value)
    if norm is None:
        return None
    return norm == "visible"


def half_from_filename(name: str) -> Optional[int]:
    head = re.split(r"[_-]", name, maxsplit=1)[0].strip()
    if head.isdigit():
        half = int(head)
        if half in (1, 2):
            return half
    return None


def parse_game_time_half(game_time: str) -> Optional[int]:
    if not game_time or "-" not in game_time:
        return None
    left = game_time.split("-", 1)[0].strip()
    try:
        half = int(left)
    except Exception:
        return None
    if half not in (1, 2):
        return None
    return half


def parse_position_ms(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except Exception:
        return None


def parse_window_seconds_list(value: str) -> List[float]:
    items: List[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        items.append(float(part))
    uniq: List[float] = []
    seen: set[float] = set()
    for item in items:
        if item <= 0:
            raise ValueError(f"window size must be > 0, got {item}")
        rounded = round(float(item), 6)
        if rounded in seen:
            continue
        seen.add(rounded)
        uniq.append(float(item))
    if not uniq:
        raise ValueError("empty window list")
    return uniq


def format_window_seconds(value: float) -> str:
    if float(value).is_integer():
        return str(int(round(float(value))))
    return f"{float(value):g}"


def make_sample_key(*, event_key: str, window_seconds: float, window_mode: str) -> str:
    raw = (
        f"{SCHEMA_VERSION}|event_key={event_key}|window_seconds={float(window_seconds):.6f}|"
        f"window_mode={window_mode}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


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


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(0).strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _strip_wrapping_quotes(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    pairs = [('"', '"'), ("'", "'"), ("`", "`")]
    for left, right in pairs:
        if text.startswith(left) and text.endswith(right) and len(text) >= 2:
            return text[1:-1].strip()
    return text


def parse_judge_response(content: str) -> Optional[Tuple[str, str, str]]:
    obj = _extract_first_json_object(content)
    if not obj:
        return None
    mentioned = canonicalize_mentioned(obj.get("mentioned"))
    if not mentioned:
        return None
    support_type = canonicalize_support_type(obj.get("support_type"))
    if support_type is None:
        support_type = "none" if mentioned == "not_mentioned" else "explicit"
    evidence_raw = obj.get("evidence", "")
    evidence = "" if evidence_raw is None else str(evidence_raw).strip()
    return mentioned, support_type, evidence


def _postprocess_judge(judge: JudgeResult) -> JudgeResult:
    mentioned = canonicalize_mentioned(judge.mentioned)
    if mentioned != "mentioned":
        return JudgeResult(
            mentioned="not_mentioned",
            support_type="none",
            evidence="",
            raw_response=judge.raw_response,
            postprocessed=(mentioned is None) or bool(judge.evidence) or judge.support_type not in {"none", ""},
            postprocess_reason="normalized_negative",
        )

    support_type = canonicalize_support_type(judge.support_type) or "explicit"
    evidence = _strip_wrapping_quotes(judge.evidence)
    if not evidence:
        return JudgeResult(
            mentioned="not_mentioned",
            support_type="none",
            evidence="",
            raw_response=judge.raw_response,
            postprocessed=True,
            postprocess_reason="missing_evidence",
        )
    return JudgeResult(
        mentioned="mentioned",
        support_type=support_type,
        evidence=evidence,
        raw_response=judge.raw_response,
    )


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
        resp = session.post(url, json=payload, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()
    except ModuleNotFoundError:
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as f:
            return json.loads(f.read().decode("utf-8"))


def _extract_message_content(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or [{}]
    if not choices:
        return ""
    message = (choices[0] or {}).get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content or "")


def call_qwen3_judge_mention(
    *,
    api_url: str,
    api_key: Optional[str],
    model: str,
    event_label: str,
    game_time: str,
    half: int,
    window_seconds: float,
    start_sec: float,
    end_sec: float,
    window_text: str,
    max_retries: int = 3,
    timeout_s: float = 60.0,
) -> JudgeResult:
    prompt = (
        MENTION_USER_PROMPT.replace("{{EVENT_LABEL}}", event_label)
        .replace("{{GAME_TIME}}", game_time)
        .replace("{{HALF}}", str(int(half)))
        .replace("{{WINDOW_SECONDS}}", format_window_seconds(window_seconds))
        .replace("{{START_SEC}}", f"{float(start_sec):.3f}")
        .replace("{{END_SEC}}", f"{float(end_sec):.3f}")
        .replace("{{COMMENTARY_TEXT}}", window_text)
    )
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 160,
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            data = _http_post_json(api_url, payload, api_key=api_key, timeout_s=timeout_s)
            content = _extract_message_content(data)
            parsed = parse_judge_response(content)
            if not parsed:
                raise ValueError(f"Unparseable judge response: {content!r}")
            mentioned, support_type, evidence = parsed
            judge = JudgeResult(
                mentioned=mentioned,
                support_type=support_type,
                evidence=evidence,
                raw_response=str(content),
            )
            return _postprocess_judge(judge)
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"Failed judge call after {max_retries} attempts: {last_err}") from last_err
    raise RuntimeError("unreachable")


def iter_asr_anno_files(annos_root: Path, *, globs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    seen: set[Path] = set()
    for pattern in globs:
        for path in sorted(annos_root.rglob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            files.append(path)
    return files


def build_asr_index(
    annos_root: Path,
    *,
    globs: Sequence[str],
) -> Tuple[Dict[Tuple[str, str, int], Dict[str, str]], List[Dict[str, str]]]:
    index: Dict[Tuple[str, str, int], Dict[str, str]] = {}
    duplicates: List[Dict[str, str]] = []
    for anno_file in iter_asr_anno_files(annos_root, globs=globs):
        half = half_from_filename(anno_file.name)
        if half is None:
            continue
        try:
            rel = anno_file.relative_to(annos_root)
        except ValueError:
            continue
        if len(rel.parts) < 4:
            continue
        split = rel.parts[0]
        league_season = rel.parts[1]
        match = rel.parts[2]
        key = (league_season, match, half)
        value = {"path": str(anno_file), "split": str(split)}
        if key in index:
            duplicates.append(
                {
                    "league_season": league_season,
                    "match": match,
                    "half": str(half),
                    "kept_path": index[key]["path"],
                    "ignored_path": str(anno_file),
                }
            )
            continue
        index[key] = value
    return index, duplicates


def iter_label_files(labels_root: Path) -> List[Path]:
    return sorted(labels_root.rglob("Labels-v2.json"))


def load_target_events(
    labels_root: Path,
    *,
    visible_only: bool,
    mentioned_only: bool,
    labels_filter: Optional[set[str]],
) -> Tuple[List[EventRef], Counter[str]]:
    events: List[EventRef] = []
    stats: Counter[str] = Counter()

    for label_file in iter_label_files(labels_root):
        try:
            rel = label_file.relative_to(labels_root)
        except ValueError:
            rel = label_file
        if len(rel.parts) < 4:
            stats["bad_label_path"] += 1
            continue
        league = rel.parts[-4]
        season = rel.parts[-3]
        match = rel.parts[-2]
        league_season = f"{league}_{season}"

        try:
            data = json.loads(label_file.read_text(encoding="utf-8"))
        except Exception:
            stats["bad_label_json"] += 1
            continue

        for ann in data.get("annotations", []) or []:
            stats["raw_annotations"] += 1
            if not isinstance(ann, dict):
                stats["bad_annotation"] += 1
                continue

            label_raw = canonicalize_event_label(ann.get("label"))
            if label_raw not in KEY_EVENT_LABEL_SET:
                continue
            if labels_filter is not None and label_raw not in labels_filter:
                continue

            visibility = str(ann.get("visibility") or "").strip()
            if visible_only and is_visible_visibility(visibility) is not True:
                continue

            mentioned = coerce_bool(ann.get("mentioned"))
            if mentioned is None:
                stats["missing_mentioned_field"] += 1
                if mentioned_only:
                    continue
                mentioned = False
            if mentioned_only and not mentioned:
                continue

            game_time = str(ann.get("gameTime") or "").strip()
            half = parse_game_time_half(game_time)
            if half is None:
                stats["bad_game_time"] += 1
                continue

            position_ms = parse_position_ms(ann.get("position"))
            if position_ms is None:
                stats["bad_position"] += 1
                continue

            events.append(
                EventRef(
                    league_season=league_season,
                    match=match,
                    half=half,
                    label_path=str(label_file),
                    label_raw=label_raw,
                    position_sec=float(position_ms) / 1000.0,
                    position_ms=int(position_ms),
                    game_time=game_time,
                    visibility=visibility,
                    mentioned=bool(mentioned),
                )
            )
            stats["selected_events"] += 1

    events.sort(key=lambda ev: (ev.league_season, ev.match, ev.half, ev.position_sec, ev.label_raw))
    return events, stats


@lru_cache(maxsize=2048)
def _load_commentary_segments_cached(
    json_path_str: str,
    text_field: str,
    fallback_fields: Tuple[str, ...],
    only_llm_keep: bool,
) -> Tuple[Tuple[float, float, str], ...]:
    json_path = Path(json_path_str)
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return tuple()

    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segments, list):
        return tuple()

    out: List[Tuple[float, float, str]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        if only_llm_keep and (seg.get("llm_keep") is False):
            continue
        start = seg.get("start")
        end = seg.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            continue

        candidate_fields = (text_field,) + tuple(fallback_fields)
        cleaned = ""
        for field in candidate_fields:
            text = seg.get(field)
            cleaned = _clean_text(text)
            if cleaned:
                break
        if not cleaned:
            continue
        out.append((start_f, end_f, cleaned))

    out.sort(key=lambda x: (x[0], x[1]))
    return tuple(out)


def load_commentary_segments_from_asr_json(
    json_path: Path,
    *,
    text_field: str,
    fallback_fields: Sequence[str],
    only_llm_keep: bool,
) -> List[Tuple[float, float, str]]:
    return list(
        _load_commentary_segments_cached(
            str(json_path),
            str(text_field),
            tuple(str(x) for x in fallback_fields),
            bool(only_llm_keep),
        )
    )


@lru_cache(maxsize=2048)
def _load_commentary_word_spans_cached(
    json_path_str: str,
    text_field: str,
    fallback_fields: Tuple[str, ...],
    only_llm_keep: bool,
) -> Tuple[Tuple[float, float, str, int], ...]:
    segments = _load_commentary_segments_cached(
        json_path_str,
        text_field,
        fallback_fields,
        only_llm_keep,
    )
    out: List[Tuple[float, float, str, int]] = []
    for seg_idx, (seg_start, seg_end, seg_text) in enumerate(segments):
        words = [tok for tok in seg_text.split(" ") if tok]
        if not words:
            continue
        seg_duration = float(seg_end) - float(seg_start)
        if seg_duration <= 0:
            continue
        word_duration = seg_duration / float(len(words))
        for word_idx, word in enumerate(words):
            word_start = float(seg_start) + float(word_idx) * word_duration
            word_end = float(seg_start) + float(word_idx + 1) * word_duration
            out.append((word_start, word_end, word, seg_idx))
    return tuple(out)


def load_commentary_word_spans_from_asr_json(
    json_path: Path,
    *,
    text_field: str,
    fallback_fields: Sequence[str],
    only_llm_keep: bool,
) -> List[Tuple[float, float, str, int]]:
    return list(
        _load_commentary_word_spans_cached(
            str(json_path),
            str(text_field),
            tuple(str(x) for x in fallback_fields),
            bool(only_llm_keep),
        )
    )


def collect_interval_text_and_stats(
    word_spans: Sequence[Tuple[float, float, str, int]],
    segments: Sequence[Tuple[float, float, str]],
    *,
    start_sec: float,
    end_sec: float,
) -> Tuple[str, int, int]:
    words: List[str] = []
    contributing_segments: set[int] = set()
    for word_start, word_end, word, seg_idx in word_spans:
        if word_start < end_sec and word_end > start_sec:
            words.append(word)
            contributing_segments.add(int(seg_idx))

    if words:
        return " ".join(words).strip(), len(contributing_segments), len(words)

    overlap_segments = 0
    for seg_start, seg_end, _seg_text in segments:
        if seg_start < end_sec and seg_end > start_sec:
            overlap_segments += 1
    return "", overlap_segments, 0


def resolve_window_bounds(
    *,
    center_sec: float,
    window_seconds: float,
    window_mode: str,
) -> Tuple[float, float, float, float]:
    mode = (window_mode or "").strip().lower()
    window_seconds = float(window_seconds)
    center_sec = float(center_sec)
    if mode == "center":
        pre_seconds = window_seconds / 2.0
        post_seconds = window_seconds / 2.0
        return center_sec - pre_seconds, center_sec + post_seconds, pre_seconds, post_seconds
    if mode == "forward":
        return center_sec, center_sec + window_seconds, 0.0, window_seconds
    if mode == "backward":
        return center_sec - window_seconds, center_sec, window_seconds, 0.0
    raise ValueError(f"unknown window mode: {window_mode!r}")


def auto_negative_judge(reason: str) -> JudgeResult:
    return JudgeResult(
        mentioned="not_mentioned",
        support_type="none",
        evidence="",
        raw_response=reason,
        postprocessed=False,
        postprocess_reason="",
    )


def _short_error_message(exc: BaseException, *, limit: int = 1200) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = _WS_RE.sub(" ", text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def build_row_for_sample(
    *,
    event: EventRef,
    asr_file: Path,
    split: str,
    window_seconds: float,
    window_mode: str,
    text_field: str,
    fallback_fields: Sequence[str],
    only_llm_keep: bool,
    api_url: str,
    api_key: Optional[str],
    judge_model: str,
    request_timeout_s: float,
    max_retries: int,
    dry_run: bool,
    store_window_text: bool,
) -> Dict[str, Any]:
    sample_key = make_sample_key(event_key=event.key, window_seconds=window_seconds, window_mode=window_mode)
    start_sec, end_sec, pre_seconds, post_seconds = resolve_window_bounds(
        center_sec=event.position_sec,
        window_seconds=window_seconds,
        window_mode=window_mode,
    )

    row: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sample_key": sample_key,
        "event_key": event.key,
        "status": "ok",
        "event": {
            "split": split,
            "league_season": event.league_season,
            "match": event.match,
            "half": int(event.half),
            "label_path": event.label_path,
            "event_label": event.label_raw,
            "position_sec": float(event.position_sec),
            "position_ms": int(event.position_ms),
            "game_time": event.game_time,
            "visibility": event.visibility,
            "mentioned": bool(event.mentioned),
        },
        "asr_file": str(asr_file),
        "window": {
            "window_seconds": float(window_seconds),
            "window_mode": window_mode,
            "pre_seconds": float(pre_seconds),
            "post_seconds": float(post_seconds),
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "segment_count": 0,
            "word_count": 0,
        },
    }

    try:
        if not asr_file.exists():
            row["status"] = "missing_asr"
            row["judge"] = asdict(auto_negative_judge("missing_asr"))
            return row

        segments = load_commentary_segments_from_asr_json(
            asr_file,
            text_field=text_field,
            fallback_fields=fallback_fields,
            only_llm_keep=only_llm_keep,
        )
        word_spans = load_commentary_word_spans_from_asr_json(
            asr_file,
            text_field=text_field,
            fallback_fields=fallback_fields,
            only_llm_keep=only_llm_keep,
        )
        window_text, segment_count, word_count = collect_interval_text_and_stats(
            word_spans,
            segments,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
        )
        row["window"]["segment_count"] = int(segment_count)
        row["window"]["word_count"] = int(word_count)
        if store_window_text:
            row["window"]["text"] = window_text

        if not window_text:
            row["status"] = "empty_text"
            row["judge"] = asdict(auto_negative_judge("empty_text"))
            return row

        if dry_run:
            row["status"] = "dry_run"
            row["judge"] = asdict(auto_negative_judge("dry_run"))
            return row

        judge = call_qwen3_judge_mention(
            api_url=api_url,
            api_key=api_key,
            model=judge_model,
            event_label=event.label_raw,
            game_time=event.game_time,
            half=event.half,
            window_seconds=window_seconds,
            start_sec=start_sec,
            end_sec=end_sec,
            window_text=window_text,
            max_retries=max_retries,
            timeout_s=request_timeout_s,
        )
        row["judge"] = asdict(judge)
    except Exception as exc:
        row["status"] = "error_skipped"
        row["error"] = {
            "type": type(exc).__name__,
            "message": _short_error_message(exc),
        }
    return row


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_existing_jsonl(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return rows
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("schema_version") != SCHEMA_VERSION:
                continue
            sample_key = row.get("sample_key")
            if isinstance(sample_key, str) and sample_key:
                rows[sample_key] = row
    return rows


def flatten_summary_rows(summary: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    overall_rows = list(summary.get("overall_by_window", []) or [])
    per_label_rows: List[Dict[str, Any]] = []
    per_label = summary.get("per_label_by_window", {}) or {}
    for label, rows in per_label.items():
        for row in rows or []:
            flat = dict(row)
            flat["event_label"] = label
            per_label_rows.append(flat)
    return overall_rows, per_label_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(
    *,
    events_with_asr: Sequence[Tuple[EventRef, Dict[str, str]]],
    rows_by_sample_key: Dict[str, Dict[str, Any]],
    window_seconds_list: Sequence[float],
    duplicate_asr_keys: Sequence[Dict[str, str]],
    skipped_missing_asr_events: int,
    selection_stats: Counter[str],
) -> Dict[str, Any]:
    total_events = len(events_with_asr)
    total_by_label = Counter(ev.label_raw for ev, _asr in events_with_asr)

    rows_by_window: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    rows_by_label_window: Dict[str, Dict[float, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    decided_by_window: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    decided_by_label_window: Dict[str, Dict[float, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    status_counter: Counter[str] = Counter()

    for row in rows_by_sample_key.values():
        if row.get("schema_version") != SCHEMA_VERSION:
            continue
        window = row.get("window") if isinstance(row.get("window"), dict) else {}
        event = row.get("event") if isinstance(row.get("event"), dict) else {}
        judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
        event_label = canonicalize_event_label(event.get("event_label"))
        window_seconds = window.get("window_seconds")
        if not isinstance(window_seconds, (int, float)):
            continue
        window_seconds = float(window_seconds)
        status_counter[str(row.get("status") or "ok")] += 1
        rows_by_window[window_seconds].append(row)
        if event_label in KEY_EVENT_LABEL_SET:
            rows_by_label_window[event_label][window_seconds].append(row)
        mentioned = canonicalize_mentioned(judge.get("mentioned"))
        if mentioned not in {"mentioned", "not_mentioned"}:
            continue
        decided_by_window[window_seconds].append(row)
        if event_label in KEY_EVENT_LABEL_SET:
            decided_by_label_window[event_label][window_seconds].append(row)

    overall_by_window: List[Dict[str, Any]] = []
    per_label_by_window: Dict[str, List[Dict[str, Any]]] = {}

    for window_seconds in window_seconds_list:
        all_rows = rows_by_window.get(float(window_seconds), [])
        rows = decided_by_window.get(float(window_seconds), [])
        judged_events = len(rows)
        mentioned_events = 0
        status_counts = Counter(str(row.get("status") or "ok") for row in all_rows)
        for row in rows:
            judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
            if canonicalize_mentioned(judge.get("mentioned")) == "mentioned":
                mentioned_events += 1

        overall_by_window.append(
            {
                "window_seconds": float(window_seconds),
                "expected_events": int(total_events),
                "processed_rows": int(len(all_rows)),
                "judged_events": int(judged_events),
                "coverage_rate": (float(judged_events) / float(total_events)) if total_events else 0.0,
                "mentioned_events": int(mentioned_events),
                "mention_rate": (float(mentioned_events) / float(judged_events)) if judged_events else 0.0,
                "ok_rows": int(status_counts.get("ok", 0)),
                "empty_text_rows": int(status_counts.get("empty_text", 0)),
                "dry_run_rows": int(status_counts.get("dry_run", 0)),
                "missing_asr_rows": int(status_counts.get("missing_asr", 0)),
                "error_skipped_rows": int(status_counts.get("error_skipped", 0)),
            }
        )

    for label in KEY_EVENT_LABELS:
        expected = int(total_by_label.get(label, 0))
        if expected <= 0:
            continue
        rows_for_label: List[Dict[str, Any]] = []
        for window_seconds in window_seconds_list:
            all_rows = rows_by_label_window.get(label, {}).get(float(window_seconds), [])
            rows = decided_by_label_window.get(label, {}).get(float(window_seconds), [])
            judged_events = len(rows)
            mentioned_events = 0
            status_counts = Counter(str(row.get("status") or "ok") for row in all_rows)
            for row in rows:
                judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
                if canonicalize_mentioned(judge.get("mentioned")) == "mentioned":
                    mentioned_events += 1
            rows_for_label.append(
                {
                    "window_seconds": float(window_seconds),
                    "expected_events": int(expected),
                    "processed_rows": int(len(all_rows)),
                    "judged_events": int(judged_events),
                    "coverage_rate": (float(judged_events) / float(expected)) if expected else 0.0,
                    "mentioned_events": int(mentioned_events),
                    "mention_rate": (float(mentioned_events) / float(judged_events)) if judged_events else 0.0,
                    "ok_rows": int(status_counts.get("ok", 0)),
                    "empty_text_rows": int(status_counts.get("empty_text", 0)),
                    "dry_run_rows": int(status_counts.get("dry_run", 0)),
                    "missing_asr_rows": int(status_counts.get("missing_asr", 0)),
                    "error_skipped_rows": int(status_counts.get("error_skipped", 0)),
                }
            )
        per_label_by_window[label] = rows_for_label

    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "window_seconds_list": [float(x) for x in window_seconds_list],
            "total_selected_events_with_asr": int(total_events),
            "skipped_missing_asr_events": int(skipped_missing_asr_events),
            "duplicate_asr_keys": int(len(duplicate_asr_keys)),
            "selection_stats": dict(selection_stats),
            "row_status_counter": dict(status_counter),
            "labels_present": [label for label in KEY_EVENT_LABELS if int(total_by_label.get(label, 0)) > 0],
        },
        "overall_by_window": overall_by_window,
        "per_label_by_window": per_label_by_window,
    }


def plot_overall_mention_rate(
    *,
    summary: Dict[str, Any],
    out_path: Path,
    dpi: int,
) -> None:
    rows = list(summary.get("overall_by_window", []) or [])
    if not rows:
        return
    xs = [float(row["window_seconds"]) for row in rows]
    ys = [float(row["mention_rate"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=4.5, color="#1f77b4")
    ax.set_title("ASR mentioned: mention_rate vs window_seconds (all labels)")
    ax.set_xlabel("window_seconds")
    ax.set_ylabel("mention_rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xticks(xs)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_per_label_mention_rate(
    *,
    summary: Dict[str, Any],
    out_path: Path,
    dpi: int,
) -> None:
    per_label = summary.get("per_label_by_window", {}) or {}
    labels_to_plot = [label for label in KEY_EVENT_LABELS if label in per_label]
    if not labels_to_plot:
        return

    labels_to_plot.sort(
        key=lambda label: float((per_label.get(label) or [{}])[-1].get("mention_rate", 0.0)),
        reverse=True,
    )

    fig, ax = plt.subplots(figsize=(11.8, 9.2))
    for label in labels_to_plot:
        rows = per_label.get(label) or []
        xs = [float(row["window_seconds"]) for row in rows]
        ys = [float(row["mention_rate"]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4.0, label=label)

    ax.set_title("ASR mentioned: mention_rate vs window_seconds (all labels)")
    ax.set_xlabel("window_seconds")
    ax.set_ylabel("mention_rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    if labels_to_plot:
        ax.legend(loc="upper left", ncol=3, fontsize=9, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary: Dict[str, Any]) -> None:
    meta = summary.get("meta", {}) or {}
    overall = list(summary.get("overall_by_window", []) or [])
    print(
        "[summary] selected_with_asr="
        f"{meta.get('total_selected_events_with_asr', 0)} "
        f"skipped_missing_asr={meta.get('skipped_missing_asr_events', 0)} "
        f"duplicate_asr_keys={meta.get('duplicate_asr_keys', 0)}"
    )
    if overall:
        first = overall[0]
        last = overall[-1]
        print(
            "[summary] mention_rate "
            f"{format_window_seconds(float(first['window_seconds']))}s={float(first['mention_rate']):.4f} "
            f"{format_window_seconds(float(last['window_seconds']))}s={float(last['mention_rate']):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-235B judge over SoccerNet-ASR windows for visible+mentioned SoccerNet events."
    )
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--annos-root", type=Path, default=DEFAULT_ASR_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", type=str, default="visible_mentioned_center")

    parser.add_argument("--asr-glob", type=str, nargs="+", default=["*processed_clean.json", "*processed.json"])
    parser.add_argument("--text-field", type=str, default="corrected_name_text")
    parser.add_argument("--fallback-fields", type=str, nargs="+", default=["llm_polished_text", "original_text"])
    parser.add_argument("--only-llm-keep", dest="only_llm_keep", action="store_true")
    parser.add_argument("--include-not-llm-keep", dest="only_llm_keep", action="store_false")
    parser.set_defaults(only_llm_keep=True)

    parser.add_argument(
        "--window-seconds-list",
        type=str,
        default=",".join(format_window_seconds(x) for x in DEFAULT_WINDOWS),
        help="Comma-separated window sizes in seconds.",
    )
    parser.add_argument(
        "--window-mode",
        type=str,
        default="forward",
        choices=["center", "forward", "backward"],
        help="How to place the window relative to the event time.",
    )

    parser.add_argument("--all-visibility", dest="visible_only", action="store_false")
    parser.add_argument("--visible-only", dest="visible_only", action="store_true", default=True)
    parser.add_argument("--all-mentioned", dest="mentioned_only", action="store_false")
    parser.add_argument("--mentioned-only", dest="mentioned_only", action="store_true", default=True)
    parser.add_argument("--labels", nargs="*", default=None, help="Optional label whitelist.")
    parser.add_argument("--max-events", type=int, default=None, help="Debug only: keep first N matched events.")

    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--api-url", type=str, default=os.environ.get("QWEN_API_URL", "http://localhost:8000/v1/chat/completions"))
    parser.add_argument("--api-key", type=str, default=os.environ.get("QWEN_API_KEY", None))
    parser.add_argument("--judge-model", type=str, default=os.environ.get("QWEN_MODEL", "Qwen3-235B-A22B-Instruct-2507"))
    parser.add_argument("--request-timeout-s", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)

    parser.add_argument("--plot-only", action="store_true", help="Skip judge calls; rebuild summary/plots from JSONL.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the judge; write negative dummy rows.")
    parser.add_argument("--omit-window-text", action="store_true", help="Do not store window text inside JSONL rows.")
    parser.add_argument("--plot-dpi", type=int, default=180)
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1")
    if args.request_timeout_s <= 0:
        raise ValueError("--request-timeout-s must be > 0")
    if not args.labels_root.exists():
        raise FileNotFoundError(f"labels root not found: {args.labels_root}")
    if not args.annos_root.exists():
        raise FileNotFoundError(f"annos root not found: {args.annos_root}")

    window_seconds_list = parse_window_seconds_list(args.window_seconds_list)

    labels_filter: Optional[set[str]] = None
    if args.labels:
        labels_filter = set()
        for item in args.labels:
            canon = canonicalize_event_label(item)
            if canon not in KEY_EVENT_LABEL_SET:
                raise ValueError(f"unknown label in --labels: {item!r}")
            labels_filter.add(canon)

    run_out_dir = args.out_dir / str(args.run_name)
    out_jsonl = run_out_dir / "event_mentions_by_window.jsonl"
    out_summary = run_out_dir / "mention_rate_summary.json"
    out_overall_csv = run_out_dir / "mention_rate_overall.csv"
    out_per_label_csv = run_out_dir / "mention_rate_per_label.csv"
    out_overall_png = run_out_dir / "overall_mention_rate_vs_window_seconds.png"
    out_per_label_png = run_out_dir / "per_label_mention_rate_vs_window_seconds.png"

    print("[scan] building ASR index...")
    asr_index, duplicate_asr_keys = build_asr_index(args.annos_root, globs=args.asr_glob)
    print(f"[scan] indexed ASR halves: {len(asr_index)}")
    if duplicate_asr_keys:
        print(f"[scan] duplicate ASR keys ignored: {len(duplicate_asr_keys)}")

    print("[scan] loading target events from labels...")
    events, selection_stats = load_target_events(
        args.labels_root,
        visible_only=bool(args.visible_only),
        mentioned_only=bool(args.mentioned_only),
        labels_filter=labels_filter,
    )
    print(f"[scan] selected events before ASR match: {len(events)}")

    events_with_asr: List[Tuple[EventRef, Dict[str, str]]] = []
    skipped_missing_asr_events = 0
    for event in events:
        key = (event.league_season, event.match, event.half)
        asr_info = asr_index.get(key)
        if asr_info is None:
            skipped_missing_asr_events += 1
            continue
        events_with_asr.append((event, asr_info))

    if args.max_events is not None and args.max_events > 0:
        events_with_asr = events_with_asr[: int(args.max_events)]

    print(f"[scan] matched events with ASR: {len(events_with_asr)}")
    print(f"[scan] skipped events without ASR match: {skipped_missing_asr_events}")

    if not events_with_asr:
        raise RuntimeError("no matched events to process")

    existing = read_existing_jsonl(out_jsonl)
    total_samples = len(events_with_asr) * len(window_seconds_list)
    done_samples = 0
    for event, _asr_info in events_with_asr:
        for window_seconds in window_seconds_list:
            sample_key = make_sample_key(
                event_key=event.key,
                window_seconds=window_seconds,
                window_mode=str(args.window_mode),
            )
            if sample_key in existing:
                done_samples += 1

    if not args.plot_only:
        pbar = None
        if tqdm is not None:
            pbar = tqdm(
                total=total_samples,
                initial=done_samples,
                desc=str(args.run_name),
                unit="sample",
                dynamic_ncols=True,
            )

        executor: Optional[ThreadPoolExecutor] = None
        if args.num_workers > 1:
            executor = ThreadPoolExecutor(max_workers=int(args.num_workers))

        in_flight: Dict[Future[Any], str] = {}
        max_in_flight = max(int(args.num_workers) * 4, 32)

        def consume_row(row: Dict[str, Any]) -> None:
            append_jsonl(out_jsonl, row)
            existing[row["sample_key"]] = row
            if pbar is not None:
                pbar.update(1)

        def drain_futures(*, blocking: bool) -> None:
            nonlocal in_flight
            if not in_flight:
                return
            timeout = None if blocking else 0
            done, _pending = wait(list(in_flight.keys()), timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                return
            for fut in done:
                in_flight.pop(fut, None)
                row = fut.result()
                consume_row(row)

        for event, asr_info in events_with_asr:
            asr_file = Path(asr_info["path"])
            split = str(asr_info["split"])
            for window_seconds in window_seconds_list:
                sample_key = make_sample_key(
                    event_key=event.key,
                    window_seconds=window_seconds,
                    window_mode=str(args.window_mode),
                )
                if sample_key in existing:
                    continue

                kwargs = {
                    "event": event,
                    "asr_file": asr_file,
                    "split": split,
                    "window_seconds": float(window_seconds),
                    "window_mode": str(args.window_mode),
                    "text_field": str(args.text_field),
                    "fallback_fields": list(args.fallback_fields),
                    "only_llm_keep": bool(args.only_llm_keep),
                    "api_url": str(args.api_url),
                    "api_key": args.api_key,
                    "judge_model": str(args.judge_model),
                    "request_timeout_s": float(args.request_timeout_s),
                    "max_retries": int(args.max_retries),
                    "dry_run": bool(args.dry_run),
                    "store_window_text": not bool(args.omit_window_text),
                }

                if executor is None:
                    row = build_row_for_sample(**kwargs)
                    consume_row(row)
                    continue

                fut = executor.submit(build_row_for_sample, **kwargs)
                in_flight[fut] = sample_key
                if len(in_flight) >= max_in_flight:
                    drain_futures(blocking=True)

        while in_flight:
            drain_futures(blocking=True)

        if executor is not None:
            executor.shutdown(wait=True)
        if pbar is not None:
            pbar.close()

    summary = build_summary(
        events_with_asr=events_with_asr,
        rows_by_sample_key=existing,
        window_seconds_list=window_seconds_list,
        duplicate_asr_keys=duplicate_asr_keys,
        skipped_missing_asr_events=skipped_missing_asr_events,
        selection_stats=selection_stats,
    )
    run_out_dir.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overall_rows, per_label_rows = flatten_summary_rows(summary)
    write_csv(out_overall_csv, overall_rows)
    write_csv(out_per_label_csv, per_label_rows)
    plot_overall_mention_rate(summary=summary, out_path=out_overall_png, dpi=int(args.plot_dpi))
    plot_per_label_mention_rate(summary=summary, out_path=out_per_label_png, dpi=int(args.plot_dpi))

    print_summary(summary)
    print(f"[done] jsonl: {out_jsonl}")
    print(f"[done] summary: {out_summary}")
    print(f"[done] plots: {out_overall_png}, {out_per_label_png}")


if __name__ == "__main__":
    main()
