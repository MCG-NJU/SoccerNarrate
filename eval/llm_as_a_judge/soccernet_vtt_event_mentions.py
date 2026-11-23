#!/usr/bin/env python3
"""
Evaluate whether StreamingVLM VTT commentary mentions SoccerNet key events,
using Qwen3-235B as an LLM-as-a-judge.

What it does:
  1) For each VTT file, find its corresponding SoccerNet Labels-v2.json.
  2) Slide windows from start to end of the half:
       - window size: 30s (default), stride: 15s (default)
  3) For each window:
       - If the window contains >=1 SoccerNet key event, ask the original binary prompt
         for each event label in that window: did this event happen NOW in this window?
       - If the window contains no key events, use a negative-window protocol (default: binary entailment for Goal/Foul/Substitution).
         (Optional legacy: 17-way open-set + Background via --neg-window-protocol open.)
  4) Compute metrics:
       - Event Recall@δ: fraction of GT events that are mentioned within ±δ seconds.
       - False Positive Rate: fraction of negative windows that mention an unmatched event (±δ).
       - Absolute Hallucination Rate (AHR) vs Off-timing Mention Rate (OMR) split for negative-window false positives.
  5) Support evaluating multiple inference model dirs under the same infer root.

Qwen judge API:
  - OpenAI ChatCompletions compatible endpoint.
  - Env: QWEN_API_URL, QWEN_API_KEY

Example:
  export QWEN_API_URL="http://localhost:8000/v1/chat/completions"
  export QWEN_API_KEY="xxx"
  python eval/llm_as_a_judge/soccernet_vtt_event_mentions.py \
    --infer-root streaming-vlm/streaming_vlm/inference/infer_test/dpo_from_streamingvlm \
    --infer-models baseline dpo_from_streamingvlm dpo_from_baseline \
    --labels-root /path/to/SoccerNet/SoccerNet-label \
    --num-workers 64 \
    --out-dir eval/llm_as_a_judge/results

Evaluate multiple explicit inference roots in one run:
  python eval/llm_as_a_judge/soccernet_vtt_event_mentions.py \
    --infer-roots /path/to/modelA /path/to/modelB \
    --labels-root /path/to/SoccerNet/SoccerNet-label \
    --num-workers 64 \
    --out-dir eval/llm_as_a_judge/results
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import html
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import local
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


DEFAULT_INFER_ROOTS: List[Path] = [
    # Path("streaming-vlm/streaming_vlm/inference/infer_test/dpo_from_livecc"),
    # Path("streaming-vlm/streaming_vlm/inference/infer_test/dpo_beta0.1_h2m"),
    # Path("streaming-vlm/streaming_vlm/inference/infer_test/dpo_beta0.1_newdata_no_lengthnorm"),
    # Path("streaming-vlm/streaming_vlm/inference/infer_test/baseline_new"),
    # Path("streaming-vlm/streaming_vlm/inference/infer_test/streamingvlm_test"),
    # Path("streaming-vlm/streaming_vlm/inference/infer_test/livecc"),
    Path("streaming-vlm/streaming_vlm/inference/infer_test/dpo_from_baseline_add_kongge"),
]
DEFAULT_LABEL_ROOT = Path("/path/to/SoccerNet/SoccerNet-label/raw_annos")

_REQUESTS_LOCAL = local()


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
BACKGROUND_LABELS: set[str] = {"Background", "Negative"}

# Candidate label list for averaging (macro/micro). Default: all 17 key-event classes.
AVG_LABEL_CANDIDATES: List[str] = list(KEY_EVENT_LABELS)

SCHEMA_VERSION = "sliding_window_v5_neg8_review2m"

# Chosen from `results_human/event_mentioned.json` (high recalled_events, excluding low-salience restart/out labels).
NEGATIVE_WINDOW_BINARY_LABELS: Tuple[str, ...] = (
    "Corner",
    "Direct free-kick",
    "Foul",
    "Goal",
    "Indirect free-kick",
    "Offside",
    "Penalty",
    "Red card",
    "Shots off target",
    "Shots on target",
    "Substitution",
    "Yellow card",
    "Yellow->red card",
    "Ball out of play",
)

# Treat mentions within this window *after* the last same-label GT event as "review/commentary" (not a false positive).
REVIEW_GRACE_SECONDS_DEFAULT = 0.0
REVIEW_GRACE_LABELS_DEFAULT: Tuple[str, ...] = ("Goal", "Foul", "Substitution")
NEG_WINDOW_PROTOCOL_CHOICES: Tuple[str, ...] = ("open", "binary3")
DEFAULT_NEG_WINDOW_PROTOCOL = "binary3"


SYSTEM_PROMPT = """
You are a careful soccer (football) commentary analysis assistant.

You will be given a short window of match commentary text.

Rules:
- Use ONLY the given text (no outside soccer knowledge).
- Be conservative: if the text is unclear/ambiguous, choose the negative/Background answer.

Return JSON only and follow the requested schema exactly.
""".strip()

BINARY_USER_PROMPT = """
Below is a window of live soccer commentary text:
{{COMMENTARY_TEXT}}

Event label: "{{EVENT_LABEL}}"

Question:
Does this commentary window indicate that the event "{{EVENT_LABEL}}" happened in this window?

Return strict JSON only:
{"happened":"happened|not_happened","evidence":"<short quote or rationale>"}
""".strip()

OPEN_USER_PROMPT = """
Below is a window of live soccer commentary text:
{{COMMENTARY_TEXT}}

Question:
Does this window indicate that ONE of the following SoccerNet key events happened?
{{EVENT_LABEL_LIST}}

- If YES: output the single best matching event_label from the list above.
- If NO: output event_label "Background".
- Provide short evidence (quote or rationale). If "Background", evidence can be empty.

Return strict JSON only:
{"event_label":"<one of the 17 labels or Background>","evidence":"<short quote or rationale>"}
""".strip()


@dataclass(frozen=True)
class EventRef:
    league_season: str
    match: str
    half: int
    label_path: str
    label_raw: str
    position_sec: float
    game_time: str
    visibility: str = ""

    @property
    def key(self) -> str:
        raw = f"{self.label_path}|{self.game_time}|{self.label_raw}|{self.position_sec:.3f}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()


@dataclass
class JudgeResult:
    happened: str  # "happened" | "not_happened"
    evidence: str
    raw_response: str
    postprocessed: bool = False
    postprocess_reason: str = ""


@dataclass
class JudgeLabelResult:
    event_label: str  # one of KEY_EVENT_LABELS or "Background"
    evidence: str
    raw_response: str
    predicted_labels: List[str] = field(default_factory=list)
    postprocessed: bool = False
    postprocess_reason: str = ""


def _norm_token(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def canonicalize_happened(value: Any) -> Optional[str]:
    v = _norm_token(value)
    if v in {"happened", "发生"}:
        return "happened"
    if v in {
        "not_happened",
        "not happened",
        "did_not_happen",
        "did not happen",
        "no_happen",
        "no happen",
        "没有发生",
    }:
        return "not_happened"
    if v in {"yes", "y", "true", "是"}:
        return "happened"
    if v in {"no", "n", "false", "否"}:
        return "not_happened"
    return None


def f1_from_precision_recall(precision: float, recall: float) -> float:
    denom = float(precision) + float(recall)
    if denom <= 0.0:
        return 0.0
    return float(2.0 * float(precision) * float(recall) / denom)


_WS_RE_CANON = re.compile(r"\s+")
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
_BACKGROUND_TOKENS: set[str] = {
    "",
    "background",
    "bg",
    "none",
    "no",
    "no event",
    "no_event",
    "negative",
    "other",
    "others",
    "else",
    "nothing",
    "n/a",
    "na",
}


def canonicalize_event_label(value: Any) -> str:
    if value is None:
        return "Background"
    if not isinstance(value, str):
        value = str(value)
    text = _WS_RE_CANON.sub(" ", value.replace("→", "->")).strip()
    if not text:
        return "Background"
    key = text.lower()
    key = key.replace("free kick", "free-kick")
    key = key.replace("throw in", "throw-in")
    key = key.replace("kick off", "kick-off")
    if key in _BACKGROUND_TOKENS:
        return "Background"
    if key in _EVENT_LABEL_CANON:
        return _EVENT_LABEL_CANON[key]

    # Case-insensitive exact match against the canonical label list.
    for label in KEY_EVENT_LABELS:
        if key == label.lower():
            return label

    return "Background"


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


def parse_judge_response(content: str) -> Optional[Tuple[str, str]]:
    obj = _extract_first_json_object(content)
    if not obj:
        return None
    happened = canonicalize_happened(obj.get("happened"))
    if not happened:
        return None
    evidence_raw = obj.get("evidence", "")
    evidence = "" if evidence_raw is None else str(evidence_raw).strip()
    return happened, evidence


def parse_open_judge_response(content: str) -> Optional[Tuple[str, str]]:
    obj = _extract_first_json_object(content)
    if not obj:
        return None
    label = canonicalize_event_label(obj.get("event_label"))
    evidence_raw = obj.get("evidence", "")
    evidence = "" if evidence_raw is None else str(evidence_raw).strip()
    return label, evidence


def _strip_wrapping_quotes(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    pairs = [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("`", "`")]
    for left, right in pairs:
        if text.startswith(left) and text.endswith(right) and len(text) >= 2:
            return text[1:-1].strip()
    return text


def _postprocess_binary_judge(
    *,
    judge: JudgeResult,
    window_text: str,
    event_label: str,
) -> JudgeResult:
    happened = canonicalize_happened(judge.happened)
    if happened != "happened":
        if happened is None:
            return JudgeResult(
                happened="not_happened",
                evidence="",
                raw_response=judge.raw_response,
                postprocessed=True,
                postprocess_reason="unparsed_happened",
            )
        return JudgeResult(
            happened=happened,
            evidence=_strip_wrapping_quotes(judge.evidence),
            raw_response=judge.raw_response,
        )

    evidence = _strip_wrapping_quotes(judge.evidence)
    if not evidence:
        return JudgeResult(
            happened="not_happened",
            evidence="",
            raw_response=judge.raw_response,
            postprocessed=True,
            postprocess_reason="missing_evidence",
        )
    return JudgeResult(happened="happened", evidence=evidence, raw_response=judge.raw_response)


def _postprocess_open_judge(
    *,
    judge: JudgeLabelResult,
    window_text: str,
) -> JudgeLabelResult:
    label = canonicalize_event_label(judge.event_label)
    evidence = _strip_wrapping_quotes(judge.evidence)
    if label == "Background":
        normalized = (judge.event_label or "").strip() != "Background"
        return JudgeLabelResult(
            event_label="Background",
            evidence="",
            raw_response=judge.raw_response,
            postprocessed=normalized or bool(evidence),
            postprocess_reason="normalized_to_background" if (normalized or bool(evidence)) else "",
        )
    if label not in KEY_EVENT_LABEL_SET:
        return JudgeLabelResult(
            event_label="Background",
            evidence="",
            raw_response=judge.raw_response,
            postprocessed=(label != "Background" or bool(evidence)),
            postprocess_reason="invalid_label",
        )
    if not evidence:
        return JudgeLabelResult(
            event_label="Background",
            evidence="",
            raw_response=judge.raw_response,
            postprocessed=True,
            postprocess_reason="missing_evidence",
        )
    return JudgeLabelResult(event_label=label, evidence=evidence, raw_response=judge.raw_response)


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


def _event_label_bullets() -> str:
    return "\n".join([f"- {label}" for label in KEY_EVENT_LABELS])


def call_qwen3_judge_binary(
    *,
    api_url: str,
    api_key: Optional[str],
    model: str,
    event_label: str,
    window_text: str,
    max_retries: int = 3,
    timeout_s: float = 60.0,
) -> JudgeResult:
    prompt = BINARY_USER_PROMPT.replace("{{COMMENTARY_TEXT}}", window_text).replace("{{EVENT_LABEL}}", event_label)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 120,
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            data = _http_post_json(api_url, payload, api_key=api_key, timeout_s=timeout_s)
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            parsed = parse_judge_response(content)
            if not parsed:
                raise ValueError(f"Unparseable judge response: {content!r}")
            happened, evidence = parsed
            judge = JudgeResult(happened=happened, evidence=evidence, raw_response=str(content))
            return _postprocess_binary_judge(judge=judge, window_text=window_text, event_label=event_label)
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Failed to call judge: {last_err}")


def call_qwen3_judge_open(
    *,
    api_url: str,
    api_key: Optional[str],
    model: str,
    window_text: str,
    max_retries: int = 3,
    timeout_s: float = 60.0,
) -> JudgeLabelResult:
    prompt = OPEN_USER_PROMPT.replace("{{COMMENTARY_TEXT}}", window_text).replace("{{EVENT_LABEL_LIST}}", _event_label_bullets())
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 120,
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            data = _http_post_json(api_url, payload, api_key=api_key, timeout_s=timeout_s)
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            parsed = parse_open_judge_response(content)
            if not parsed:
                raise ValueError(f"Unparseable judge response: {content!r}")
            event_label, evidence = parsed
            judge = JudgeLabelResult(event_label=event_label, evidence=evidence, raw_response=str(content))
            return _postprocess_open_judge(judge=judge, window_text=window_text)
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Failed to call judge: {last_err}")


def call_qwen3_judge_negative_window(
    *,
    api_url: str,
    api_key: Optional[str],
    model: str,
    window_text: str,
    protocol: str = DEFAULT_NEG_WINDOW_PROTOCOL,
    max_retries: int = 3,
    timeout_s: float = 60.0,
) -> JudgeLabelResult:
    """
    Negative-window judging protocol.

    - protocol="open": legacy open-set (17-way + Background).
    - protocol="binary3": run binary entailment checks for `NEGATIVE_WINDOW_BINARY_LABELS`; if all not_happened -> Background.
    """

    protocol = (protocol or "").strip().lower()
    if protocol not in NEG_WINDOW_PROTOCOL_CHOICES:
        raise ValueError(f"Unknown neg window protocol: {protocol!r} (choices: {NEG_WINDOW_PROTOCOL_CHOICES})")

    if protocol == "open":
        return call_qwen3_judge_open(
            api_url=api_url,
            api_key=api_key,
            model=model,
            window_text=window_text,
            max_retries=max_retries,
            timeout_s=timeout_s,
        )

    binary_results: Dict[str, JudgeResult] = {}
    for label in NEGATIVE_WINDOW_BINARY_LABELS:
        binary_results[label] = call_qwen3_judge_binary(
            api_url=api_url,
            api_key=api_key,
            model=model,
            event_label=label,
            window_text=window_text,
            max_retries=max_retries,
            timeout_s=timeout_s,
        )

    positives = [(label, res) for label, res in binary_results.items() if canonicalize_happened(res.happened) == "happened"]
    raw = json.dumps(
        {
            "protocol": "binary3",
            "binary_labels": list(NEGATIVE_WINDOW_BINARY_LABELS),
            "results": {label: asdict(res) for label, res in binary_results.items()},
        },
        ensure_ascii=False,
    )

    if not positives:
        return JudgeLabelResult(
            event_label="Background",
            evidence="",
            raw_response=raw,
            predicted_labels=[],
        )

    chosen_label, chosen = max(positives, key=lambda x: len((x[1].evidence or "").strip()))
    return JudgeLabelResult(
        event_label=chosen_label,
        evidence=chosen.evidence,
        raw_response=raw,
        predicted_labels=[label for label, _res in positives],
        postprocessed=True,
        postprocess_reason="neg_window_binary3",
    )


_VTT_TS_LINE_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})(?:\s+.*)?$"
)
_VTT_TAG_RE = re.compile(r"<[^>]+>")
_TRAILING_ELLIPSIS_RE = re.compile(r"(?:\s*(?:\.\.\.|…)+)\s*$")
_WS_RE = re.compile(r"\s+")
_LEAGUE_SEASON_RE = re.compile(r"^(?P<league>.+)_(?P<season>\d{4}-\d{4})$")


def _ts_to_seconds(ts: str) -> Optional[float]:
    try:
        hh, mm, rest = ts.split(":", 2)
        ss, ms = rest.split(".", 1)
    except ValueError:
        return None
    try:
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
    except ValueError:
        return None


def _clean_vtt_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = html.unescape(text)
    text = _VTT_TAG_RE.sub("", text)
    text = _TRAILING_ELLIPSIS_RE.sub("", text).strip()
    text = _WS_RE.sub(" ", text).strip()
    return text


def _dedupe_overlap(prev_text: str, new_text: str) -> str:
    prev_text = _WS_RE.sub(" ", (prev_text or "")).strip()
    new_text = _WS_RE.sub(" ", (new_text or "")).strip()
    if not prev_text or not new_text:
        return new_text

    prev_tokens = re.findall(r"\S+", prev_text)
    new_tokens = re.findall(r"\S+", new_text)
    max_k = min(len(prev_tokens), len(new_tokens), 20)
    for k in range(max_k, 0, -1):
        if prev_tokens[-k:] == new_tokens[:k]:
            return " ".join(new_tokens[k:]).strip()

    prev_tail = prev_text[-200:]
    max_len = min(len(prev_tail), len(new_text))
    for n in range(max_len, 4, -1):
        if prev_tail.endswith(new_text[:n]):
            return new_text[n:].lstrip()
    return new_text


def load_commentary_segments_from_vtt(vtt_path: Path) -> List[Tuple[float, float, str]]:
    segments: List[Tuple[float, float, str]] = []
    with open(vtt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    prev_tail = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line == "WEBVTT":
            continue

        m = _VTT_TS_LINE_RE.match(line)
        if not m:
            continue
        start_s = _ts_to_seconds(m.group("start"))
        end_s = _ts_to_seconds(m.group("end"))
        if start_s is None or end_s is None:
            continue

        payload: List[str] = []
        while i < len(lines) and lines[i].strip() != "":
            pl = lines[i].strip()
            if not pl.lower().startswith("infer time:"):
                payload.append(pl)
            i += 1

        raw_text = " ".join(payload)
        cleaned = _clean_vtt_text(raw_text)
        cleaned = _dedupe_overlap(prev_tail, cleaned).strip()
        cleaned = _clean_vtt_text(cleaned)
        if not cleaned:
            continue
        segments.append((float(start_s), float(end_s), cleaned))
        prev_tail = (prev_tail + " " + cleaned).strip()[-500:]

    return segments


def half_from_filename(name: str) -> Optional[int]:
    # Common: 1_720p.vtt / 2_720p.vtt
    head = re.split(r"[_-]", name, maxsplit=1)[0].strip()
    if head.isdigit():
        half = int(head)
        if half in (1, 2):
            return half
    return None


def parse_game_time_half(game_time: str) -> Optional[int]:
    # "1 - 13:10" -> 1
    if not game_time or "-" not in game_time:
        return None
    left = game_time.split("-", 1)[0].strip()
    try:
        return int(left)
    except Exception:
        return None


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


def label_file_for_infer(infer_path: Path, infer_root: Path, label_root: Path) -> Path:
    """
    SoccerNet-label(raw) directory layout:
      label_root/<league>/<season>/<match_dir>/Labels-v2.json
    StreamingVLM inference outputs (typical):
      infer_root/<league>_<season>/<match_dir>/<half>_720p.vtt
    """
    if (label_root / "annos" / "raw").is_dir():
        label_root = label_root / "annos" / "raw"
    elif (label_root / "raw").is_dir() and label_root.name != "raw":
        label_root = label_root / "raw"

    try:
        rel = infer_path.relative_to(infer_root)
    except ValueError:
        rel_parts = infer_path.parts[-4:] if len(infer_path.parts) >= 4 else (infer_path.name,)
        rel = Path(*rel_parts)

    match_dir = rel.parent.name if rel.parent != Path(".") else ""
    league_season_dir = None

    if infer_root.name and _LEAGUE_SEASON_RE.match(infer_root.name):
        league_season_dir = infer_root.name
    else:
        for parent in rel.parents:
            if parent == Path("."):
                continue
            if _LEAGUE_SEASON_RE.match(parent.name):
                league_season_dir = parent.name
                break

    if league_season_dir:
        m = _LEAGUE_SEASON_RE.match(league_season_dir)
        if m and match_dir:
            league = m.group("league")
            season = m.group("season")
            candidates = [
                label_root / league / season / match_dir / "Labels-v2.json",
                label_root / league / season / match_dir / "Labels.json",
            ]
            for cand in candidates:
                if cand.exists():
                    return cand
            return candidates[0]

    return label_root / rel.parent / "Labels-v2.json"


def infer_meta_from_path(infer_path: Path, infer_root: Path) -> Tuple[str, str]:
    try:
        rel = infer_path.relative_to(infer_root)
    except ValueError:
        rel = infer_path

    match_dir = rel.parent.name if rel.parent != Path(".") else ""
    league_season = ""
    for part in rel.parts:
        if _LEAGUE_SEASON_RE.match(part):
            league_season = part
            break
    if not league_season and _LEAGUE_SEASON_RE.match(infer_root.name):
        league_season = infer_root.name
    return league_season, match_dir


@lru_cache(maxsize=4096)
def load_label_json(label_path: str) -> Dict[str, Any]:
    return json.loads(Path(label_path).read_text(encoding="utf-8"))


def load_label_events(
    label_file: Path,
    *,
    half: int,
    league_season: str,
    match: str,
    visible_only: bool,
    include_background: bool,
    include_unknown_labels: bool,
) -> List[EventRef]:
    try:
        data = load_label_json(str(label_file))
    except Exception:
        data = {}

    out: List[EventRef] = []
    for ann in data.get("annotations", []) or []:
        label_raw = (ann.get("label") or "").strip()
        if not label_raw:
            continue
        if visible_only:
            is_vis = is_visible_visibility(ann.get("visibility"))
            if is_vis is False:
                continue
        game_time = (ann.get("gameTime") or "").strip()
        ann_half = parse_game_time_half(game_time)
        if ann_half != half:
            continue
        try:
            position_ms = float(ann.get("position", 0))
        except Exception:
            continue
        position_sec = position_ms / 1000.0
        visibility = str(ann.get("visibility") or "").strip()

        if label_raw in BACKGROUND_LABELS:
            if not include_background:
                continue
        else:
            if not include_unknown_labels and label_raw not in KEY_EVENT_LABEL_SET:
                continue

        out.append(
            EventRef(
                league_season=league_season,
                match=match,
                half=half,
                label_path=str(label_file),
                label_raw=label_raw,
                position_sec=position_sec,
                game_time=game_time,
                visibility=visibility,
            )
        )
    return out


def collect_window_text(
    segments: Sequence[Tuple[float, float, str]],
    *,
    center: float,
    pre_seconds: float,
    post_seconds: float,
) -> str:
    start_t = center - pre_seconds
    end_t = center + post_seconds
    texts = []
    for seg_start, seg_end, seg_text in segments:
        if seg_start < end_t and seg_end > start_t:
            texts.append(seg_text)
    return " ".join(texts).strip()


def collect_interval_text(
    segments: Sequence[Tuple[float, float, str]],
    *,
    start_sec: float,
    end_sec: float,
) -> str:
    texts: List[str] = []
    for seg_start, seg_end, seg_text in segments:
        if seg_start < end_sec and seg_end > start_sec:
            texts.append(seg_text)
    return " ".join(texts).strip()


def iter_sliding_windows(
    *,
    max_time_sec: float,
    window_seconds: float,
    stride_seconds: float,
) -> Iterable[Tuple[int, float, float]]:
    if window_seconds <= 0:
        return
    if stride_seconds <= 0:
        return
    if max_time_sec < 0:
        return
    n = int(math.floor(float(max_time_sec) / float(stride_seconds))) + 1
    for idx in range(n):
        start = float(idx) * float(stride_seconds)
        end = start + float(window_seconds)
        yield idx, start, end


def make_window_key(*, infer_file: str, start_sec: float, end_sec: float) -> str:
    raw = f"{SCHEMA_VERSION}|infer_file={infer_file}|start_sec={start_sec:.3f}|end_sec={end_sec:.3f}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def make_sample_key_open(*, window_key: str) -> str:
    raw = f"{SCHEMA_VERSION}|task=open|window_key={window_key}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def make_sample_key_event(*, window_key: str, event_key: str) -> str:
    raw = f"{SCHEMA_VERSION}|task=binary|window_key={window_key}|event_key={event_key}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def read_existing_jsonl(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return existing
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
                existing[sample_key] = row
    return existing


def append_jsonl(path: Path, row: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_vtt_files(infer_root: Path, *, max_files: Optional[int]) -> List[Path]:
    files = sorted(infer_root.rglob("*.vtt"))
    if max_files is not None:
        files = files[: int(max_files)]
    return files


def interval_distance_sec(*, t: float, start: float, end: float) -> float:
    if t < start:
        return float(start) - float(t)
    if t >= end:
        return float(t) - float(end)
    return 0.0


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def predicted_labels_from_open_judge(judge: Dict[str, Any]) -> List[str]:
    """
    Extract predicted key-event labels from an "open" (negative-window) judge dict.

    - For protocol=binary3, judge may contain "predicted_labels" (multi-label).
    - For protocol=open, judge contains a single "event_label".
    """

    if not isinstance(judge, dict):
        return []

    raw_predicted = judge.get("predicted_labels")
    if isinstance(raw_predicted, list) and raw_predicted:
        labels = [canonicalize_event_label(x) for x in raw_predicted]
        labels = [x for x in labels if x in KEY_EVENT_LABEL_SET]
        return _unique_preserve_order(labels)

    label = canonicalize_event_label(judge.get("event_label"))
    if label in KEY_EVENT_LABEL_SET:
        return [label]
    return []


def build_events_by_label(events: Sequence[EventRef]) -> Dict[str, List[float]]:
    events_by_label: Dict[str, List[float]] = defaultdict(list)
    for ev in events:
        events_by_label[ev.label_raw].append(float(ev.position_sec))
    for label in list(events_by_label.keys()):
        events_by_label[label].sort()
    return events_by_label


def nearest_gt_time_and_signed_window_distance(
    *,
    gt_times_sec: Sequence[float],
    window_start_sec: float,
    window_end_sec: float,
) -> Optional[Tuple[float, float]]:
    """
    Return (nearest_gt_time_sec, signed_time_diff_sec) where the signed time diff is:
      - positive: GT is before the window (window starts after GT), diff = window_start - gt
      - negative: GT is after the window (window ends before GT), diff = window_end - gt
      - zero: GT is inside [start, end)

    This corresponds to the minimum absolute distance between the point event and the window interval,
    with a sign indicating whether the window is late (+) or early (-) relative to the event.
    """

    if not gt_times_sec:
        return None
    start = float(window_start_sec)
    end = float(window_end_sec)
    if end < start:
        start, end = end, start

    i0 = bisect.bisect_left(gt_times_sec, start)
    i1 = bisect.bisect_left(gt_times_sec, end)
    if i0 < i1:
        t = float(gt_times_sec[i0])
        return t, 0.0

    best_t: Optional[float] = None
    best_diff: Optional[float] = None

    if i0 > 0:
        t_prev = float(gt_times_sec[i0 - 1])
        diff_prev = start - t_prev
        best_t, best_diff = t_prev, diff_prev

    j = i1
    if 0 <= j < len(gt_times_sec):
        t_next = float(gt_times_sec[j])
        diff_next = end - t_next  # negative if t_next > end
        if best_diff is None or abs(diff_next) < abs(best_diff):
            best_t, best_diff = t_next, diff_next

    if best_t is None or best_diff is None:
        t = float(gt_times_sec[-1])
        return t, start - t

    return best_t, float(best_diff)


def compute_otm_offsets_for_window(
    *,
    predicted_labels: Sequence[str],
    events_by_label: Dict[str, Sequence[float]],
    window_start_sec: float,
    window_end_sec: float,
    delta_seconds: float,
    review_grace_seconds: float,
    review_grace_labels: set[str],
) -> List[Dict[str, Any]]:
    """
    Compute Off-timing Mention (OTM) offsets for a negative window.

    For each predicted label:
      - skip if there's no same-label GT event at all (absolute hallucination)
      - skip if it matches a same-label GT event within ±delta of the window interval
      - skip if it's treated as review/commentary (within review_grace_seconds after last same-label GT)
      - otherwise record the signed time diff to the nearest same-label GT event point
    """

    out: List[Dict[str, Any]] = []
    start = float(window_start_sec)
    end = float(window_end_sec)
    delta = float(delta_seconds)
    grace_s = float(review_grace_seconds)

    for pred_label in predicted_labels:
        label = canonicalize_event_label(pred_label)
        if label not in KEY_EVENT_LABEL_SET:
            continue
        gt_times = events_by_label.get(label) or []
        if not gt_times:
            continue

        nearest = nearest_gt_time_and_signed_window_distance(
            gt_times_sec=gt_times,
            window_start_sec=start,
            window_end_sec=end,
        )
        if nearest is None:
            continue
        nearest_gt_t, signed_diff = nearest

        if abs(float(signed_diff)) <= delta:
            continue

        if label in review_grace_labels and grace_s > 0:
            idx = bisect.bisect_left(gt_times, start)
            if idx > 0 and (start - float(gt_times[idx - 1])) <= grace_s:
                continue

        out.append(
            {
                "pred_label": label,
                "nearest_gt_sec": float(nearest_gt_t),
                "signed_time_diff_sec": float(signed_diff),
                "abs_time_diff_sec": float(abs(float(signed_diff))),
            }
        )

    return out


def key_events_only(events: Sequence[EventRef]) -> List[EventRef]:
    return [ev for ev in events if ev.label_raw in KEY_EVENT_LABEL_SET]


def events_in_interval(events: Sequence[EventRef], *, start: float, end: float) -> List[EventRef]:
    return [ev for ev in events if float(start) <= float(ev.position_sec) < float(end)]


def count_total_and_done_sliding(
    *,
    vtt_files: Sequence[Path],
    infer_root: Path,
    labels_root: Path,
    processed_keys: Sequence[str],
    window_seconds: float,
    stride_seconds: float,
    visible_only: bool,
    include_background: bool,
    include_unknown_labels: bool,
) -> Tuple[int, int]:
    processed = set(processed_keys)
    total = 0
    done = 0
    for infer_file in vtt_files:
        half = half_from_filename(infer_file.name)
        if half is None:
            continue
        label_path = label_file_for_infer(infer_file, infer_root, labels_root)
        if not label_path.exists():
            continue
        league_season, match_dir = infer_meta_from_path(infer_file, infer_root)
        events = key_events_only(
            load_label_events(
                label_path,
                half=half,
                league_season=league_season,
                match=match_dir,
                visible_only=visible_only,
                include_background=include_background,
                include_unknown_labels=include_unknown_labels,
            )
        )
        try:
            segments = load_commentary_segments_from_vtt(infer_file)
        except Exception:
            segments = []

        if not events and not segments:
            continue

        max_seg_end = max((end for _, end, _ in segments), default=0.0)
        max_event_t = max((ev.position_sec for ev in events), default=0.0)
        max_time = max(float(max_seg_end), float(max_event_t))

        for _, start_sec, end_sec in iter_sliding_windows(
            max_time_sec=max_time,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        ):
            window_key = make_window_key(infer_file=str(infer_file), start_sec=start_sec, end_sec=end_sec)
            win_events = events_in_interval(events, start=start_sec, end=end_sec)
            if win_events:
                for ev in win_events:
                    sample_key = make_sample_key_event(window_key=window_key, event_key=ev.key)
                    total += 1
                    if sample_key in processed:
                        done += 1
            else:
                sample_key = make_sample_key_open(window_key=window_key)
                total += 1
                if sample_key in processed:
                    done += 1
    return total, done


def _row_predicted_labels(row: Dict[str, Any]) -> set[str]:
    task = row.get("task")
    judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}

    if task == "binary":
        happened_norm = canonicalize_happened(judge.get("happened"))
        if happened_norm != "happened":
            return set()
        ev = row.get("event") if isinstance(row.get("event"), dict) else {}
        label = ev.get("label_raw")
        if isinstance(label, str) and label in KEY_EVENT_LABEL_SET:
            return {label}
        return set()

    if task == "open":
        predicted_labels = judge.get("predicted_labels")
        if isinstance(predicted_labels, list):
            out: set[str] = set()
            for item in predicted_labels:
                label = canonicalize_event_label(item)
                if label in KEY_EVENT_LABEL_SET:
                    out.add(label)
            if out:
                return out

        label = canonicalize_event_label(judge.get("event_label"))
        if label in KEY_EVENT_LABEL_SET:
            return {label}
        return set()

    return set()


def compute_sliding_window_metrics(
    *,
    vtt_files: Sequence[Path],
    infer_root: Path,
    labels_root: Path,
    rows_by_sample_key: Dict[str, Dict[str, Any]],
    delta_seconds: float,
    review_grace_seconds: float = REVIEW_GRACE_SECONDS_DEFAULT,
    review_grace_labels: Sequence[str] = REVIEW_GRACE_LABELS_DEFAULT,
    visible_only: bool,
    include_background: bool,
    include_unknown_labels: bool,
) -> Dict[str, Any]:
    # Aggregate predictions per window.
    window_meta: Dict[str, Tuple[str, float, float]] = {}
    pred_labels_by_window: Dict[str, set[str]] = defaultdict(set)
    open_window_keys: set[str] = set()

    for row in rows_by_sample_key.values():
        if row.get("schema_version") != SCHEMA_VERSION:
            continue
        window_key = row.get("window_key")
        if not isinstance(window_key, str) or not window_key:
            continue
        infer_file = row.get("infer_file")
        if not isinstance(infer_file, str) or not infer_file:
            continue
        window = row.get("window") if isinstance(row.get("window"), dict) else {}
        start = window.get("start_sec")
        end = window.get("end_sec")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        window_meta[window_key] = (infer_file, float(start), float(end))
        pred_labels_by_window[window_key].update(_row_predicted_labels(row))
        if row.get("task") == "open":
            open_window_keys.add(window_key)

    windows_by_file: Dict[str, List[Tuple[float, float, set[str], bool]]] = defaultdict(list)
    for window_key, (infer_file, start, end) in window_meta.items():
        windows_by_file[infer_file].append(
            (start, end, pred_labels_by_window.get(window_key, set()), window_key in open_window_keys)
        )
    for infer_file in list(windows_by_file.keys()):
        windows_by_file[infer_file].sort(key=lambda x: (x[0], x[1]))

    total_events = 0
    recalled_events = 0
    per_label_total: Dict[str, int] = defaultdict(int)
    per_label_recalled: Dict[str, int] = defaultdict(int)

    neg_windows_total = 0
    neg_windows_predicted = 0
    neg_windows_matched = 0
    neg_windows_fp = 0
    fp_by_label: Dict[str, int] = defaultdict(int)
    abs_halluc_by_label: Dict[str, int] = defaultdict(int)
    off_timing_by_label: Dict[str, int] = defaultdict(int)
    review_by_label: Dict[str, int] = defaultdict(int)
    neg_windows_abs_halluc = 0
    neg_windows_off_timing = 0
    neg_windows_review = 0

    # Mention-level TP/FP_off/FP_abs counts over predicted labels in negative/open windows.
    pred_mentions_total = 0
    pred_tp_total = 0
    pred_fp_off_total = 0
    pred_fp_abs_total = 0
    pred_mentions_by_label: Dict[str, int] = defaultdict(int)
    pred_tp_by_label: Dict[str, int] = defaultdict(int)
    pred_fp_off_by_label: Dict[str, int] = defaultdict(int)
    pred_fp_abs_by_label: Dict[str, int] = defaultdict(int)

    grace_labels: set[str] = set()
    for label in review_grace_labels:
        canon = canonicalize_event_label(label)
        if canon:
            grace_labels.add(canon)

    for infer_file in vtt_files:
        half = half_from_filename(infer_file.name)
        if half is None:
            continue
        label_path = label_file_for_infer(infer_file, infer_root, labels_root)
        if not label_path.exists():
            continue
        league_season, match_dir = infer_meta_from_path(infer_file, infer_root)
        events = key_events_only(
            load_label_events(
                label_path,
                half=half,
                league_season=league_season,
                match=match_dir,
                visible_only=visible_only,
                include_background=include_background,
                include_unknown_labels=include_unknown_labels,
            )
        )

        events_by_label: Dict[str, List[float]] = defaultdict(list)
        for ev in events:
            events_by_label[ev.label_raw].append(float(ev.position_sec))
        for label in list(events_by_label.keys()):
            events_by_label[label].sort()

        win_list = windows_by_file.get(str(infer_file), [])

        for ev in events:
            total_events += 1
            per_label_total[ev.label_raw] += 1
            t = float(ev.position_sec)
            label = ev.label_raw

            matched = False
            for w_start, w_end, w_pred, _is_open in win_list:
                if label not in w_pred:
                    continue
                if interval_distance_sec(t=t, start=w_start, end=w_end) <= float(delta_seconds):
                    matched = True
                    break
            if matched:
                recalled_events += 1
                per_label_recalled[label] += 1

        for w_start, w_end, w_pred, is_open in win_list:
            if not is_open:
                continue
            neg_windows_total += 1
            if not w_pred:
                continue
            neg_windows_predicted += 1

            window_has_fp = False
            window_has_abs_halluc = False
            window_has_review = False
            for pred_label in w_pred:
                label_event_times = events_by_label.get(pred_label) or []
                matched = any(
                    interval_distance_sec(t=et, start=w_start, end=w_end) <= float(delta_seconds)
                    for et in label_event_times
                )
                if matched:
                    pred_mentions_total += 1
                    pred_mentions_by_label[pred_label] += 1
                    pred_tp_total += 1
                    pred_tp_by_label[pred_label] += 1
                    continue
                pred_mentions_total += 1
                pred_mentions_by_label[pred_label] += 1
                if label_event_times:
                    pred_fp_off_total += 1
                    pred_fp_off_by_label[pred_label] += 1
                else:
                    pred_fp_abs_total += 1
                    pred_fp_abs_by_label[pred_label] += 1
                if label_event_times and pred_label in grace_labels and review_grace_seconds > 0:
                    # Allow review/commentary: if there's a recent past GT event of the same label.
                    # (We use window start as an approximate mention time; intentionally lenient.)
                    idx = bisect.bisect_left(label_event_times, w_start)
                    if idx > 0 and (w_start - label_event_times[idx - 1]) <= float(review_grace_seconds):
                        window_has_review = True
                        review_by_label[pred_label] += 1
                        continue
                window_has_fp = True
                fp_by_label[pred_label] += 1
                if label_event_times:
                    off_timing_by_label[pred_label] += 1
                else:
                    window_has_abs_halluc = True
                    abs_halluc_by_label[pred_label] += 1
            if window_has_fp:
                neg_windows_fp += 1
                if window_has_abs_halluc:
                    neg_windows_abs_halluc += 1
                else:
                    neg_windows_off_timing += 1
            else:
                neg_windows_matched += 1
                if window_has_review:
                    neg_windows_review += 1

    faithfulness_precision = (
        (float(pred_tp_total + pred_fp_off_total) / float(pred_mentions_total)) if pred_mentions_total else 0.0
    )
    temporal_precision = (float(pred_tp_total) / float(pred_mentions_total)) if pred_mentions_total else 0.0
    overall_event_recall = (recalled_events / total_events) if total_events else 0.0
    overall_temporal_f1 = f1_from_precision_recall(temporal_precision, overall_event_recall)

    per_label: Dict[str, Any] = {}
    macro_labels: List[str] = []
    for label in KEY_EVENT_LABELS:
        tot = int(per_label_total.get(label, 0))
        rec = int(per_label_recalled.get(label, 0))
        if tot > 0:
            macro_labels.append(label)
        neg_total = int(neg_windows_total)
        fp_cnt = int(fp_by_label.get(label, 0))
        abs_cnt = int(abs_halluc_by_label.get(label, 0))
        off_cnt = int(off_timing_by_label.get(label, 0))
        pred_cnt = int(pred_mentions_by_label.get(label, 0))
        pred_tp = int(pred_tp_by_label.get(label, 0))
        pred_fp_off = int(pred_fp_off_by_label.get(label, 0))
        pred_fp_abs = int(pred_fp_abs_by_label.get(label, 0))
        label_event_recall = (rec / tot) if tot else 0.0
        label_faithfulness_precision = (
            (float(pred_tp + pred_fp_off) / float(pred_cnt)) if pred_cnt else 0.0
        )
        label_temporal_precision = (float(pred_tp) / float(pred_cnt)) if pred_cnt else 0.0
        label_temporal_f1 = f1_from_precision_recall(label_temporal_precision, label_event_recall)
        per_label[label] = {
            "total_events": tot,
            "recalled_events": rec,
            "event_recall@delta": float(label_event_recall),
            "false_positives_in_neg_windows": fp_cnt,
            "false_positive_rate": (fp_cnt / neg_total) if neg_total else 0.0,
            "absolute_hallucinations_in_neg_windows": abs_cnt,
            "absolute_hallucination_rate": (abs_cnt / neg_total) if neg_total else 0.0,
            "off_timing_mentions_in_neg_windows": off_cnt,
            "off_timing_mention_rate": (off_cnt / neg_total) if neg_total else 0.0,
            "review_mentions_in_neg_windows": int(review_by_label.get(label, 0)),
            "review_rate": (int(review_by_label.get(label, 0)) / neg_total) if neg_total else 0.0,
            "pred_mentions_in_neg_windows": pred_cnt,
            "tp_mentions_in_neg_windows": pred_tp,
            "fp_off_mentions_in_neg_windows": pred_fp_off,
            "fp_abs_mentions_in_neg_windows": pred_fp_abs,
            "faithfulness_precision": float(label_faithfulness_precision),
            "temporal_precision": float(label_temporal_precision),
            "temporal_f1": float(label_temporal_f1),
        }

    macro_event_recall = 0.0
    macro_false_positive_rate = 0.0
    macro_abs_hallucination_rate = 0.0
    macro_off_timing_mention_rate = 0.0
    macro_faithfulness_precision = 0.0
    macro_temporal_precision = 0.0
    macro_temporal_f1 = 0.0
    macro_precision_labels: List[str] = [
        label for label in KEY_EVENT_LABELS if int(pred_mentions_by_label.get(label, 0)) > 0
    ]
    macro_f1_labels: List[str] = [
        label
        for label in KEY_EVENT_LABELS
        if int(per_label_total.get(label, 0)) > 0 or int(pred_mentions_by_label.get(label, 0)) > 0
    ]
    if macro_labels:
        macro_event_recall = (
            sum(per_label[label]["event_recall@delta"] for label in macro_labels) / float(len(macro_labels))
        )
        macro_false_positive_rate = (
            sum(per_label[label]["false_positive_rate"] for label in macro_labels) / float(len(macro_labels))
        )
        macro_abs_hallucination_rate = (
            sum(per_label[label]["absolute_hallucination_rate"] for label in macro_labels) / float(len(macro_labels))
        )
        macro_off_timing_mention_rate = (
            sum(per_label[label]["off_timing_mention_rate"] for label in macro_labels) / float(len(macro_labels))
        )
    if macro_precision_labels:
        macro_faithfulness_precision = (
            sum(per_label[label]["faithfulness_precision"] for label in macro_precision_labels)
            / float(len(macro_precision_labels))
        )
        macro_temporal_precision = (
            sum(per_label[label]["temporal_precision"] for label in macro_precision_labels)
            / float(len(macro_precision_labels))
        )
    if macro_f1_labels:
        macro_temporal_f1 = (
            sum(per_label[label]["temporal_f1"] for label in macro_f1_labels) / float(len(macro_f1_labels))
        )

    return {
        "overall": {
            "total_events": int(total_events),
            "recalled_events": int(recalled_events),
            "event_recall@delta": float(overall_event_recall),
            "negative_windows_evaluated": int(neg_windows_total),
            "negative_windows_predicted": int(neg_windows_predicted),
            "negative_windows_matched": int(neg_windows_matched),
            "false_positive_windows": int(neg_windows_fp),
            "false_positive_rate": (neg_windows_fp / neg_windows_total) if neg_windows_total else 0.0,
            "absolute_hallucination_windows": int(neg_windows_abs_halluc),
            "absolute_hallucination_rate": (neg_windows_abs_halluc / neg_windows_total) if neg_windows_total else 0.0,
            "off_timing_mention_windows": int(neg_windows_off_timing),
            "off_timing_mention_rate": (neg_windows_off_timing / neg_windows_total) if neg_windows_total else 0.0,
            "review_windows": int(neg_windows_review),
            "review_rate": (neg_windows_review / neg_windows_total) if neg_windows_total else 0.0,
            "pred_mentions_in_neg_windows": int(pred_mentions_total),
            "tp_mentions_in_neg_windows": int(pred_tp_total),
            "fp_off_mentions_in_neg_windows": int(pred_fp_off_total),
            "fp_abs_mentions_in_neg_windows": int(pred_fp_abs_total),
            "faithfulness_precision": float(faithfulness_precision),
            "temporal_precision": float(temporal_precision),
            "temporal_f1": float(overall_temporal_f1),
        },
        "macro_avg": {
            "num_labels": int(len(macro_labels)),
            "event_recall@delta": float(macro_event_recall),
            "false_positive_rate": float(macro_false_positive_rate),
            "absolute_hallucination_rate": float(macro_abs_hallucination_rate),
            "off_timing_mention_rate": float(macro_off_timing_mention_rate),
            "num_labels_with_predictions": int(len(macro_precision_labels)),
            "num_labels_for_temporal_f1": int(len(macro_f1_labels)),
            "faithfulness_precision": float(macro_faithfulness_precision),
            "temporal_precision": float(macro_temporal_precision),
            "temporal_f1": float(macro_temporal_f1),
        },
        "per_label": per_label,
    }


def evaluate_one_model(
    *,
    infer_root: Path,
    labels_root: Path,
    out_dir: Path,
    model_dir_name: str,
    neg_window_protocol: str,
    judge_api_url: str,
    judge_api_key: Optional[str],
    judge_model: str,
    window_seconds: float,
    stride_seconds: float,
    delta_seconds: float,
    review_grace_seconds: float,
    review_grace_labels: Sequence[str],
    visible_only: bool,
    include_background: bool,
    include_unknown_labels: bool,
    num_workers: int,
    request_timeout_s: float,
    max_retries: int,
    max_files: Optional[int],
    dry_run: bool,
):
    vtt_files = iter_vtt_files(infer_root, max_files=max_files)
    if not vtt_files:
        print(f"[skip] no vtt files under: {infer_root}")
        return None

    model_out_dir = out_dir / model_dir_name
    out_jsonl = model_out_dir / "sliding_window_event_mentions.jsonl"
    out_summary = model_out_dir / "sliding_window_event_mentions_summary.json"

    existing = read_existing_jsonl(out_jsonl)
    processed_keys = set(existing.keys())

    total_samples, done_samples = count_total_and_done_sliding(
        vtt_files=vtt_files,
        infer_root=infer_root,
        labels_root=labels_root,
        processed_keys=sorted(processed_keys),
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        visible_only=visible_only,
        include_background=include_background,
        include_unknown_labels=include_unknown_labels,
    )

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=total_samples,
            initial=done_samples,
            desc=f"{model_dir_name}",
            unit="sample",
            dynamic_ncols=True,
        )

    executor: Optional[ThreadPoolExecutor] = None
    if not dry_run and num_workers > 1:
        executor = ThreadPoolExecutor(max_workers=num_workers)

    in_flight: Dict[Future[Any], Dict[str, Any]] = {}
    in_progress_keys: set[str] = set()
    error_samples = 0

    grace_label_set: set[str] = set()
    for label in review_grace_labels:
        canon = canonicalize_event_label(label)
        if canon in KEY_EVENT_LABEL_SET:
            grace_label_set.add(canon)

    events_by_label_cache: Dict[Tuple[str, int], Dict[str, List[float]]] = {}

    def get_events_by_label_for_row(row: Dict[str, Any]) -> Dict[str, List[float]]:
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        half = meta.get("half")
        label_path = meta.get("label_path")
        league_season = meta.get("league_season")
        match_dir = meta.get("match")
        if not isinstance(half, int) or half not in (1, 2):
            return {}
        if not isinstance(label_path, str) or not label_path:
            return {}
        if not isinstance(league_season, str) or not league_season:
            league_season = ""
        if not isinstance(match_dir, str) or not match_dir:
            match_dir = ""

        key = (label_path, int(half))
        cached = events_by_label_cache.get(key)
        if cached is not None:
            return cached

        path = Path(label_path)
        if not path.exists():
            events_by_label_cache[key] = {}
            return {}

        events = key_events_only(
            load_label_events(
                path,
                half=int(half),
                league_season=str(league_season),
                match=str(match_dir),
                visible_only=visible_only,
                include_background=include_background,
                include_unknown_labels=include_unknown_labels,
            )
        )
        built = build_events_by_label(events)
        events_by_label_cache[key] = built
        return built

    def annotate_otm_offsets(row: Dict[str, Any]):
        if row.get("task") != "open":
            return
        judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
        predicted = predicted_labels_from_open_judge(judge)
        if not predicted:
            return
        window = row.get("window") if isinstance(row.get("window"), dict) else {}
        start = window.get("start_sec")
        end = window.get("end_sec")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return
        offsets = compute_otm_offsets_for_window(
            predicted_labels=predicted,
            events_by_label=get_events_by_label_for_row(row),
            window_start_sec=float(start),
            window_end_sec=float(end),
            delta_seconds=float(delta_seconds),
            review_grace_seconds=float(review_grace_seconds),
            review_grace_labels=grace_label_set,
        )
        if offsets:
            row["otm_nearest_gt"] = offsets

    def consume_and_write_row(row: Dict[str, Any]):
        append_jsonl(out_jsonl, row)
        existing[row["sample_key"]] = row
        processed_keys.add(row["sample_key"])
        if pbar is not None:
            pbar.update(1)

    def drain_futures(*, blocking: bool):
        nonlocal error_samples
        if not in_flight:
            return
        timeout = None if blocking else 0.0
        done, _ = wait(in_flight, timeout=timeout, return_when=FIRST_COMPLETED)
        for fut in done:
            row = in_flight.pop(fut)
            sample_key = row.get("sample_key", "")
            if isinstance(sample_key, str) and sample_key:
                in_progress_keys.discard(sample_key)
            try:
                judge = fut.result()
            except Exception as e:
                error_samples += 1
                print(f"[judge_error] model={model_dir_name} sample_key={sample_key} err={e}", file=sys.stderr, flush=True)
                if pbar is not None:
                    pbar.update(1)
                continue
            row["judge"] = asdict(judge)
            annotate_otm_offsets(row)
            consume_and_write_row(row)

    try:
        for infer_file in vtt_files:
            half = half_from_filename(infer_file.name)
            if half is None:
                continue

            label_path = label_file_for_infer(infer_file, infer_root, labels_root)
            if not label_path.exists():
                continue

            league_season, match_dir = infer_meta_from_path(infer_file, infer_root)
            events = key_events_only(
                load_label_events(
                    label_path,
                    half=half,
                    league_season=league_season,
                    match=match_dir,
                    visible_only=visible_only,
                    include_background=include_background,
                    include_unknown_labels=include_unknown_labels,
                )
            )
            events_by_label_cache[(str(label_path), int(half))] = build_events_by_label(events)
            try:
                segments = load_commentary_segments_from_vtt(infer_file)
            except Exception as e:
                print(f"[vtt_error] file={infer_file} err={e}", file=sys.stderr, flush=True)
                segments = []

            if not events and not segments:
                continue

            max_seg_end = max((end for _, end, _ in segments), default=0.0)
            max_event_t = max((ev.position_sec for ev in events), default=0.0)
            max_time = max(float(max_seg_end), float(max_event_t))

            for window_idx, start_sec, end_sec in iter_sliding_windows(
                max_time_sec=max_time,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
            ):
                window_key = make_window_key(infer_file=str(infer_file), start_sec=start_sec, end_sec=end_sec)
                win_events = events_in_interval(events, start=start_sec, end=end_sec)

                if win_events:
                    pending: List[Tuple[EventRef, str]] = []
                    for ev in win_events:
                        sample_key = make_sample_key_event(window_key=window_key, event_key=ev.key)
                        if sample_key in processed_keys or sample_key in in_progress_keys:
                            continue
                        pending.append((ev, sample_key))
                    if not pending:
                        continue

                    window_text = ""
                    if segments:
                        window_text = collect_interval_text(segments, start_sec=start_sec, end_sec=end_sec)

                    for ev, sample_key in pending:
                        row: Dict[str, Any] = {
                            "schema_version": SCHEMA_VERSION,
                            "sample_key": sample_key,
                            "task": "binary",
                            "model_dir": model_dir_name,
                            "infer_file": str(infer_file),
                            "window_key": window_key,
                            "window_idx": int(window_idx),
                            "event": {
                                "league_season": ev.league_season,
                                "match": ev.match,
                                "half": ev.half,
                                "label_path": ev.label_path,
                                "label_raw": ev.label_raw,
                                "position_sec": ev.position_sec,
                                "game_time": ev.game_time,
                                "visibility": ev.visibility,
                            },
                            "window": {
                                "window_seconds": float(window_seconds),
                                "stride_seconds": float(stride_seconds),
                                "start_sec": float(start_sec),
                                "end_sec": float(end_sec),
                                "text": window_text,
                            },
                        }

                        if dry_run:
                            row["judge"] = {
                                "happened": "not_happened",
                                "evidence": "",
                                "raw_response": "",
                            }
                            consume_and_write_row(row)
                            continue

                        if not window_text:
                            row["judge"] = {
                                "happened": "not_happened",
                                "evidence": "",
                                "raw_response": "",
                            }
                            row["skip_reason"] = "empty_window_text"
                            consume_and_write_row(row)
                            continue

                        if executor is None:
                            try:
                                judge = call_qwen3_judge_binary(
                                    api_url=judge_api_url,
                                    api_key=judge_api_key,
                                    model=judge_model,
                                    event_label=ev.label_raw,
                                    window_text=window_text,
                                    max_retries=max_retries,
                                    timeout_s=request_timeout_s,
                                )
                            except Exception as e:
                                error_samples += 1
                                print(
                                    f"[judge_error] model={model_dir_name} sample_key={sample_key} err={e}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                if pbar is not None:
                                    pbar.update(1)
                                continue
                            row["judge"] = asdict(judge)
                            consume_and_write_row(row)
                            continue

                        in_progress_keys.add(sample_key)
                        fut = executor.submit(
                            call_qwen3_judge_binary,
                            api_url=judge_api_url,
                            api_key=judge_api_key,
                            model=judge_model,
                            event_label=ev.label_raw,
                            window_text=window_text,
                            max_retries=max_retries,
                            timeout_s=request_timeout_s,
                        )
                        in_flight[fut] = row

                        while len(in_flight) >= max(num_workers, 1):
                            drain_futures(blocking=True)
                else:
                    sample_key = make_sample_key_open(window_key=window_key)
                    if sample_key in processed_keys or sample_key in in_progress_keys:
                        continue

                    window_text = ""
                    if segments:
                        window_text = collect_interval_text(segments, start_sec=start_sec, end_sec=end_sec)

                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "sample_key": sample_key,
                        "task": "open",
                        "model_dir": model_dir_name,
                        "infer_file": str(infer_file),
                        "window_key": window_key,
                        "window_idx": int(window_idx),
                        "meta": {
                            "league_season": league_season,
                            "match": match_dir,
                            "half": int(half),
                            "label_path": str(label_path),
                        },
                        "window": {
                            "window_seconds": float(window_seconds),
                            "stride_seconds": float(stride_seconds),
                            "start_sec": float(start_sec),
                            "end_sec": float(end_sec),
                            "text": window_text,
                        },
                    }

                    if dry_run:
                        row["judge"] = {
                            "event_label": "Background",
                            "evidence": "",
                            "raw_response": "",
                            "predicted_labels": [],
                        }
                        consume_and_write_row(row)
                        continue

                    if not window_text:
                        row["judge"] = {
                            "event_label": "Background",
                            "evidence": "",
                            "raw_response": "",
                            "predicted_labels": [],
                        }
                        row["skip_reason"] = "empty_window_text"
                        consume_and_write_row(row)
                        continue

                    if executor is None:
                        try:
                            judge = call_qwen3_judge_negative_window(
                                api_url=judge_api_url,
                                api_key=judge_api_key,
                                model=judge_model,
                                window_text=window_text,
                                protocol=neg_window_protocol,
                                max_retries=max_retries,
                                timeout_s=request_timeout_s,
                            )
                        except Exception as e:
                            error_samples += 1
                            print(
                                f"[judge_error] model={model_dir_name} sample_key={sample_key} err={e}",
                                file=sys.stderr,
                                flush=True,
                            )
                            if pbar is not None:
                                pbar.update(1)
                            continue
                        row["judge"] = asdict(judge)
                        annotate_otm_offsets(row)
                        consume_and_write_row(row)
                        continue

                    in_progress_keys.add(sample_key)
                    fut = executor.submit(
                        call_qwen3_judge_negative_window,
                        api_url=judge_api_url,
                        api_key=judge_api_key,
                        model=judge_model,
                        window_text=window_text,
                        protocol=neg_window_protocol,
                        max_retries=max_retries,
                        timeout_s=request_timeout_s,
                    )
                    in_flight[fut] = row

                    while len(in_flight) >= max(num_workers, 1):
                        drain_futures(blocking=True)

            drain_futures(blocking=False)

        while in_flight:
            drain_futures(blocking=True)
    finally:
        if pbar is not None:
            pbar.close()
        if executor is not None:
            executor.shutdown(wait=True)

    metrics = compute_sliding_window_metrics(
        vtt_files=vtt_files,
        infer_root=infer_root,
        labels_root=labels_root,
        rows_by_sample_key=existing,
        delta_seconds=float(delta_seconds),
        review_grace_seconds=float(review_grace_seconds),
        review_grace_labels=list(review_grace_labels),
        visible_only=visible_only,
        include_background=include_background,
        include_unknown_labels=include_unknown_labels,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "infer_root": str(infer_root),
            "labels_root": str(labels_root),
            "model_dir": model_dir_name,
            "window_seconds": float(window_seconds),
            "stride_seconds": float(stride_seconds),
            "delta_seconds": float(delta_seconds),
            "review_grace_seconds": float(review_grace_seconds),
            "review_grace_labels": list(review_grace_labels),
            "visible_only": bool(visible_only),
            "include_background": bool(include_background),
            "include_unknown_labels": bool(include_unknown_labels),
            "neg_window_protocol": str(neg_window_protocol),
            "neg_window_binary_labels": list(NEGATIVE_WINDOW_BINARY_LABELS),
            "judge_api_url": str(judge_api_url),
            "judge_model": str(judge_model),
            "dry_run": bool(dry_run),
            "num_workers": int(num_workers),
            "request_timeout_s": float(request_timeout_s),
            "max_retries": int(max_retries),
            "error_samples": int(error_samples),
            "total_samples": int(total_samples),
            "done_samples": int(len(existing)),
        },
        **metrics,
    }

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[done] model={model_dir_name} "
        f"events={summary['overall']['total_events']} "
        f"recall@{float(delta_seconds):g}={summary['overall']['event_recall@delta']:.4f} "
        f"fpr={summary['overall']['false_positive_rate']:.4f} "
        f"ahr={summary['overall'].get('absolute_hallucination_rate', 0.0):.4f} "
        f"omr={summary['overall'].get('off_timing_mention_rate', 0.0):.4f} "
        f"p_faith={summary['overall'].get('faithfulness_precision', 0.0):.4f} "
        f"p_on={summary['overall'].get('temporal_precision', 0.0):.4f} "
        f"f1_on={summary['overall'].get('temporal_f1', 0.0):.4f} "
        f"jsonl={out_jsonl} summary={out_summary}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-judge: SoccerNet key event mentions from VTT (sliding windows)")
    infer_group = parser.add_mutually_exclusive_group()
    infer_group.add_argument(
        "--infer-root",
        type=Path,
        default=None,
        help="Single model inference root (ends with model dir name)",
    )
    infer_group.add_argument(
        "--infer-roots",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit inference roots to evaluate sequentially (each should contain .vtt files)",
    )
    parser.add_argument(
        "--infer-models",
        nargs="*",
        default=None,
        help="Model dir names under infer_root.parent to evaluate sequentially (default: [infer_root.name])",
    )
    parser.add_argument(
        "--model-names",
        nargs="*",
        default=None,
        help="Optional output names for --infer-roots (must match length)",
    )
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABEL_ROOT, help="SoccerNet-label root (supports {root}/annos/raw)")
    parser.add_argument("--window-seconds", type=float, default=30.0, help="Sliding window size in seconds (default: 30)")
    parser.add_argument("--stride-seconds", type=float, default=15.0, help="Sliding window stride in seconds (default: 15)")
    parser.add_argument("--delta-seconds", type=float, default=15.0, help="Tolerance δ in seconds for Recall@δ and FPR matching (default: 15)")
    parser.add_argument(
        "--review-grace-seconds",
        type=float,
        default=REVIEW_GRACE_SECONDS_DEFAULT,
        help="If a predicted label is mentioned within this many seconds after the last same-label GT event, do not count it as a false positive (default: 120).",
    )
    parser.add_argument(
        "--review-grace-labels",
        nargs="*",
        default=list(REVIEW_GRACE_LABELS_DEFAULT),
        help='Labels to apply --review-grace-seconds to (default: Goal Foul Substitution).',
    )
    parser.add_argument("--all-events", dest="visible_only", action="store_false", help="Include non-visible events")
    parser.add_argument("--visible-only", dest="visible_only", action="store_true", default=True, help="Only keep visibility=visible events (default)")
    parser.add_argument("--include-background", default=True, help='Include background labels (e.g., "Background"/"Negative")')
    parser.add_argument("--include-unknown-labels", action="store_true", help="Include labels beyond the predefined key-event label set")
    parser.add_argument(
        "--neg-window-protocol",
        type=str,
        default=DEFAULT_NEG_WINDOW_PROTOCOL,
        choices=list(NEG_WINDOW_PROTOCOL_CHOICES),
        help='Negative-window judging protocol: "binary3" (binary entailment over configured labels) or "open" (legacy 17-way open-set).',
    )
    parser.add_argument("--num-workers", type=int, default=128, help="Parallel workers for judge calls")
    parser.add_argument("--api-url", type=str, default=os.environ.get("QWEN_API_URL", "http://localhost:8000/v1/chat/completions"))
    parser.add_argument("--api-key", type=str, default=os.environ.get("QWEN_API_KEY", None))
    parser.add_argument("--judge-model", type=str, default=os.environ.get("QWEN_MODEL", "Qwen3-235B-A22B-Instruct-2507"))
    parser.add_argument("--request-timeout-s", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/llm_as_a_judge/results"))
    parser.add_argument("--max-files", type=int, default=None, help="Max VTT files to process (debug)")
    parser.add_argument("--dry-run", action="store_true", help="Do not call judge; write dummy results")
    args = parser.parse_args()

    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be > 0")
    if args.stride_seconds <= 0:
        raise ValueError("--stride-seconds must be > 0")
    if args.delta_seconds < 0:
        raise ValueError("--delta-seconds must be >= 0")

    if args.infer_root is None and args.infer_roots is None:
        if args.infer_models:
            if not DEFAULT_INFER_ROOTS:
                raise ValueError("DEFAULT_INFER_ROOTS is empty; please pass --infer-root with --infer-models")
            args.infer_root = DEFAULT_INFER_ROOTS[0]
        else:
            if not DEFAULT_INFER_ROOTS:
                raise ValueError("DEFAULT_INFER_ROOTS is empty; please pass --infer-root or --infer-roots")
            args.infer_roots = list(DEFAULT_INFER_ROOTS)

    summaries: List[Dict[str, Any]] = []

    if args.infer_roots:
        if args.infer_models:
            raise ValueError("--infer-models cannot be used with --infer-roots")
        if args.model_names and len(args.model_names) != len(args.infer_roots):
            raise ValueError("--model-names length must match --infer-roots length")

        used_names: Dict[str, int] = defaultdict(int)
        for idx, infer_root in enumerate(args.infer_roots):
            model_name = (args.model_names[idx] if args.model_names else infer_root.name) or f"model_{idx}"
            used_names[model_name] += 1
            if used_names[model_name] > 1:
                model_name = f"{model_name}__{used_names[model_name]}"

            summary = evaluate_one_model(
                infer_root=infer_root,
                labels_root=args.labels_root,
                out_dir=args.out_dir,
                model_dir_name=model_name,
                neg_window_protocol=str(args.neg_window_protocol),
                judge_api_url=args.api_url,
                judge_api_key=args.api_key,
                judge_model=args.judge_model,
                window_seconds=float(args.window_seconds),
                stride_seconds=float(args.stride_seconds),
                delta_seconds=float(args.delta_seconds),
                review_grace_seconds=float(args.review_grace_seconds),
                review_grace_labels=list(args.review_grace_labels),
                visible_only=args.visible_only,
                include_background=args.include_background,
                include_unknown_labels=args.include_unknown_labels,
                num_workers=max(int(args.num_workers), 1),
                request_timeout_s=float(args.request_timeout_s),
                max_retries=int(args.max_retries),
                max_files=args.max_files,
                dry_run=bool(args.dry_run),
            )
            if summary:
                summaries.append(summary)
    else:
        infer_models = args.infer_models if args.infer_models else [args.infer_root.name]
        infer_base = args.infer_root.parent

        for model_name in infer_models:
            model_infer_root = infer_base / model_name
            summary = evaluate_one_model(
                infer_root=model_infer_root,
                labels_root=args.labels_root,
                out_dir=args.out_dir,
                model_dir_name=model_name,
                neg_window_protocol=str(args.neg_window_protocol),
                judge_api_url=args.api_url,
                judge_api_key=args.api_key,
                judge_model=args.judge_model,
                window_seconds=float(args.window_seconds),
                stride_seconds=float(args.stride_seconds),
                delta_seconds=float(args.delta_seconds),
                review_grace_seconds=float(args.review_grace_seconds),
                review_grace_labels=list(args.review_grace_labels),
                visible_only=args.visible_only,
                include_background=args.include_background,
                include_unknown_labels=args.include_unknown_labels,
                num_workers=max(int(args.num_workers), 1),
                request_timeout_s=float(args.request_timeout_s),
                max_retries=int(args.max_retries),
                max_files=args.max_files,
                dry_run=bool(args.dry_run),
            )
            if summary:
                summaries.append(summary)

    if summaries:
        index = {
            "models": [
                {
                    "model_dir": s.get("config", {}).get("model_dir"),
                    "infer_root": s.get("config", {}).get("infer_root"),
                    "total_events": s.get("overall", {}).get("total_events"),
                    "event_recall@delta": s.get("overall", {}).get("event_recall@delta"),
                    "false_positive_rate": s.get("overall", {}).get("false_positive_rate"),
                    "absolute_hallucination_rate": s.get("overall", {}).get("absolute_hallucination_rate"),
                    "off_timing_mention_rate": s.get("overall", {}).get("off_timing_mention_rate"),
                    "faithfulness_precision": s.get("overall", {}).get("faithfulness_precision"),
                    "temporal_precision": s.get("overall", {}).get("temporal_precision"),
                    "temporal_f1": s.get("overall", {}).get("temporal_f1"),
                    "macro_avg_event_recall@delta": s.get("macro_avg", {}).get("event_recall@delta"),
                    "macro_avg_false_positive_rate": s.get("macro_avg", {}).get("false_positive_rate"),
                    "macro_avg_absolute_hallucination_rate": s.get("macro_avg", {}).get("absolute_hallucination_rate"),
                    "macro_avg_off_timing_mention_rate": s.get("macro_avg", {}).get("off_timing_mention_rate"),
                    "macro_avg_faithfulness_precision": s.get("macro_avg", {}).get("faithfulness_precision"),
                    "macro_avg_temporal_precision": s.get("macro_avg", {}).get("temporal_precision"),
                    "macro_avg_temporal_f1": s.get("macro_avg", {}).get("temporal_f1"),
                }
                for s in summaries
            ]
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "summary_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
