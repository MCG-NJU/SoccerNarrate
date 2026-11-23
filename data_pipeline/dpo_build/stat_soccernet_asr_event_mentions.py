#!/usr/bin/env python3
"""
Measure whether SoccerNet ASR commentary can reveal that a key event happened
using a Qwen3-235B judge.

Inputs (defaults match local paths; override via CLI):
    - ASR:   /path/to/SoccerNet/asr/annos/{train,valid,test}/.../*processed*.json
    - Label: /path/to/SoccerNet/soccernet-label/annos/raw/.../Labels-v2.json

Workflow:
    1) Iterate ASR files and map each to its corresponding Labels-v2.json.
    2) Only evaluate label annotations with visibility="visible" by default.
    3) For each label event at time t (seconds within the half), extract ASR text in a window around t.
       - Each ASR segment has (start, end, text). We assign timestamps to words uniformly within the segment.
    4) Send (event label + window text) to a local Qwen3-235B service (OpenAI ChatCompletions style).
       - Expected: {"happened": "happened|not_happened", "evidence": "..."}
    5) Aggregate "happened" rates per event label (for each window length), and write jsonl + summary.json.

Example:
    export QWEN_API_KEY="xxx"
    python data_pipeline/dpo_build/stat_soccernet_asr_event_mentions.py \
        --api-url http://localhost:8000/v1/chat/completions \
        --model Qwen3-235B-A22B-Instruct-2507 \
        --window-seconds 15 \
        --num-workers 64 \
        --out-jsonl /tmp/event_mentions.jsonl \
        --out-summary /tmp/event_mentions_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from threading import local
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


ASR_ROOT_DEFAULT = Path("/path/to/SoccerNet/asr/annos")
LABEL_ROOT_DEFAULT = Path("/path/to/SoccerNet/soccernet-label/annos/raw")
PRE_EVENT_SECONDS = 5.0

_REQUESTS_LOCAL = local()


SYSTEM_PROMPT = """
You are a careful soccer (football) commentary analysis assistant.

You will be given a short window of live match commentary (ASR) and a SoccerNet event label.
Judge the event based only on what is present in the ASR text window.

Task:
Decide whether a viewer can directly tell from this ASR window, or with a small amount of simple reasoning,
that the event happened during this window.

Be conservative: if the text is ambiguous, output the negative class.

Return JSON only and follow the requested schema exactly.
"""

USER_PROMPT = """
Below is a window of live soccer commentary (ASR):
{{ASR_TEXT}}

Question:
Can a viewer directly tell from this ASR commentary, or with a small amount of simple reasoning,
that the key event "{{EVENT_LABEL}}" happened during this time window?

Return strict JSON only:
{"happened":"happened|not_happened","evidence":"<short quote or rationale>"}
"""

@dataclass(frozen=True)
class EventRef:
    split: str
    league_season: str
    match: str
    half: int
    label_path: str
    event_label: str
    position_sec: float
    game_time: str
    visibility: str = ""

    @property
    def key(self) -> str:
        raw = f"{self.label_path}|{self.game_time}|{self.event_label}|{self.position_sec:.3f}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()


@dataclass
class JudgeResult:
    happened: str  # "happened" | "not_happened"
    evidence: str
    raw_response: str


def parse_league_season(league_season: str) -> Tuple[str, str]:
    if "_" not in league_season:
        return league_season, ""
    league, season = league_season.rsplit("_", 1)
    return league, season


def get_half_from_filename(filename: str) -> Optional[int]:
    # 1-224p_processed_clean.json / 2-224p_processed_clean.json
    head = filename.split("-", 1)[0].strip()
    if head.isdigit():
        h = int(head)
        if h in (1, 2):
            return h
    return None


def get_label_path(asr_path: Path, asr_root: Path, label_root: Path) -> Path:
    """
    ASR:   .../asr/annos/{split}/{league_season}/{match}/{file}
    Label: .../soccernet-label/annos/raw/{league}/{season}/{match}/Labels-v2.json
    """
    rel = asr_path.relative_to(asr_root)
    parts = rel.parts
    if len(parts) < 3:
        raise ValueError(f"ASR path structure unexpected: {asr_path}")
    league_season, match_dir = parts[1], parts[2]
    league, season = parse_league_season(league_season)
    return label_root / league / season / match_dir / "Labels-v2.json"


def parse_game_time_half(game_time: str) -> Optional[int]:
    # "1 - 13:10" -> 1
    if not game_time:
        return None
    if "-" not in game_time:
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


@lru_cache(maxsize=4096)
def load_visibility_index(label_path: str) -> Dict[Tuple[str, str, int], str]:
    """
    Build a lookup: (gameTime, label, position_ms) -> normalized visibility string.
    """
    try:
        data = json.loads(Path(label_path).read_text(encoding="utf-8"))
    except Exception:
        return {}

    index: Dict[Tuple[str, str, int], str] = {}
    for ann in data.get("annotations", []) or []:
        label = (ann.get("label") or "").strip()
        game_time = (ann.get("gameTime") or "").strip()
        if not label or not game_time:
            continue
        pos = ann.get("position", None)
        try:
            position_ms = int(float(pos)) if pos is not None else None
        except Exception:
            position_ms = None
        if position_ms is None:
            continue
        vis_norm = normalize_visibility(ann.get("visibility"))
        if vis_norm is None:
            continue
        index[(game_time, label, position_ms)] = vis_norm
    return index


def row_is_visible(row: Dict[str, Any]) -> bool:
    """
    Determine whether an event row is visible.
    If visibility is missing or cannot be resolved, default to visible (do not drop data).
    """
    event = row.get("event") if isinstance(row.get("event"), dict) else {}
    direct = is_visible_visibility(event.get("visibility"))
    if direct is not None:
        return direct

    label_path = event.get("label_path")
    game_time = event.get("game_time")
    event_label = event.get("event_label") or row.get("event_label")
    position_sec = event.get("position_sec")
    if not label_path or not game_time or not event_label or not isinstance(position_sec, (int, float)):
        return True

    position_ms = int(round(float(position_sec) * 1000.0))
    vis_norm = load_visibility_index(str(label_path)).get((str(game_time).strip(), str(event_label).strip(), position_ms))
    resolved = is_visible_visibility(vis_norm)
    if resolved is None:
        return True
    return resolved


def load_label_events(
    label_file: Path,
    *,
    half: int,
    split: str,
    league_season: str,
    match: str,
    visible_only: bool,
) -> List[EventRef]:
    data = json.loads(label_file.read_text(encoding="utf-8"))
    out: List[EventRef] = []
    for ann in data.get("annotations", []):
        event_label = (ann.get("label") or "").strip()
        if not event_label:
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
        out.append(
            EventRef(
                split=split,
                league_season=league_season,
                match=match,
                half=half,
                label_path=str(label_file),
                event_label=event_label,
                position_sec=position_sec,
                game_time=game_time,
                visibility=visibility,
            )
        )
    return out


def load_asr_segments(
    json_path: Path,
    *,
    text_field: str = "llm_polished_text",
    fallback_field: str = "original_text",
    only_llm_keep: bool = True,
) -> List[Tuple[float, float, str]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments: List[Tuple[float, float, str]] = []
    for seg in data.get("segments", []):
        if only_llm_keep and seg.get("llm_keep") is False:
            continue
        raw_text = (seg.get(text_field) or "").strip()
        if not raw_text:
            raw_text = (seg.get(fallback_field) or "").strip()
        if not raw_text:
            continue
        try:
            st = float(seg.get("start", 0))
            ed = float(seg.get("end", st))
        except Exception:
            continue
        segments.append((st, ed, raw_text))
    return segments


def build_word_timeline(segments: Sequence[Tuple[float, float, str]]) -> Tuple[List[float], List[str]]:
    """
    将 [(st, ed, text)] 展开成 (times, words)：
    - 每段内把词平均分配时间戳（词的时间戳取段内均匀分布的“中心点”）
    """
    times: List[float] = []
    words: List[str] = []
    for st, ed, text in segments:
        toks = text.split()
        if not toks:
            continue
        dur = max(ed - st, 0.0)
        if dur <= 0:
            for w in toks:
                times.append(st)
                words.append(w)
            continue
        dt = dur / len(toks)
        for i, w in enumerate(toks):
            times.append(st + dt * (i + 0.5))
            words.append(w)
    return times, words


def collect_window_words(times: Sequence[float], words: Sequence[str], *, center: float, pre: float, post: float) -> str:
    start_t = center - pre
    end_t = center + post
    i0 = bisect_left(times, start_t)
    i1 = bisect_right(times, end_t)
    return " ".join(words[i0:i1]).strip()


def split_window_seconds(window_seconds: int) -> Tuple[float, float]:
    """
    SoccerNet 时间点 t 的窗口定义：
        [t - PRE_EVENT_SECONDS, t + (window_seconds - PRE_EVENT_SECONDS)]
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")
    if window_seconds <= PRE_EVENT_SECONDS:
        raise ValueError(f"window_seconds must be > PRE_EVENT_SECONDS ({PRE_EVENT_SECONDS})")
    pre = float(PRE_EVENT_SECONDS)
    post = float(window_seconds) - pre
    return pre, post


def make_sample_key(*, event_key: str, window_seconds: int, pre_seconds: float, post_seconds: float) -> str:
    raw = (
        f"{event_key}|"
        f"window_seconds={int(window_seconds)}|"
        f"pre_seconds={float(pre_seconds):.3f}|"
        f"post_seconds={float(post_seconds):.3f}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    # 1) direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2) try to find the first {...}
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
        # fallback: urllib
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as f:
            return json.loads(f.read().decode("utf-8"))


def call_qwen3_judge(
    *,
    api_url: str,
    api_key: Optional[str],
    model: str,
    event_label: str,
    window_text: str,
    max_retries: int = 3,
    timeout_s: float = 60.0,
) -> JudgeResult:
    prompt = USER_PROMPT.replace("{{ASR_TEXT}}", window_text).replace("{{EVENT_LABEL}}", event_label)
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
            return JudgeResult(happened=happened, evidence=evidence, raw_response=str(content))
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Failed to call judge: {last_err}")


def iter_asr_files(asr_root: Path, splits: Sequence[str], patterns: Sequence[str]) -> Iterable[Path]:
    for split in splits:
        base = asr_root / split
        if not base.exists():
            continue
        matched = False
        for pat in patterns:
            for p in sorted(base.rglob(pat)):
                matched = True
                yield p
        if not matched:
            # fallback: scan all json, but avoid huge accidental traversal
            for p in sorted(base.rglob("*.json")):
                if "processed" in p.name:
                    yield p


def read_existing_keys(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    读取已有 jsonl 结果，返回 {sample_key: row}
    """
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
            sample_key = row.get("sample_key")
            if isinstance(sample_key, str) and sample_key:
                existing[sample_key] = row

            event_key = row.get("event_key")
            window = row.get("window") if isinstance(row.get("window"), dict) else {}
            pre = window.get("pre_seconds")
            post = window.get("post_seconds")
            w = window.get("window_seconds")
            try:
                pre_f = float(pre) if isinstance(pre, (int, float)) else None
                post_f = float(post) if isinstance(post, (int, float)) else None
                w_i = int(round(float(w))) if isinstance(w, (int, float)) else None
            except Exception:
                pre_f, post_f, w_i = None, None, None

            if w_i is None and pre_f is not None and post_f is not None:
                w_i = int(round(pre_f + post_f))
            if w_i is not None:
                if pre_f is None and post_f is not None:
                    pre_f = float(w_i) - float(post_f)
                if post_f is None and pre_f is not None:
                    post_f = float(w_i) - float(pre_f)
            if w_i is not None and pre_f is None and post_f is None and w_i > PRE_EVENT_SECONDS:
                pre_f = float(PRE_EVENT_SECONDS)
                post_f = float(w_i) - float(PRE_EVENT_SECONDS)

            normalized_key: Optional[str] = None
            if isinstance(event_key, str) and event_key and w_i is not None and pre_f is not None and post_f is not None:
                normalized_key = make_sample_key(
                    event_key=event_key,
                    window_seconds=w_i,
                    pre_seconds=pre_f,
                    post_seconds=post_f,
                )
            if normalized_key:
                existing[normalized_key] = row
    return existing


def append_jsonl(path: Path, row: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="统计 ASR 是否能让观众判断 SoccerNet 事件发生率（Qwen3 judge）")
    parser.add_argument("--asr-root", type=Path, default=ASR_ROOT_DEFAULT)
    parser.add_argument("--label-root", type=Path, default=LABEL_ROOT_DEFAULT)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--asr-glob", type=str, nargs="+", default=["*processed_clean.json", "*processed.json"])
    parser.add_argument(
        "--window-seconds",
        type=int,
        nargs="+",
        default=[20, 30, 40, 50, 60, 70],
        help="网格搜索窗口长度 W（秒），窗口为 [t-5s, t+(W-5)s]，默认 20 30 40 50 60 70",
    )
    parser.add_argument("--asr-text-field", type=str, default="llm_polished_text")
    parser.add_argument("--asr-fallback-field", type=str, default="original_text")
    parser.add_argument("--only-llm-keep", dest="only_llm_keep", action="store_true", help="仅使用 llm_keep=True 的 segments（默认）")
    parser.add_argument("--include-not-llm-keep", dest="only_llm_keep", action="store_false", help="包含 llm_keep=False 的 segments")
    parser.set_defaults(only_llm_keep=True)
    parser.add_argument("--visible-only", dest="visible_only", action="store_true", help="只评估 visibility=visible 的事件（默认）")
    parser.add_argument("--include-not-visible", dest="visible_only", action="store_false", help="包含 visibility!=visible 的事件")
    parser.set_defaults(visible_only=True)

    parser.add_argument("--api-url", type=str, default=os.environ.get("QWEN_API_URL", "http://localhost:8000/v1/chat/completions"))
    parser.add_argument("--model", type=str, default=os.environ.get("QWEN_MODEL", "Qwen3-235B-A22B-Instruct-2507"))
    parser.add_argument("--api-key", type=str, default=os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    default_num_workers = int(os.environ.get("QWEN_NUM_WORKERS") or os.environ.get("QWEN_CONCURRENCY") or "64")
    parser.add_argument("--num-workers", "--concurrency", dest="num_workers", type=int, default=default_num_workers, help="并发请求线程数（>1 可显著提升吞吐）")
    parser.add_argument("--request-timeout-s", type=float, default=60.0, help="单次请求超时（秒）")
    parser.add_argument("--max-retries", type=int, default=3, help="单次请求失败重试次数")

    parser.add_argument("--out-jsonl", type=Path, default=Path("text_cls/stat/event_mentions.jsonl"))
    parser.add_argument("--out-summary", type=Path, default=Path("text_cls/stat/event_mentions_summary.json"))
    parser.add_argument("--max-files", type=int, default=0, help="最多处理多少个 ASR 文件（0=不限制）")
    parser.add_argument("--max-events", type=int, default=0, help="最多处理多少个样本(event-window)（0=不限制）")
    parser.add_argument("--dry-run", action="store_true", help="不请求模型，只提取窗口并统计样本数")
    parser.add_argument("--no-progress", action="store_true", help="禁用进度条（tqdm）")
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if not args.window_seconds:
        raise ValueError("--window-seconds must be non-empty")
    for w in args.window_seconds:
        if w <= PRE_EVENT_SECONDS:
            raise ValueError(f"--window-seconds must be > {PRE_EVENT_SECONDS}: got {w}")

    existing = read_existing_keys(args.out_jsonl)
    processed_keys = set(existing.keys())

    target_windows = {int(w) for w in (args.window_seconds or [])}
    totals: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    happened_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def window_seconds_from_row(row: Dict[str, Any]) -> Optional[int]:
        window = row.get("window") if isinstance(row.get("window"), dict) else {}
        w = window.get("window_seconds")
        if isinstance(w, (int, float)):
            return int(round(float(w)))
        pre = window.get("pre_seconds")
        post = window.get("post_seconds")
        if isinstance(pre, (int, float)) and isinstance(post, (int, float)):
            return int(round(float(pre) + float(post)))
        return None

    def consume_row(row: Dict[str, Any]):
        w = window_seconds_from_row(row)
        if w is None or (target_windows and w not in target_windows):
            return
        if args.visible_only and not row_is_visible(row):
            return
        label = row.get("event_label", "")
        if not isinstance(label, str) or not label:
            return
        totals[w][label] += 1
        judge = row.get("judge") or {}
        if isinstance(judge, dict):
            if canonicalize_happened(judge.get("happened")) == "happened":
                happened_counts[w][label] += 1

    seen_existing_row_ids: set[int] = set()
    for row in existing.values():
        rid = id(row)
        if rid in seen_existing_row_ids:
            continue
        seen_existing_row_ids.add(rid)
        consume_row(row)

    asr_files = list(iter_asr_files(args.asr_root, args.splits, args.asr_glob))
    if args.max_files and args.max_files > 0:
        asr_files = asr_files[: args.max_files]

    use_progress = (tqdm is not None) and (not args.no_progress)

    def count_pending_samples() -> int:
        if not use_progress:
            return 0
        total_pending = 0
        scan_iter = tqdm(asr_files, desc="scan", unit="file")
        try:
            for asr_path in scan_iter:
                if args.max_events and args.max_events > 0 and total_pending >= args.max_events:
                    break
                half = get_half_from_filename(asr_path.name)
                if half is None:
                    continue
                label_path = get_label_path(asr_path, args.asr_root, args.label_root)
                if not label_path.exists():
                    continue

                segments = load_asr_segments(
                    asr_path,
                    text_field=args.asr_text_field,
                    fallback_field=args.asr_fallback_field,
                    only_llm_keep=args.only_llm_keep,
                )
                if not segments:
                    continue
                times, words = build_word_timeline(segments)
                if not times:
                    continue

                rel = asr_path.relative_to(args.asr_root)
                split, league_season, match = rel.parts[0], rel.parts[1], rel.parts[2]
                events = load_label_events(
                    label_path,
                    half=half,
                    split=split,
                    league_season=league_season,
                    match=match,
                    visible_only=args.visible_only,
                )
                if not events:
                    continue

                for ev in events:
                    if args.max_events and args.max_events > 0 and total_pending >= args.max_events:
                        break
                    for window_seconds in args.window_seconds:
                        if args.max_events and args.max_events > 0 and total_pending >= args.max_events:
                            break
                        pre_seconds, post_seconds = split_window_seconds(window_seconds)
                        sample_key = make_sample_key(
                            event_key=ev.key,
                            window_seconds=window_seconds,
                            pre_seconds=pre_seconds,
                            post_seconds=post_seconds,
                        )
                        if sample_key in processed_keys:
                            continue
                        window_text = collect_window_words(
                            times,
                            words,
                            center=ev.position_sec,
                            pre=pre_seconds,
                            post=post_seconds,
                        )
                        if not window_text:
                            continue
                        total_pending += 1

                scan_iter.set_postfix_str(f"pending={total_pending}")
        finally:
            scan_iter.close()
        return total_pending

    total_pending = count_pending_samples() if use_progress else 0
    pbar = tqdm(total=total_pending, desc="judge", unit="sample") if use_progress else None

    submitted_samples = 0
    in_progress_keys: set[str] = set()
    in_flight: Dict[Future, Dict[str, Any]] = {}

    def drain_futures(*, blocking: bool) -> None:
        if not in_flight:
            return
        if blocking:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
        else:
            done = {f for f in in_flight.keys() if f.done()}

        for fut in done:
            row = in_flight.pop(fut)
            sample_key = str(row.get("sample_key", ""))
            in_progress_keys.discard(sample_key)
            try:
                judge = fut.result()
            except Exception as e:
                print(f"[judge_error] sample_key={sample_key} err={e}", file=sys.stderr, flush=True)
                if pbar is not None:
                    pbar.update(1)
                continue
            row["judge"] = asdict(judge)
            append_jsonl(args.out_jsonl, row)
            processed_keys.add(sample_key)
            consume_row(row)
            if pbar is not None:
                pbar.update(1)

    should_stop = False
    executor: Optional[ThreadPoolExecutor] = None
    if not args.dry_run and args.num_workers > 1:
        executor = ThreadPoolExecutor(max_workers=args.num_workers)

    try:
        for idx, asr_path in enumerate(asr_files, start=1):
            if should_stop:
                break
            half = get_half_from_filename(asr_path.name)
            if half is None:
                continue
            rel = asr_path.relative_to(args.asr_root)
            split, league_season, match = rel.parts[0], rel.parts[1], rel.parts[2]
            label_path = get_label_path(asr_path, args.asr_root, args.label_root)
            if not label_path.exists():
                continue

            segments = load_asr_segments(
                asr_path,
                text_field=args.asr_text_field,
                fallback_field=args.asr_fallback_field,
                only_llm_keep=args.only_llm_keep,
            )
            if not segments:
                continue
            times, words = build_word_timeline(segments)
            if not times:
                continue

            events = load_label_events(
                label_path,
                half=half,
                split=split,
                league_season=league_season,
                match=match,
                visible_only=args.visible_only,
            )
            if not events:
                continue

            if pbar is None:
                print(
                    f"[{idx}/{len(asr_files)}] {split} | {league_season} | {match} | half={half} | events={len(events)}",
                    flush=True,
                )
            else:
                pbar.set_postfix_str(f"{split}|{league_season}|{match}|h{half}|events={len(events)}")
            for ev in events:
                if args.max_events and args.max_events > 0 and submitted_samples >= args.max_events:
                    should_stop = True
                    break
                for window_seconds in args.window_seconds:
                    if args.max_events and args.max_events > 0 and submitted_samples >= args.max_events:
                        should_stop = True
                        break
                    pre_seconds, post_seconds = split_window_seconds(window_seconds)
                    sample_key = make_sample_key(
                        event_key=ev.key,
                        window_seconds=window_seconds,
                        pre_seconds=pre_seconds,
                        post_seconds=post_seconds,
                    )
                    if sample_key in processed_keys or sample_key in in_progress_keys:
                        continue

                    window_text = collect_window_words(
                        times,
                        words,
                        center=ev.position_sec,
                        pre=pre_seconds,
                        post=post_seconds,
                    )
                    if not window_text:
                        continue

                    row = {
                        "sample_key": sample_key,
                        "event_key": ev.key,
                        "event": asdict(ev),
                        "event_label": ev.event_label,
                        "window": {
                            "window_seconds": window_seconds,
                            "pre_seconds": pre_seconds,
                            "post_seconds": post_seconds,
                            "text": window_text,
                        },
                    }

                    if args.dry_run:
                        row["judge"] = {"happened": "not_happened", "evidence": "", "raw_response": ""}
                        append_jsonl(args.out_jsonl, row)
                        processed_keys.add(sample_key)
                        consume_row(row)
                        submitted_samples += 1
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    if executor is None:
                        submitted_samples += 1
                        try:
                            judge = call_qwen3_judge(
                                api_url=args.api_url,
                                api_key=args.api_key,
                                model=args.model,
                                event_label=ev.event_label,
                                window_text=window_text,
                                max_retries=args.max_retries,
                                timeout_s=args.request_timeout_s,
                            )
                        except Exception as e:
                            print(f"[judge_error] sample_key={sample_key} err={e}", file=sys.stderr, flush=True)
                            if pbar is not None:
                                pbar.update(1)
                            continue
                        row["judge"] = asdict(judge)
                        append_jsonl(args.out_jsonl, row)
                        processed_keys.add(sample_key)
                        consume_row(row)
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    in_progress_keys.add(sample_key)
                    fut = executor.submit(
                        call_qwen3_judge,
                        api_url=args.api_url,
                        api_key=args.api_key,
                        model=args.model,
                        event_label=ev.event_label,
                        window_text=window_text,
                        max_retries=args.max_retries,
                        timeout_s=args.request_timeout_s,
                    )
                    in_flight[fut] = row
                    submitted_samples += 1

                    while len(in_flight) >= args.num_workers:
                        drain_futures(blocking=True)

            drain_futures(blocking=False)

        while in_flight:
            drain_futures(blocking=True)
    finally:
        if pbar is not None:
            pbar.close()
        if executor is not None:
            executor.shutdown(wait=True)

    summary: Dict[str, Any] = {
        "config": {
            "asr_root": str(args.asr_root),
            "label_root": str(args.label_root),
            "splits": list(args.splits),
            "asr_glob": list(args.asr_glob),
            "window_seconds": list(args.window_seconds),
            "pre_event_seconds": PRE_EVENT_SECONDS,
            "asr_text_field": args.asr_text_field,
            "only_llm_keep": args.only_llm_keep,
            "visible_only": args.visible_only,
            "api_url": args.api_url,
            "model": args.model,
            "dry_run": args.dry_run,
            "num_workers": args.num_workers,
            "request_timeout_s": args.request_timeout_s,
            "max_retries": args.max_retries,
        },
        "primary_window_seconds": None,
        "overall": {},
        "per_label": {},
        "per_window_seconds": {},
    }

    best_window: Optional[int] = None
    best_h_rate: float = -1.0

    for w in sorted(totals.keys()):
        per_label: Dict[str, Any] = {}
        total_all = sum(totals[w].values())
        h_all = sum(happened_counts[w].values())

        for label in sorted(totals[w].keys()):
            total = totals[w][label]
            h = happened_counts[w].get(label, 0)
            per_label[label] = {
                "total": total,
                "happened": h,
                "happened_rate": (h / total) if total else 0.0,
            }

        overall = {
            "total": total_all,
            "happened": h_all,
            "happened_rate": (h_all / total_all) if total_all else 0.0,
        }
        summary["per_window_seconds"][str(w)] = {"overall": overall, "per_label": per_label}

        if total_all > 0 and overall["happened_rate"] >= best_h_rate:
            best_h_rate = float(overall["happened_rate"])
            best_window = w

    if best_window is None and args.window_seconds:
        best_window = int(args.window_seconds[0])
    summary["primary_window_seconds"] = best_window
    if best_window is not None:
        best_block = summary["per_window_seconds"].get(str(best_window), {}) or {}
        summary["overall"] = best_block.get("overall", {}) or {}
        summary["per_label"] = best_block.get("per_label", {}) or {}

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    done_samples = sum(sum(d.values()) for d in totals.values())
    done_labels = len({label for w in totals for label in totals[w].keys()})
    print(f"done: samples={done_samples}, labels={done_labels}, windows={len(totals)}")
    print(f"jsonl: {args.out_jsonl}")
    print(f"summary: {args.out_summary}")


if __name__ == "__main__":
    main()
