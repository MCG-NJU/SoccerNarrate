#!/usr/bin/env python3
"""
Compute Event Recall@δ and False Positive Rate for human (SoccerNet-ASR) commentary,
using the same sliding-window + LLM-as-a-judge protocol as `soccernet_vtt_event_mentions.py`.

Human commentary source:
  - SoccerNet-ASR annotation JSON files (NOT VTT)
  - Default: /path/to/SoccerNet/asr/annos/test
  - Expected layout:
      annos_root/<league>_<season>/<match_dir>/<half>-224p_processed_clean.json

This script:
  1) Loads ASR segments (start/end/text) from the JSON files.
  2) Slides windows across the half (default window=30s, stride=15s).
  3) If a window contains >=1 GT key event, asks the binary judge prompt per event label.
     If a window contains no GT key event, uses a negative-window protocol (default: binary entailment for a small label set).
  4) Computes:
     - Event Recall@δ (match GT event time to predicted window within ±δ seconds)
     - False Positive Rate (negative windows that mention an unmatched event)
     - Absolute Hallucination Rate (AHR) vs Off-timing Mention Rate (OMR) split for negative-window false positives

Judge API:
  - OpenAI ChatCompletions compatible endpoint.
  - Env: QWEN_API_URL, QWEN_API_KEY, QWEN_MODEL
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
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


shared = _import_shared()

SCHEMA_VERSION = shared.SCHEMA_VERSION
DEFAULT_LABEL_ROOT = shared.DEFAULT_LABEL_ROOT


_TRAILING_ELLIPSIS_RE = re.compile(r"(?:\s*(?:\.\.\.|…)+)\s*$")
_WS_RE = re.compile(r"\s+")


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


def iter_asr_anno_files(annos_root: Path, *, globs: Sequence[str], max_files: Optional[int]) -> List[Path]:
    files: List[Path] = []
    for g in globs:
        files.extend(sorted(annos_root.rglob(g)))
    files = sorted(set(files))
    if max_files is not None and max_files > 0:
        files = files[: int(max_files)]
    return files


def load_commentary_segments_from_asr_json(
    json_path: Path,
    *,
    text_field: str,
    fallback_field: str,
    only_llm_keep: bool,
) -> List[Tuple[float, float, str]]:
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segments, list):
        return []

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

        text = seg.get(text_field)
        if not isinstance(text, str) or not text.strip():
            text = seg.get(fallback_field)
        cleaned = _clean_text(text)
        if not cleaned:
            continue

        out.append((start_f, end_f, cleaned))

    out.sort(key=lambda x: (x[0], x[1]))
    return out


def count_total_and_done_human(
    *,
    annos_files: Sequence[Path],
    annos_root: Path,
    labels_root: Path,
    processed_keys: Sequence[str],
    window_seconds: float,
    stride_seconds: float,
    visible_only: bool,
    include_background: bool,
    include_unknown_labels: bool,
    text_field: str,
    fallback_field: str,
    only_llm_keep: bool,
) -> Tuple[int, int]:
    processed = set(processed_keys)
    total = 0
    done = 0
    for anno_file in annos_files:
        half = shared.half_from_filename(anno_file.name)
        if half is None:
            continue
        label_path = shared.label_file_for_infer(anno_file, annos_root, labels_root)
        if not label_path.exists():
            continue
        league_season, match_dir = shared.infer_meta_from_path(anno_file, annos_root)
        events = shared.key_events_only(
            shared.load_label_events(
                label_path,
                half=half,
                league_season=league_season,
                match=match_dir,
                visible_only=visible_only,
                include_background=include_background,
                include_unknown_labels=include_unknown_labels,
            )
        )
        segments = load_commentary_segments_from_asr_json(
            anno_file,
            text_field=text_field,
            fallback_field=fallback_field,
            only_llm_keep=only_llm_keep,
        )
        if not events and not segments:
            continue

        max_seg_end = max((end for _, end, _ in segments), default=0.0)
        max_event_t = max((ev.position_sec for ev in events), default=0.0)
        max_time = max(float(max_seg_end), float(max_event_t))

        for _window_idx, start_sec, end_sec in shared.iter_sliding_windows(
            max_time_sec=max_time,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        ):
            window_key = shared.make_window_key(infer_file=str(anno_file), start_sec=start_sec, end_sec=end_sec)
            win_events = shared.events_in_interval(events, start=start_sec, end=end_sec)
            if win_events:
                for ev in win_events:
                    sample_key = shared.make_sample_key_event(window_key=window_key, event_key=ev.key)
                    total += 1
                    if sample_key in processed:
                        done += 1
            else:
                sample_key = shared.make_sample_key_open(window_key=window_key)
                total += 1
                if sample_key in processed:
                    done += 1
    return total, done


def evaluate_human_asr(
    *,
    annos_root: Path,
    labels_root: Path,
    out_dir: Path,
    run_name: str,
    text_field: str,
    fallback_field: str,
    only_llm_keep: bool,
    globs: Sequence[str],
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
    neg_window_protocol: str,
    num_workers: int,
    request_timeout_s: float,
    max_retries: int,
    max_files: Optional[int],
    dry_run: bool,
):
    annos_files = iter_asr_anno_files(annos_root, globs=globs, max_files=max_files)
    if not annos_files:
        print(f"[skip] no asr annotation json files under: {annos_root}")
        return None

    run_out_dir = out_dir / run_name
    out_jsonl = run_out_dir / "human_sliding_window_event_mentions.jsonl"
    out_summary = run_out_dir / "human_sliding_window_event_mentions_summary.json"

    existing = shared.read_existing_jsonl(out_jsonl)
    processed_keys = set(existing.keys())

    total_samples, done_samples = count_total_and_done_human(
        annos_files=annos_files,
        annos_root=annos_root,
        labels_root=labels_root,
        processed_keys=sorted(processed_keys),
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        visible_only=visible_only,
        include_background=include_background,
        include_unknown_labels=include_unknown_labels,
        text_field=text_field,
        fallback_field=fallback_field,
        only_llm_keep=only_llm_keep,
    )

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=total_samples,
            initial=done_samples,
            desc=f"{run_name}",
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
        canon = shared.canonicalize_event_label(label)
        if canon in shared.KEY_EVENT_LABEL_SET:
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

        events = shared.key_events_only(
            shared.load_label_events(
                path,
                half=int(half),
                league_season=str(league_season),
                match=str(match_dir),
                visible_only=visible_only,
                include_background=include_background,
                include_unknown_labels=include_unknown_labels,
            )
        )
        built = shared.build_events_by_label(events)
        events_by_label_cache[key] = built
        return built

    def annotate_otm_offsets(row: Dict[str, Any]):
        if row.get("task") != "open":
            return
        judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
        predicted = shared.predicted_labels_from_open_judge(judge)
        if not predicted:
            return
        window = row.get("window") if isinstance(row.get("window"), dict) else {}
        start = window.get("start_sec")
        end = window.get("end_sec")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return
        offsets = shared.compute_otm_offsets_for_window(
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
        shared.append_jsonl(out_jsonl, row)
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
                print(f"[judge_error] run={run_name} sample_key={sample_key} err={e}", file=sys.stderr, flush=True)
                if pbar is not None:
                    pbar.update(1)
                continue
            row["judge"] = asdict(judge)
            annotate_otm_offsets(row)
            consume_and_write_row(row)

    try:
        for anno_file in annos_files:
            half = shared.half_from_filename(anno_file.name)
            if half is None:
                continue

            label_path = shared.label_file_for_infer(anno_file, annos_root, labels_root)
            if not label_path.exists():
                continue

            league_season, match_dir = shared.infer_meta_from_path(anno_file, annos_root)
            events = shared.key_events_only(
                shared.load_label_events(
                    label_path,
                    half=half,
                    league_season=league_season,
                    match=match_dir,
                    visible_only=visible_only,
                    include_background=include_background,
                    include_unknown_labels=include_unknown_labels,
                )
            )
            events_by_label_cache[(str(label_path), int(half))] = shared.build_events_by_label(events)
            segments = load_commentary_segments_from_asr_json(
                anno_file,
                text_field=text_field,
                fallback_field=fallback_field,
                only_llm_keep=only_llm_keep,
            )
            if not events and not segments:
                continue

            max_seg_end = max((end for _, end, _ in segments), default=0.0)
            max_event_t = max((ev.position_sec for ev in events), default=0.0)
            max_time = max(float(max_seg_end), float(max_event_t))

            for window_idx, start_sec, end_sec in shared.iter_sliding_windows(
                max_time_sec=max_time,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
            ):
                window_key = shared.make_window_key(infer_file=str(anno_file), start_sec=start_sec, end_sec=end_sec)
                win_events = shared.events_in_interval(events, start=start_sec, end=end_sec)

                window_text = ""
                if segments:
                    window_text = shared.collect_interval_text(segments, start_sec=start_sec, end_sec=end_sec)

                if win_events:
                    for ev in win_events:
                        sample_key = shared.make_sample_key_event(window_key=window_key, event_key=ev.key)
                        if sample_key in processed_keys or sample_key in in_progress_keys:
                            continue

                        row: Dict[str, Any] = {
                            "schema_version": SCHEMA_VERSION,
                            "sample_key": sample_key,
                            "task": "binary",
                            "source": "human_asr",
                            "run_name": run_name,
                            "infer_file": str(anno_file),
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
                            "text": {
                                "text_field": str(text_field),
                                "fallback_field": str(fallback_field),
                                "only_llm_keep": bool(only_llm_keep),
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
                                judge = shared.call_qwen3_judge_binary(
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
                                    f"[judge_error] run={run_name} sample_key={sample_key} err={e}",
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
                            shared.call_qwen3_judge_binary,
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
                    sample_key = shared.make_sample_key_open(window_key=window_key)
                    if sample_key in processed_keys or sample_key in in_progress_keys:
                        continue

                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "sample_key": sample_key,
                        "task": "open",
                        "source": "human_asr",
                        "run_name": run_name,
                        "infer_file": str(anno_file),
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
                        "text": {
                            "text_field": str(text_field),
                            "fallback_field": str(fallback_field),
                            "only_llm_keep": bool(only_llm_keep),
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
                            judge = shared.call_qwen3_judge_negative_window(
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
                                f"[judge_error] run={run_name} sample_key={sample_key} err={e}",
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
                        shared.call_qwen3_judge_negative_window,
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

    metrics = shared.compute_sliding_window_metrics(
        vtt_files=annos_files,
        infer_root=annos_root,
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
            "source": "human_asr",
            "annos_root": str(annos_root),
            "labels_root": str(labels_root),
            "run_name": str(run_name),
            "text_field": str(text_field),
            "fallback_field": str(fallback_field),
            "only_llm_keep": bool(only_llm_keep),
            "asr_globs": list(globs),
            "window_seconds": float(window_seconds),
            "stride_seconds": float(stride_seconds),
            "delta_seconds": float(delta_seconds),
            "review_grace_seconds": float(review_grace_seconds),
            "review_grace_labels": list(review_grace_labels),
            "visible_only": bool(visible_only),
            "include_background": bool(include_background),
            "include_unknown_labels": bool(include_unknown_labels),
            "neg_window_protocol": str(neg_window_protocol),
            "neg_window_binary_labels": list(shared.NEGATIVE_WINDOW_BINARY_LABELS),
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
        f"[done] run={run_name} "
        f"events={summary['overall']['total_events']} "
        f"recall@{float(delta_seconds):g}={summary['overall']['event_recall@delta']:.4f} "
        f"fpr={summary['overall']['false_positive_rate']:.4f} "
        f"ahr={summary['overall'].get('absolute_hallucination_rate', 0.0):.4f} "
        f"omr={summary['overall'].get('off_timing_mention_rate', 0.0):.4f} "
        f"p_faith={summary['overall'].get('faithfulness_precision', 0.0):.4f} "
        f"p_on={summary['overall'].get('temporal_precision', 0.0):.4f} "
        f"jsonl={out_jsonl} summary={out_summary}"
    )
    return summary


def main() -> None:
    default_annos_root = Path("/path/to/SoccerNet/asr/annos/test")
    abs_annos_root = Path("/path/to/SoccerNet/asr/annos/test")
    if not default_annos_root.exists() and abs_annos_root.exists():
        default_annos_root = abs_annos_root

    default_labels_root = "/path/to/SoccerNet/SoccerNet-label/raw_annos"

    parser = argparse.ArgumentParser(description="LLM-as-a-judge: human SoccerNet-ASR Event Recall@δ and FPR (sliding windows)")
    parser.add_argument(
        "--annos-root",
        type=Path,
        default=default_annos_root,
        help="SoccerNet-ASR human annotation root (default: work/datasets/SoccerNet/asr/annos/test)",
    )
    parser.add_argument("--labels-root", type=Path, default=default_labels_root, help="SoccerNet-label root (supports {root}/annos/raw)")
    parser.add_argument("--out-dir", type=Path, default=Path("eval/llm_as_a_judge/results_human"))
    parser.add_argument("--run-name", type=str, default="human_test")

    parser.add_argument("--asr-glob", type=str, nargs="+", default=["*processed_clean.json", "*processed.json"])
    parser.add_argument("--text-field", type=str, default="corrected_name_text")
    parser.add_argument("--fallback-field", type=str, default="original_text")
    parser.add_argument("--only-llm-keep", dest="only_llm_keep", action="store_true", help="Only keep segments with llm_keep=True (default)")
    parser.add_argument("--include-not-llm-keep", dest="only_llm_keep", action="store_false", help="Include llm_keep=False segments")
    parser.set_defaults(only_llm_keep=True)

    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--stride-seconds", type=float, default=15.0)
    parser.add_argument("--delta-seconds", type=float, default=15.0)
    parser.add_argument(
        "--review-grace-seconds",
        type=float,
        default=float(getattr(shared, "REVIEW_GRACE_SECONDS_DEFAULT", 120.0)),
        help="If a predicted label is mentioned within this many seconds after the last same-label GT event, do not count it as a false positive (default: 120).",
    )
    parser.add_argument(
        "--review-grace-labels",
        nargs="*",
        default=list(getattr(shared, "REVIEW_GRACE_LABELS_DEFAULT", ("Goal", "Foul", "Substitution"))),
        help='Labels to apply --review-grace-seconds to (default: Goal Foul Substitution).',
    )

    parser.add_argument("--all-events", dest="visible_only", action="store_false", help="Include non-visible events")
    parser.add_argument(
        "--visible-only",
        dest="visible_only",
        action="store_true",
        default=True,
        help="Only keep visibility=visible events (default)",
    )
    parser.add_argument("--include-background", default=True, help='Include background labels (e.g., "Background"/"Negative")')
    parser.add_argument("--include-unknown-labels", action="store_true", help="Include labels beyond the predefined key-event label set")
    parser.add_argument(
        "--neg-window-protocol",
        type=str,
        default=shared.DEFAULT_NEG_WINDOW_PROTOCOL,
        choices=list(shared.NEG_WINDOW_PROTOCOL_CHOICES),
        help='Negative-window judging protocol: "binary3" (binary entailment over configured labels) or "open" (legacy 17-way open-set).',
    )

    parser.add_argument("--num-workers", type=int, default=128, help="Parallel workers for judge calls")
    parser.add_argument("--api-url", type=str, default=os.environ.get("QWEN_API_URL", "http://localhost:8000/v1/chat/completions"))
    parser.add_argument("--api-key", type=str, default=os.environ.get("QWEN_API_KEY", None))
    parser.add_argument("--judge-model", type=str, default=os.environ.get("QWEN_MODEL", "Qwen3-235B-A22B-Instruct-2507"))
    parser.add_argument("--request-timeout-s", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)

    parser.add_argument("--max-files", type=int, default=None, help="Max JSON files to process (debug)")
    parser.add_argument("--dry-run", action="store_true", help="Do not call judge; write dummy results")
    args = parser.parse_args()

    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be > 0")
    if args.stride_seconds <= 0:
        raise ValueError("--stride-seconds must be > 0")
    if args.delta_seconds < 0:
        raise ValueError("--delta-seconds must be >= 0")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if not args.asr_glob:
        raise ValueError("--asr-glob must be non-empty")

    evaluate_human_asr(
        annos_root=args.annos_root,
        labels_root=args.labels_root,
        out_dir=args.out_dir,
        run_name=str(args.run_name),
        text_field=str(args.text_field),
        fallback_field=str(args.fallback_field),
        only_llm_keep=bool(args.only_llm_keep),
        globs=list(args.asr_glob),
        judge_api_url=str(args.api_url),
        judge_api_key=args.api_key,
        judge_model=str(args.judge_model),
        window_seconds=float(args.window_seconds),
        stride_seconds=float(args.stride_seconds),
        delta_seconds=float(args.delta_seconds),
        review_grace_seconds=float(args.review_grace_seconds),
        review_grace_labels=list(args.review_grace_labels),
        visible_only=bool(args.visible_only),
        include_background=bool(args.include_background),
        include_unknown_labels=bool(args.include_unknown_labels),
        neg_window_protocol=str(args.neg_window_protocol),
        num_workers=int(args.num_workers),
        request_timeout_s=float(args.request_timeout_s),
        max_retries=int(args.max_retries),
        max_files=args.max_files,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
