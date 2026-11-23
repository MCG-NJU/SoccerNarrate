#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-process streaming inference for SoccerNarrate-Eval (SoccerNet full matches).

This script loads the model once in a single process, iterates over match-half
videos sequentially, and writes timestamped WebVTT (.vtt) commentary files that
can be directly evaluated by `eval/llm_as_a_judge/soccernet_vtt_event_mentions.py`.

Expected video layout (--video_root):
  video_root/
    <league>/
      <season>/
        <match_name>/
          1_720p.mkv (or .mp4)
          2_720p.mkv (or .mp4)

Optional split directory (--annos_root, e.g., SoccerNet/asr/annos/test):
  annos_root/
    <league>_<season>/
      <match_name>/
        1-224p_processed_clean.json
        2-224p_processed_clean.json

Output layout (--output_root):
  output_root/
    <league>_<season>/
      <match_name>/
        1_720p.vtt
        2_720p.vtt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import sys
import traceback
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

_LEAGUE_SEASON_RE = re.compile(r"^(?P<league>.+)_(?P<season>\d{4}-\d{4})$")

SOCCER_COMMENTARY_QUERIES = [
    "Commentate on this match.",
    "Describe what's happening in this video.",
    "Provide live commentary for this clip.",
    "Give a live commentary of this play.",
    "What's going on in this match?",
    "Provide commentary for this moment.",
    "Give me a live commentary.",
    "Comment on this football moment.",
    "Provide match commentary.",
    "Commentate on this scene.",
]


@dataclass(frozen=True)
class VideoJob:
    rel_match_dir: Path
    league: str
    season: str
    match_name: str
    half: int
    video_path: Path
    output_vtt: Path


def _ensure_import_paths() -> None:
    this_file = Path(__file__).resolve()
    for p in (this_file.parents[2], this_file.parents[1]):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def _stable_int_hash(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _pick_query_for_job(query: Optional[str], query_seed: Optional[int], job: VideoJob) -> str:
    if query:
        return query
    if query_seed is None:
        return random.choice(SOCCER_COMMENTARY_QUERIES)
    job_key = f"{query_seed}|{job.league}|{job.season}|{job.match_name}|{job.half}"
    return random.Random(_stable_int_hash(job_key)).choice(SOCCER_COMMENTARY_QUERIES)


def _normalize_video_root(video_root: Path) -> Path:
    if (video_root / "videos").is_dir():
        return video_root / "videos"
    return video_root


def _find_half_video(match_video_dir: Path, half: int, exts: Sequence[str]) -> Optional[Path]:
    for ext in exts:
        candidate = match_video_dir / f"{half}_720p.{ext.lstrip('.')}"
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1024)
def _get_video_duration_seconds(video_path: str) -> Optional[float]:
    try:
        import decord

        vr = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(vr)
        if total_frames > 0:
            try:
                last_ts = vr.get_frame_timestamp(total_frames - 1)
                if hasattr(last_ts, "tolist"):
                    last_ts = last_ts.tolist()
                if isinstance(last_ts, (list, tuple)) and last_ts:
                    end_s = float(last_ts[1] if len(last_ts) >= 2 else last_ts[0])
                    if end_s > 0:
                        return end_s
            except Exception:
                pass
            fps = float(vr.get_avg_fps())
            if fps > 0:
                return float(total_frames) / fps
    except Exception:
        pass
    return None


def _effective_duration_seconds(
    *,
    requested_duration: float,
    actual_duration: Optional[float],
    chunk_duration: float,
    skip_first_chunk: int,
) -> float:
    effective = max(0.0, float(requested_duration))
    if actual_duration is None or actual_duration <= 0:
        return effective
    skip_offset_s = max(0.0, float(skip_first_chunk) * float(chunk_duration))
    remaining_s = max(0.0, float(actual_duration) - skip_offset_s)
    return min(effective, remaining_s)


def _iter_matches_from_annos(annos_root: Path) -> Iterable[tuple[str, str, str]]:
    for league_season_dir in sorted(p for p in annos_root.iterdir() if p.is_dir()):
        m = _LEAGUE_SEASON_RE.match(league_season_dir.name)
        if not m:
            continue
        league, season = m.group("league"), m.group("season")
        for match_dir in sorted(p for p in league_season_dir.iterdir() if p.is_dir()):
            yield league, season, match_dir.name


def _iter_matches_from_videos(video_root: Path) -> Iterable[tuple[str, str, str]]:
    for league_dir in sorted(p for p in video_root.iterdir() if p.is_dir()):
        for season_dir in sorted(p for p in league_dir.iterdir() if p.is_dir()):
            for match_dir in sorted(p for p in season_dir.iterdir() if p.is_dir()):
                yield league_dir.name, season_dir.name, match_dir.name


def build_jobs(
    *,
    video_root: Path,
    output_root: Path,
    video_exts: Sequence[str],
    annos_root: Optional[Path] = None,
    strict_missing_video: bool = False,
) -> list[VideoJob]:
    video_root = _normalize_video_root(video_root)
    matches = (
        _iter_matches_from_annos(annos_root)
        if annos_root is not None
        else _iter_matches_from_videos(video_root)
    )

    jobs: list[VideoJob] = []
    for league, season, match_name in matches:
        rel_match_dir = Path(f"{league}_{season}") / match_name
        match_video_dir = video_root / league / season / match_name
        out_match_dir = output_root / rel_match_dir

        for half in (1, 2):
            video_path = _find_half_video(match_video_dir, half, video_exts)
            if video_path is None:
                msg = f"Missing video: {match_video_dir}/{half}_720p.{{{','.join(video_exts)}}}"
                if strict_missing_video:
                    raise FileNotFoundError(msg)
                print(f"[skip] {msg}")
                continue

            jobs.append(
                VideoJob(
                    rel_match_dir=rel_match_dir,
                    league=league,
                    season=season,
                    match_name=match_name,
                    half=half,
                    video_path=video_path,
                    output_vtt=out_match_dir / f"{half}_720p.vtt",
                )
            )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-process streaming inference for SoccerNarrate-Eval."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint path or HuggingFace model ID")
    parser.add_argument("--model_base", type=str, choices=["Qwen2_5", "Qwen2"], default="Qwen2_5")
    parser.add_argument(
        "--video_root",
        type=Path,
        default=Path(os.environ.get("SOCCERNET_VIDEOS_ROOT", "/path/to/SoccerNet/videos")),
        help="SoccerNet videos root (<league>/<season>/<match>/{1,2}_720p.mkv)",
    )
    parser.add_argument(
        "--annos_root",
        type=Path,
        default=None,
        help="Optional split annotation root (e.g., SoccerNet/asr/annos/test) to restrict evaluated matches",
    )
    parser.add_argument("--output_root", type=Path, required=True, help="Output root for generated .vtt files")
    parser.add_argument("--video_exts", type=str, default="mkv,mp4", help="Comma-separated video extensions")
    parser.add_argument("--strict_missing_video", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .vtt outputs")
    parser.add_argument("--dry_run", action="store_true", help="List matched video jobs without running inference")
    parser.add_argument("--max_jobs", type=int, default=0, help="Limit number of video halves to process (0 = all)")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index for optional manual sharding")
    parser.add_argument("--num_shards", type=int, default=1, help="Total shard count (default: 1, single process)")

    # Streaming decoding parameters (matching SoccerNarrate paper defaults)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--chunk_duration", type=int, default=1)
    parser.add_argument("--text_round", type=int, default=16)
    parser.add_argument("--text_sink", type=int, default=512)
    parser.add_argument("--text_sliding_window", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--duration", type=float, default=6000.0)
    parser.add_argument("--skip_first_chunk", type=int, default=0)
    parser.add_argument("--pos_mode", type=str, default="shrink", choices=["append", "shrink"])
    parser.add_argument("--all_text", action="store_true", default=False)
    parser.add_argument("--recompute", action="store_true", default=False)
    parser.add_argument("--previous_text", type=str, default="")
    parser.add_argument("--query", type=str, default="", help="Fixed prompt (empty = sample per match half)")
    parser.add_argument("--query_seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    if not os.environ.get("FPS_MAX_FRAMES", "").strip():
        os.environ["FPS_MAX_FRAMES"] = "64"

    args = parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"Invalid shard config: shard_id={args.shard_id}, num_shards={args.num_shards}")
    if args.annos_root is not None and not args.annos_root.is_dir():
        raise FileNotFoundError(f"annos_root not found: {args.annos_root}")

    video_exts = [e.strip() for e in args.video_exts.split(",") if e.strip()]
    jobs = build_jobs(
        video_root=args.video_root,
        output_root=args.output_root,
        video_exts=video_exts,
        annos_root=args.annos_root,
        strict_missing_video=args.strict_missing_video,
    )
    jobs.sort(key=lambda j: (str(j.rel_match_dir), j.half))
    jobs = [j for idx, j in enumerate(jobs) if idx % args.num_shards == args.shard_id]
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]

    print(f"model_path:  {args.model_path}")
    print(f"video_root:  {_normalize_video_root(args.video_root)}")
    if args.annos_root is not None:
        print(f"annos_root:  {args.annos_root}")
    print(f"output_root: {args.output_root}")
    print(f"jobs:        {len(jobs)} (shard {args.shard_id}/{args.num_shards})")

    if args.dry_run:
        for j in jobs[:20]:
            print(f"[dry] {j.video_path} -> {j.output_vtt}")
        if len(jobs) > 20:
            print(f"[dry] ... ({len(jobs) - 20} more)")
        return 0

    _ensure_import_paths()
    from streaming_vlm.inference.inference import load_model_and_processor, streaming_inference

    model, processor = load_model_and_processor(args.model_path, args.model_base)

    completed = skipped = failed = 0
    for idx, j in enumerate(jobs, start=1):
        if j.output_vtt.exists() and not args.overwrite:
            skipped += 1
            if not args.quiet:
                print(f"[{idx}/{len(jobs)}] [skip] exists: {j.output_vtt}")
            continue

        j.output_vtt.parent.mkdir(parents=True, exist_ok=True)
        if not args.quiet:
            print(f"[{idx}/{len(jobs)}] [run]  {j.video_path} -> {j.output_vtt}")

        try:
            actual_duration_s = _get_video_duration_seconds(str(j.video_path))
            effective_duration_s = _effective_duration_seconds(
                requested_duration=args.duration,
                actual_duration=actual_duration_s,
                chunk_duration=args.chunk_duration,
                skip_first_chunk=args.skip_first_chunk,
            )
            if effective_duration_s <= 0:
                failed += 1
                print(f"[fail] {j.video_path}: no playable duration left")
                continue

            streaming_inference(
                model_path=args.model_path,
                model_base=args.model_base,
                model=model,
                processor=processor,
                video_path=str(j.video_path),
                output_dir=str(j.output_vtt),
                window_size=args.window_size,
                chunk_duration=args.chunk_duration,
                text_round=args.text_round,
                previous_text=args.previous_text,
                pos_mode=args.pos_mode,
                all_text=args.all_text,
                skip_first_chunk=args.skip_first_chunk,
                recompute=args.recompute,
                text_sink=args.text_sink,
                text_sliding_window=args.text_sliding_window,
                temperature=args.temperature,
                duration=effective_duration_s,
                query=_pick_query_for_job(args.query, args.query_seed, j),
                repetition_penalty=args.repetition_penalty,
                quiet=args.quiet,
                emit_json=False,
            )
            completed += 1
        except Exception as e:
            failed += 1
            print(f"[fail] {j.video_path} -> {j.output_vtt}: {e}")
            if not args.quiet:
                traceback.print_exc()

    print(f"done: completed={completed} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
