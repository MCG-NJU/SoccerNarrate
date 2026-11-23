#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch StreamingVLM inference for SoccerNet ASR test split.

This script scans the SoccerNet test annotation tree and runs StreamingVLM
inference for each match half video, writing WebVTT files while mirroring the
directory layout under the annotation root.

Expected annotation layout (annos_root):
  annos_root/
    <league>_<season>/
      <match_name>/
        1-224p_processed_clean.json
        2-224p_processed_clean.json
        ...

Expected video layout (video_root):
  video_root/
    <league>/
      <season>/
        <match_name>/
          1_720p.mkv (or .mp4)
          2_720p.mkv (or .mp4)

Output layout (output_root):
  output_root/
    <league>_<season>/
      <match_name>/
        1_720p.vtt
        2_720p.vtt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shlex
import socket
import sys
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence


_LEAGUE_SEASON_RE = re.compile(r"^(?P<league>.+)_(?P<season>\d{4}-\d{4})$")
_GAME_TIME_RE = re.compile(r"^\s*(?P<half>[12])\s*-\s*(?P<min>\d+):(?P<sec>\d+)\s*$")
DEFAULT_REMOTE_SETUP = (
    'for act in '
    '/path/to/streaming-vlm/venv/bin/activate '
    '; do [ -f "$act" ] && source "$act" && break; done'
)

_VIDEO_PREPROCESS_ENV_DEFAULTS: dict[str, str] = {
    # Inference defaults: avoid forcing a pixel cap here (let downstream defaults / env handle it).
    "FPS_MAX_FRAMES": "64",
}


def _setenv_default(key: str, value: str) -> None:
    if not os.environ.get(key, "").strip():
        os.environ[key] = str(value)


def _ensure_video_preprocess_env() -> None:
    for key, value in _VIDEO_PREPROCESS_ENV_DEFAULTS.items():
        _setenv_default(key, value)


def _render_video_preprocess_exports() -> str:
    lines: list[str] = []
    for key, default in _VIDEO_PREPROCESS_ENV_DEFAULTS.items():
        value = os.environ.get(key, "").strip() or default
        lines.append(f"export {key}={shlex.quote(str(value))}")
    return "\n".join(lines)


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {v}")


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
    match_dir: Path
    league: str
    season: str
    match_name: str
    half: int
    video_path: Path
    output_vtt: Path


@dataclass(frozen=True)
class HostSpec:
    ssh_target: str
    host: str
    is_local: bool


def _stable_int_hash(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _pick_query_for_job(query: Optional[str], query_seed: Optional[int], job: VideoJob) -> str:
    """
    - If `query` is provided, use it as-is.
    - Otherwise choose randomly from `SOCCER_COMMENTARY_QUERIES`.
      When `query_seed` is set, the choice is deterministic per job.
    """
    if query:
        return query
    if query_seed is None:
        return random.choice(SOCCER_COMMENTARY_QUERIES)
    job_key = f"{query_seed}|{job.league}|{job.season}|{job.match_name}|{job.half}"
    rng = random.Random(_stable_int_hash(job_key))
    return rng.choice(SOCCER_COMMENTARY_QUERIES)


def _ensure_import_paths() -> None:
    """
    Make imports work when this file is executed as a script.

    - Adds the `streaming-vlm/` root so `import streaming_vlm` works.
    - Adds `streaming-vlm/streaming_vlm/` so `import livecc_utils` (vendored) works.
    """
    this_file = Path(__file__).resolve()
    streaming_vlm_root = this_file.parents[2]  # .../streaming-vlm
    streaming_vlm_pkg = this_file.parents[1]  # .../streaming_vlm

    for p in (streaming_vlm_root, streaming_vlm_pkg):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def _parse_league_season(dirname: str) -> Optional[tuple[str, str]]:
    m = _LEAGUE_SEASON_RE.match(dirname)
    if not m:
        return None
    return m.group("league"), m.group("season")


def _normalize_video_root(video_root: Path) -> Path:
    if (video_root / "videos").is_dir():
        return video_root / "videos"
    return video_root


def _normalize_label_root(label_root: Path) -> Path:
    """
    Accept either:
      - .../soccernet-label/annos/raw
      - .../soccernet-label (contains annos/raw)
      - .../soccernet-label/annos (contains raw)
    """
    if (label_root / "annos" / "raw").is_dir():
        return label_root / "annos" / "raw"
    if (label_root / "raw").is_dir() and label_root.name != "raw":
        return label_root / "raw"
    return label_root


def _label_file_for_job(job: VideoJob, label_root: Path) -> Path:
    base = label_root / job.league / job.season / job.match_name
    for name in ("Labels-v2.json", "Labels.json"):
        cand = base / name
        if cand.exists():
            return cand
    return base / "Labels-v2.json"


def _parse_game_time(game_time: str) -> tuple[Optional[int], Optional[float]]:
    if not isinstance(game_time, str):
        return None, None
    m = _GAME_TIME_RE.match(game_time)
    if not m:
        return None, None
    half = int(m.group("half"))
    minutes = int(m.group("min"))
    seconds = int(m.group("sec"))
    return half, float(minutes * 60 + seconds)


def _load_label_events_by_half(label_json: Path) -> dict[int, list[dict]]:
    """
    Load SoccerNet-label Labels(-v2).json and return events grouped by half.
    Each event uses half-local seconds (same coord as half video).
    """
    with open(label_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped: dict[int, list[dict]] = {1: [], 2: []}
    for ann in data.get("annotations", []) or []:
        half, game_secs = _parse_game_time(ann.get("gameTime", ""))
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
            t = game_secs
        if t is None:
            continue

        label = ann.get("label", None)
        if not label:
            continue

        event = {"t": float(t), "label": str(label), "conf": 1.0}
        team = ann.get("team", None)
        if team and team != "not applicable":
            event["team"] = team
        visibility = ann.get("visibility", None)
        if visibility:
            event["visibility"] = visibility

        grouped[half].append(event)

    grouped[1].sort(key=lambda x: x.get("t", 0.0))
    grouped[2].sort(key=lambda x: x.get("t", 0.0))
    return grouped


def _find_half_video(match_video_dir: Path, half: int, exts: Sequence[str]) -> Optional[Path]:
    for ext in exts:
        ext = ext.lstrip(".")
        candidate = match_video_dir / f"{half}_720p.{ext}"
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1024)
def _get_video_duration_seconds(video_path: str) -> Optional[float]:
    """
    Return the best-effort video duration in seconds.

    We prefer the last frame's end timestamp because `streaming_inference()`
    indexes chunks using real frame timestamps, and those can differ slightly
    from `len(vr) / fps` near the end of the file.
    """
    try:
        import decord

        vr = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(vr)
        if total_frames <= 0:
            return None

        try:
            last_ts = vr.get_frame_timestamp(total_frames - 1)
            if hasattr(last_ts, "tolist"):
                last_ts = last_ts.tolist()
            if isinstance(last_ts, (list, tuple)):
                if len(last_ts) >= 2:
                    end_s = float(last_ts[1])
                elif len(last_ts) == 1:
                    end_s = float(last_ts[0])
                else:
                    end_s = None
                if end_s is not None and end_s > 0:
                    return end_s
        except Exception:
            pass

        fps = float(vr.get_avg_fps())
        if fps > 0:
            return float(total_frames) / fps
    except Exception:
        pass

    try:
        from torchvision.io import VideoReader

        vr = VideoReader(video_path, "video")
        meta = vr.get_metadata() or {}
        video_meta = meta.get("video", {}) or {}
        fps_list = video_meta.get("fps", []) or []
        dur_list = video_meta.get("duration", []) or []
        nframes_list = video_meta.get("num_frames", []) or []
        if dur_list:
            duration = float(dur_list[0])
            if duration > 0:
                return duration
        if fps_list and nframes_list:
            fps = float(fps_list[0])
            total_frames = float(nframes_list[0])
            if fps > 0 and total_frames > 0:
                return total_frames / fps
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


def _iter_match_dirs(annos_root: Path) -> Iterable[Path]:
    for league_season_dir in sorted(p for p in annos_root.iterdir() if p.is_dir()):
        for match_dir in sorted(p for p in league_season_dir.iterdir() if p.is_dir()):
            yield match_dir


def build_jobs(
    annos_root: Path,
    output_root: Path,
    video_root: Path,
    video_exts: Sequence[str],
    strict_missing_video: bool,
) -> list[VideoJob]:
    jobs: list[VideoJob] = []
    video_root = _normalize_video_root(video_root)

    for match_dir in _iter_match_dirs(annos_root):
        rel_match_dir = match_dir.relative_to(annos_root)
        league_season = rel_match_dir.parts[0]
        parsed = _parse_league_season(league_season)
        if parsed is None:
            raise ValueError(
                f"Unexpected league-season directory name: {league_season!r} "
                f"(expected '<league>_<season>' like 'england_epl_2015-2016')"
            )
        league, season = parsed
        match_name = rel_match_dir.parts[1]

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

            output_vtt = out_match_dir / f"{half}_720p.vtt"
            jobs.append(
                VideoJob(
                    rel_match_dir=rel_match_dir,
                    match_dir=match_dir,
                    league=league,
                    season=season,
                    match_name=match_name,
                    half=half,
                    video_path=video_path,
                    output_vtt=output_vtt,
                )
            )

    return jobs


def _read_hostfile(hostfile: Path) -> list[str]:
    hosts: list[str] = []
    for line in hostfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        hosts.append(line)
    return hosts


def _split_hosts(hosts_arg: str) -> list[str]:
    hosts: list[str] = []
    for h in hosts_arg.split(","):
        h = h.strip()
        if h:
            hosts.append(h)
    return hosts


def _local_identifiers() -> set[str]:
    ids: set[str] = {"localhost", "127.0.0.1", socket.gethostname()}
    try:
        ids.add(socket.getfqdn())
    except Exception:
        pass

    try:
        out = subprocess.check_output(["hostname", "-I"], text=True).strip()
        for ip in out.split():
            ids.add(ip)
    except Exception:
        pass

    return ids


def _strip_user(ssh_target: str) -> str:
    return ssh_target.split("@", 1)[1] if "@" in ssh_target else ssh_target


def _build_host_specs(hosts: Sequence[str]) -> list[HostSpec]:
    local_ids = _local_identifiers()
    specs: list[HostSpec] = []
    for raw in hosts:
        ssh_target = raw.strip()
        if not ssh_target:
            continue
        host = _strip_user(ssh_target)
        specs.append(HostSpec(ssh_target=ssh_target, host=host, is_local=host in local_ids))
    return specs


def _parse_int_from_stdout(stdout: str) -> int:
    tokens = stdout.strip().split()
    for tok in reversed(tokens):
        try:
            return int(tok)
        except ValueError:
            continue
    return 0


def _detect_gpu_count_local() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        count = sum(1 for line in out.splitlines() if line.startswith("GPU "))
        if count > 0:
            return count
    except Exception:
        pass

    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _detect_gpu_count_ssh(
    ssh_bin: str,
    ssh_base_args: Sequence[str],
    ssh_target: str,
    remote_setup: str,
    python_bin: str,
) -> int:
    remote_setup_snippet = (remote_setup + "; ") if remote_setup else ""
    detect_cmd = (
        "set -euo pipefail; "
        + remote_setup_snippet
        + "if command -v nvidia-smi >/dev/null 2>&1; then "
        + "nvidia-smi -L | grep -c '^GPU '; "
        + "else "
        + shlex.quote(python_bin)
        + " - <<'PY'\n"
        + "try:\n"
        + "  import torch\n"
        + "  print(int(torch.cuda.device_count()))\n"
        + "except Exception:\n"
        + "  print(0)\n"
        + "PY\n"
        + "fi"
    )
    proc = subprocess.run(
        [ssh_bin, *ssh_base_args, ssh_target, "bash", "-lc", detect_cmd],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"GPU detect failed on {ssh_target} (rc={proc.returncode}):\n{proc.stdout}")
    return _parse_int_from_stdout(proc.stdout)


def _sanitize_tag(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


def _has_arg(argv: Sequence[str], flag: str) -> bool:
    if flag in argv:
        return True
    prefix = flag + "="
    return any(a.startswith(prefix) for a in argv)


def _strip_multi_host_args(argv: Sequence[str]) -> list[str]:
    # Remove launcher-only args so they don't get forwarded to worker processes.
    # Supports both `--opt value` and `--opt=value` forms.
    to_strip_with_value = {
        "--hosts",
        "--hostfile",
        "--workdir",
        "--remote_setup",
        "--gpus_per_host",
        "--ssh",
        "--ssh_port",
        "--ssh_opt",
        "--multi_host_log_dir",
        "--python",
    }
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        stripped = False
        for opt in to_strip_with_value:
            if a == opt:
                # consume next token as value if present
                i += 2
                stripped = True
                break
            if a.startswith(opt + "="):
                i += 1
                stripped = True
                break
        if stripped:
            continue
        cleaned.append(a)
        i += 1
    return cleaned


def _build_remote_bash(
    *,
    workdir: Path,
    python_bin: str,
    script_rel: str,
    passthrough_args: Sequence[str],
    total_shards: int,
    shard_offset: int,
    gpus_on_host: int,
    log_dir: Path,
    host_tag: str,
    remote_setup: str,
) -> str:
    extra_args = " ".join(shlex.quote(a) for a in passthrough_args)
    remote_setup_quoted = shlex.quote(remote_setup)
    video_env_exports = _render_video_preprocess_exports()
    return f"""\
set -euo pipefail

cd {shlex.quote(str(workdir))}

REMOTE_SETUP={remote_setup_quoted}
if [ -n "$REMOTE_SETUP" ]; then
  eval "$REMOTE_SETUP"
fi

{video_env_exports}

PYTHON={shlex.quote(python_bin)}
SCRIPT={shlex.quote(script_rel)}
TOTAL_SHARDS={total_shards}
OFFSET={shard_offset}
GPUS_ON_HOST={gpus_on_host}
HOST_TAG={shlex.quote(host_tag)}
LOG_DIR={shlex.quote(str(log_dir))}

mkdir -p "$LOG_DIR"

EXTRA_ARGS=({extra_args})

pids=()
for local_gpu in $(seq 0 $((GPUS_ON_HOST - 1))); do
  global_shard=$((OFFSET + local_gpu))
  log_file="$LOG_DIR/${{HOST_TAG}}_shard${{global_shard}}_gpu${{local_gpu}}.log"
  echo "[launch] host=${{HOST_TAG}} local_gpu=${{local_gpu}} shard_id=${{global_shard}} log=${{log_file}}"
  CUDA_VISIBLE_DEVICES=${{local_gpu}} "$PYTHON" "$SCRIPT" \\
    --num_shards "$TOTAL_SHARDS" \\
    --shard_id "$global_shard" \\
    --disable_auto_gpus \\
    "${{EXTRA_ARGS[@]}}" \\
    >"$log_file" 2>&1 &
  pids+=($!)
done

exit_code=0
for pid in "${{pids[@]}}"; do
  if ! wait "$pid"; then
    exit_code=2
  fi
done
exit "$exit_code"
"""


def _start_host_job(
    *,
    host: HostSpec,
    ssh_bin: str,
    ssh_base_args: Sequence[str],
    remote_bash: str,
) -> subprocess.Popen:
    if host.is_local:
        proc = subprocess.Popen(["bash", "-s"], stdin=subprocess.PIPE)
    else:
        proc = subprocess.Popen([ssh_bin, *ssh_base_args, host.ssh_target, "bash", "-s"], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    proc.stdin.write(remote_bash.encode("utf-8"))
    proc.stdin.close()
    return proc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch infer SoccerNet ASR test videos with StreamingVLM and save WebVTT files."
    )
    parser.add_argument(
        "--annos_root",
        type=Path,
        default=Path("/path/to/SoccerNet/asr/annos/test"),
        help="SoccerNet ASR test annotation root (directory tree to mirror).",
    )
    parser.add_argument(
        "--video_root",
        type=Path,
        default=Path(os.environ.get("SOCCERNET_VIDEOS_ROOT", "/path/to/SoccerNet/videos")),
        help="SoccerNet videos root (contains <league>/<season>/<match>/[1|2]_720p.*). "
        "If this points to the SoccerNet root, a 'videos/' subdir is auto-detected.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default="streaming-vlm/streaming_vlm/inference/infer_test/baseline",
        help="Output root directory. Will mirror annos_root structure and write 1_720p.vtt/2_720p.vtt.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="streaming-vlm/checkpoints/qwen2_5_vl_lr1e-5_20251216_180847_baseline/checkpoint-205",
        help="HuggingFace model path or local checkpoint dir.",
    )
    parser.add_argument("--model_base", type=str, choices=["Qwen2_5", "Qwen2"], default="Qwen2_5")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query/instruction text. If omitted, one is chosen randomly from SOCCER_COMMENTARY_QUERIES.",
    )
    parser.add_argument(
        "--query_seed",
        type=int,
        default=None,
        help="Seed for random query selection (deterministic per job). Only used when --query is omitted.",
    )
    parser.add_argument(
        "--label_root",
        type=Path,
        default=Path(os.environ.get("SOCCERNET_LABEL_DIR", "/path/to/SoccerNet/SoccerNet-label/raw_annos")),
        help="SoccerNet-label root directory, either raw or its parent. When set, oracle labels are injected into <event_candidates> in the prompt.",
    )
    parser.add_argument(
        "--no_label_oracle",
        action="store_true",
        default=True, # True keeps oracle injection disabled by default
        help="Disable SoccerNet-label oracle injection and do not add <event_candidates>.",
    )

    parser.add_argument("--pos_mode", type=str, default="shrink", choices=["append", "shrink"])
    parser.add_argument("--all_text", action="store_true", default=False)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--chunk_duration", type=int, default=1)
    parser.add_argument("--text_round", type=int, default=16)
    parser.add_argument("--skip_first_chunk", type=int, default=0)
    parser.add_argument("--recompute", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument(
        "--do_sample",
        type=_str2bool,
        default=True,
        help="If true, enable sampling (temperature is used). If false, decoding is greedy and temperature is ignored.",
    )
    parser.add_argument("--min_new_tokens", type=int, default=0, help="Minimum new tokens per second chunk (optional).")
    parser.add_argument("--duration", type=int, default=6000)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--text_sink", type=int, default=512)
    parser.add_argument("--text_sliding_window", type=int, default=512)
    parser.add_argument(
        "--previous_text",
        type=str,
        default="",
        help="Initial 'previous text' context fed to the model (empty by default).",
    )

    parser.add_argument(
        "--video_exts",
        type=str,
        default="mkv,mp4",
        help="Video extensions to try, comma-separated (e.g. 'mkv,mp4').",
    )
    parser.add_argument("--strict_missing_video", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument(
        "--write_sentence_txt",
        action="store_true",
        default=False,
        help="Also write sentence-merged .txt alongside .vtt (off by default).",
    )
    parser.add_argument("--num_shards", type=int, default=1, help="Split jobs into N shards for parallel runs.")
    parser.add_argument("--shard_id", type=int, default=0, help="Which shard index to run (0..num_shards-1).")
    parser.add_argument("--max_jobs", type=int, default=0, help="If >0, only run first N jobs after sharding.")
    parser.add_argument(
        "--disable_auto_gpus",
        action="store_true",
        default=False,
        help="Disable automatic multi-GPU parallel inference (auto-enabled when multiple GPUs are visible).",
    )
    parser.add_argument(
        "--hosts",
        type=str,
        default="",
        help="Comma-separated SSH targets for multi-host inference (e.g. '10.0.0.1,10.0.0.2' or 'user@10.0.0.2'). "
        "When set, this script will launch sharded workers on each host via SSH.",
    )
    parser.add_argument(
        "--hostfile",
        type=Path,
        default=None,
        help="Optional file with one SSH target per line (# comments allowed). Appends to --hosts.",
    )
    try:
        default_workdir = Path(__file__).resolve().parents[4]
    except Exception:
        default_workdir = Path.cwd()
    parser.add_argument(
        "--workdir",
        type=Path,
        default=default_workdir,
        help="Repository root path on every host (must exist and contain this script).",
    )
    parser.add_argument(
        "--remote_setup",
        type=str,
        default=DEFAULT_REMOTE_SETUP,
        help="Shell snippet run on each host before launching workers (default: auto-source StreamingVLM venv if found).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default="python",
        help="Python executable to use on each host in multi-host mode (default: python in PATH after remote_setup).",
    )
    parser.add_argument(
        "--gpus_per_host",
        type=str,
        default="auto",
        help="GPUs per host in multi-host mode: 'auto' (default) or an integer like '8'.",
    )
    parser.add_argument("--ssh", type=str, default="ssh", help="SSH binary (default: ssh).")
    parser.add_argument("--ssh_port", type=int, default=22, help="SSH port (default: 22).")
    parser.add_argument(
        "--ssh_opt",
        action="append",
        default=[],
        help="Extra ssh option (repeatable), e.g. --ssh_opt '-o' --ssh_opt 'StrictHostKeyChecking=no'.",
    )
    parser.add_argument(
        "--multi_host_log_dir",
        type=Path,
        default=Path("multi_host_logs"),
        help="Log directory in multi-host mode (created on each host; use a shared path if you want central logs).",
    )

    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must be in [0, num_shards)")

    _ensure_video_preprocess_env()
    if not args.quiet:
        print(
            "[video_preprocess] " f"FPS_MAX_FRAMES={os.environ.get('FPS_MAX_FRAMES')}",
            flush=True,
        )

    # Multi-host launcher: if --hosts/--hostfile is provided and includes multiple hosts (or any remote),
    # launch one worker process per GPU on each host via SSH, using global shard ids across all GPUs.
    host_list: list[str] = []
    host_list.extend(_split_hosts(args.hosts) if args.hosts else [])
    if args.hostfile is not None:
        host_list.extend(_read_hostfile(args.hostfile))
    # De-dup while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for h in host_list:
        if h not in seen:
            seen.add(h)
            deduped.append(h)
    host_list = deduped
    if host_list:
        hosts = _build_host_specs(host_list)
        use_multi_host = len(hosts) > 1 or any(not h.is_local for h in hosts)
    else:
        hosts = []
        use_multi_host = False

    if use_multi_host:
        argv = sys.argv[1:]
        if _has_arg(argv, "--num_shards") or _has_arg(argv, "--shard_id"):
            raise ValueError("Do not pass --num_shards/--shard_id when using --hosts; they are set automatically.")

        workdir = Path(args.workdir).resolve()
        default_rel = Path("streaming-vlm/streaming_vlm/inference/batch_infer_soccernet_asr_test.py")
        if (workdir / default_rel).exists():
            script_rel = str(default_rel)
        else:
            try:
                script_rel = str(Path(__file__).resolve().relative_to(workdir))
            except Exception as e:
                raise ValueError(
                    f"--workdir must point to the repo root containing this script. "
                    f"Got workdir={workdir} but __file__={Path(__file__).resolve()}"
                ) from e

        ssh_base_args = ["-p", str(args.ssh_port), *args.ssh_opt]

        if args.gpus_per_host != "auto":
            try:
                fixed = int(args.gpus_per_host)
            except ValueError as e:
                raise ValueError("--gpus_per_host must be 'auto' or an integer") from e
            if fixed < 1:
                raise ValueError("--gpus_per_host must be >= 1")
            per_host_gpus = [fixed for _ in hosts]
        else:
            per_host_gpus: list[int] = []
            for h in hosts:
                if h.is_local:
                    g = _detect_gpu_count_local()
                else:
                    g = _detect_gpu_count_ssh(args.ssh, ssh_base_args, h.ssh_target, args.remote_setup, args.python)
                if g < 1:
                    raise RuntimeError(
                        f"No GPUs detected on host {h.ssh_target}. Set --gpus_per_host explicitly if needed."
                    )
                per_host_gpus.append(g)

        total_shards = sum(per_host_gpus)
        if total_shards < 1:
            raise RuntimeError("No GPUs detected across all hosts.")

        passthrough_args = _strip_multi_host_args(sys.argv[1:])

        print("==========================================", flush=True)
        print("Multi-host inference launcher", flush=True)
        print("==========================================", flush=True)
        print(f"hosts:        {[h.ssh_target for h in hosts]}", flush=True)
        print(f"gpus/host:    {per_host_gpus}", flush=True)
        print(f"total_shards: {total_shards}", flush=True)
        print(f"workdir:      {workdir}", flush=True)
        print(f"script:       {script_rel}", flush=True)
        print(f"python:       {args.python}", flush=True)
        print(f"log_dir:      {args.multi_host_log_dir}", flush=True)
        if args.remote_setup:
            print("remote_setup: (set)", flush=True)
        else:
            print("remote_setup: (empty)", flush=True)
        print("==========================================", flush=True)

        procs: list[tuple[HostSpec, int, int, str, subprocess.Popen]] = []
        shard_offset = 0
        for h, gpus_on_host in zip(hosts, per_host_gpus, strict=True):
            shard_start = shard_offset
            shard_end = shard_offset + gpus_on_host - 1
            host_tag = _sanitize_tag(h.ssh_target.replace("@", "_"))
            remote_bash = _build_remote_bash(
                workdir=workdir,
                python_bin=args.python,
                script_rel=script_rel,
                passthrough_args=passthrough_args,
                total_shards=total_shards,
                shard_offset=shard_offset,
                gpus_on_host=gpus_on_host,
                log_dir=args.multi_host_log_dir,
                host_tag=host_tag,
                remote_setup=args.remote_setup,
            )
            print(f"[plan] {h.ssh_target}: shard_id {shard_start}..{shard_end} (gpus={gpus_on_host})", flush=True)
            shard_offset += gpus_on_host

            procs.append(
                (
                    h,
                    shard_start,
                    shard_end,
                    host_tag,
                    _start_host_job(host=h, ssh_bin=args.ssh, ssh_base_args=ssh_base_args, remote_bash=remote_bash),
                )
            )

        exit_code = 0
        for h, shard_start, shard_end, host_tag, p in procs:
            rc = p.wait()
            if rc != 0:
                exit_code = 2
                example_log = args.multi_host_log_dir / f"{host_tag}_shard{shard_start}_gpu0.log"
                print(
                    f"[fail] host {h.ssh_target} exited with code {rc} (shards {shard_start}..{shard_end}). "
                    f"Check logs under {args.multi_host_log_dir} (e.g. {example_log})."
                )
            else:
                print(f"[ok]   host {h.ssh_target}")
        return exit_code

    # Auto multi-GPU: if the user didn't manually shard, use one process per visible GPU.
    # We spawn subprocesses with CUDA_VISIBLE_DEVICES pinned to a single GPU so each process loads
    # the model on that GPU without extra device_map plumbing.
    if (
        not args.disable_auto_gpus
        and not args.dry_run
        and args.num_shards == 1
        and args.shard_id == 0
    ):
        try:
            import torch

            num_gpus = int(torch.cuda.device_count())
        except Exception:
            num_gpus = 0

        if num_gpus > 1:
            script_path = str(Path(__file__).resolve())
            base_argv = [sys.executable, script_path]
            passthrough = sys.argv[1:]

            procs: list[subprocess.Popen] = []
            for gpu_id in range(num_gpus):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                argv = (
                    base_argv
                    + passthrough
                    + [
                        "--num_shards",
                        str(num_gpus),
                        "--shard_id",
                        str(gpu_id),
                        "--disable_auto_gpus",
                    ]
                )
                procs.append(subprocess.Popen(argv, env=env))

            exit_codes = [p.wait() for p in procs]
            if any(code != 0 for code in exit_codes):
                print(f"[fail] multi-gpu subprocess exit codes: {exit_codes}")
                return 2
            return 0

    annos_root: Path = args.annos_root
    output_root: Path = args.output_root
    video_root: Path = args.video_root

    if not annos_root.is_dir():
        raise FileNotFoundError(f"annos_root not found: {annos_root}")

    video_exts = [e.strip() for e in args.video_exts.split(",") if e.strip()]
    if not video_exts:
        raise ValueError("--video_exts must contain at least one extension")

    jobs = build_jobs(
        annos_root=annos_root,
        output_root=output_root,
        video_root=video_root,
        video_exts=video_exts,
        strict_missing_video=args.strict_missing_video,
    )
    jobs.sort(key=lambda j: (str(j.rel_match_dir), j.half))
    jobs = [j for idx, j in enumerate(jobs) if idx % args.num_shards == args.shard_id]

    if args.max_jobs and args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]

    print(f"annos_root:  {annos_root}")
    print(f"video_root:  {_normalize_video_root(video_root)}")
    print(f"output_root: {output_root}")
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

    label_root: Optional[Path] = None
    label_cache: dict[Path, dict[int, list[dict]]] = {}
    # if not args.no_label_oracle and args.label_root:
    #     cand = _normalize_label_root(args.label_root)
    #     if cand.is_dir():
    #         label_root = cand
    #         print(f"label_root:  {label_root} (oracle enabled)")
    #     else:
    #         print(f"[warn] label_root not found, disable oracle: {cand}")

    completed = 0
    skipped = 0
    failed = 0
    for j in jobs:
        if j.output_vtt.exists() and not args.overwrite:
            skipped += 1
            if not args.quiet:
                print(f"[skip] exists: {j.output_vtt}")
            continue

        j.output_vtt.parent.mkdir(parents=True, exist_ok=True)
        if not args.quiet:
            print(f"[run]  {j.video_path} -> {j.output_vtt}")

        try:
            event_stream = None
            if label_root is not None:
                label_json = _label_file_for_job(j, label_root)
                if label_json.exists():
                    if label_json not in label_cache:
                        try:
                            label_cache[label_json] = _load_label_events_by_half(label_json)
                        except Exception as e:
                            if not args.quiet:
                                print(f"[warn] failed to load label: {label_json}: {e}")
                            label_cache[label_json] = {1: [], 2: []}
                    event_stream = label_cache[label_json].get(j.half, [])
                elif not args.quiet:
                    print(f"[warn] missing label: {label_json}")

            actual_duration_s = _get_video_duration_seconds(str(j.video_path))
            effective_duration_s = _effective_duration_seconds(
                requested_duration=args.duration,
                actual_duration=actual_duration_s,
                chunk_duration=args.chunk_duration,
                skip_first_chunk=args.skip_first_chunk,
            )
            estimated_chunks = int((effective_duration_s + float(args.chunk_duration) - 1) // float(args.chunk_duration))
            if effective_duration_s <= 0 or estimated_chunks <= 0:
                failed += 1
                print(
                    f"[fail] {j.video_path} -> {j.output_vtt}: "
                    f"no playable duration left "
                    f"(actual={actual_duration_s}, skip_first_chunk={args.skip_first_chunk}, "
                    f"chunk_duration={args.chunk_duration}, effective_duration={effective_duration_s})"
                )
                continue
            if (actual_duration_s is not None) and (effective_duration_s + 1e-6 < float(args.duration)) and (not args.quiet):
                skip_offset_s = float(args.skip_first_chunk) * float(args.chunk_duration)
                print(
                    f"[info] clamp duration: requested={float(args.duration):.3f}s "
                    f"actual={float(actual_duration_s):.3f}s skip_offset={skip_offset_s:.3f}s "
                    f"run={effective_duration_s:.3f}s for {j.video_path.name}"
                )

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
                # do_sample=bool(args.do_sample),
                # min_new_tokens=int(args.min_new_tokens),
                duration=effective_duration_s,
                query=_pick_query_for_job(args.query, args.query_seed, j),
                repetition_penalty=args.repetition_penalty,
                quiet=args.quiet,
                emit_json=False,
                # disable_sentence_output=not args.write_sentence_txt,
                # event_stream=event_stream,
            )
            completed += 1
        except Exception as e:
            failed += 1
            print(f"[fail] {j.video_path} -> {j.output_vtt}: {e}")
            if not args.quiet:
                import traceback

                traceback.print_exc()

    print(f"done: completed={completed} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
