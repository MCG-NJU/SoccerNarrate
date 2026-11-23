#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_DEFAULT_INFER_SCRIPT = "streaming-vlm/streaming_vlm/inference/batch_infer_soccernet_asr_test.py"


@dataclass(frozen=True)
class HostSpec:
    raw: str
    ssh_target: str
    host: str
    is_local: bool


@dataclass(frozen=True)
class LaunchedHost:
    host: HostSpec
    proc: subprocess.Popen
    host_tag: str
    shard_start: int
    shard_end: int
    gpus_on_host: int
    log_dir: Path


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
        is_local = host in local_ids
        specs.append(HostSpec(raw=raw, ssh_target=ssh_target, host=host, is_local=is_local))
    return specs


def _parse_int_from_stdout(stdout: str) -> int:
    # Be tolerant to noisy stdout; take the last integer token.
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
        import torch  # type: ignore

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _detect_gpu_count_ssh(
    ssh_bin: str, ssh_base_args: Sequence[str], ssh_target: str, remote_setup: str
) -> int:
    detect_cmd = (
        "set -euo pipefail; "
        + (remote_setup + "; " if remote_setup else "")
        + "if command -v nvidia-smi >/dev/null 2>&1; then "
        + "nvidia-smi -L | grep -c '^GPU '; "
        + "else echo 0; fi"
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


def _validate_passthrough_args(script_args: Sequence[str]) -> None:
    forbidden = {"--num_shards", "--shard_id", "--disable_auto_gpus"}
    for a in script_args:
        if a in forbidden:
            raise ValueError(f"Do not pass {a}; it is set by the launcher.")


def _sanitize_tag(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


def _build_remote_bash(
    *,
    workdir: Path,
    python_bin: str,
    script_path: str,
    script_args: Sequence[str],
    total_shards: int,
    shard_offset: int,
    gpus_on_host: int,
    log_dir: Path,
    host_tag: str,
    remote_setup: str,
) -> str:
    extra_args = " ".join(shlex.quote(a) for a in script_args)
    remote_setup_quoted = shlex.quote(remote_setup)

    return f"""\
set -euo pipefail

cd {shlex.quote(str(workdir))}

REMOTE_SETUP={remote_setup_quoted}
if [ -n "$REMOTE_SETUP" ]; then
  eval "$REMOTE_SETUP"
fi

PYTHON={shlex.quote(python_bin)}
SCRIPT={shlex.quote(script_path)}
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
        description=(
            "Launch sharded multi-GPU batch inference across one or more machines via SSH. "
            "This uses the target script's --num_shards/--shard_id to split jobs deterministically."
        )
    )
    parser.add_argument(
        "--hosts",
        type=str,
        default="localhost",
        help="Comma-separated SSH targets (e.g. '10.0.0.1,10.0.0.2' or 'user@10.0.0.2').",
    )
    parser.add_argument(
        "--hostfile",
        type=Path,
        default=None,
        help="Optional file with one SSH target per line (# comments allowed). Appends to --hosts.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="Repository workdir on every host (must exist and contain the script path).",
    )
    parser.add_argument(
        "--script",
        type=str,
        default=_DEFAULT_INFER_SCRIPT,
        help="Target inference script path (relative to workdir, or absolute).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default="python",
        help="Python executable to use on each host (default: python in PATH).",
    )
    parser.add_argument(
        "--remote_setup",
        type=str,
        default="",
        help="Optional shell snippet run on each host before launching workers (e.g. 'source ~/.bashrc && conda activate ...').",
    )
    parser.add_argument(
        "--gpus_per_host",
        type=str,
        default="auto",
        help="GPUs per host: 'auto' (default) or an integer like '8'.",
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
        "--log_dir",
        type=Path,
        default=Path("multi_host_logs"),
        help="Log directory (created on each host; use a shared path if you want central logs).",
    )
    parser.add_argument("--dry_run", action="store_true", default=False, help="Print plan/commands, don't run.")
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the target script (prefix with --).",
    )
    args = parser.parse_args()

    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]
    _validate_passthrough_args(script_args)

    host_list: list[str] = []
    host_list.extend(_split_hosts(args.hosts))
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
    if not host_list:
        raise ValueError("No hosts provided.")

    hosts = _build_host_specs(host_list)
    if not hosts:
        raise ValueError("No valid hosts after parsing.")

    ssh_base_args = ["-p", str(args.ssh_port), *args.ssh_opt]

    # GPU counts
    per_host_gpus: list[int] = []
    if args.gpus_per_host != "auto":
        try:
            fixed = int(args.gpus_per_host)
        except ValueError as e:
            raise ValueError("--gpus_per_host must be 'auto' or an integer") from e
        if fixed < 1:
            raise ValueError("--gpus_per_host must be >= 1")
        per_host_gpus = [fixed for _ in hosts]
    else:
        for h in hosts:
            if h.is_local:
                g = _detect_gpu_count_local()
            else:
                g = _detect_gpu_count_ssh(args.ssh, ssh_base_args, h.ssh_target, args.remote_setup)
            if g < 1:
                raise RuntimeError(f"No GPUs detected on host {h.ssh_target}. Set --gpus_per_host explicitly if needed.")
            per_host_gpus.append(g)

    total_shards = sum(per_host_gpus)
    if total_shards < 1:
        raise RuntimeError("No GPUs detected across all hosts.")

    print("==========================================")
    print("Multi-host sharded inference launcher")
    print("==========================================")
    print(f"hosts:        {[h.ssh_target for h in hosts]}")
    print(f"gpus/host:    {per_host_gpus}")
    print(f"total_shards: {total_shards}")
    print(f"workdir:      {args.workdir}")
    print(f"script:       {args.script}")
    print(f"python:       {args.python}")
    print(f"log_dir:      {args.log_dir}")
    if args.remote_setup:
        print("remote_setup: (set)")
    else:
        print("remote_setup: (empty)")
    print("==========================================")

    # Launch one SSH job per host; each host launches one process per GPU.
    launched: list[LaunchedHost] = []
    shard_offset = 0
    for h, gpus_on_host in zip(hosts, per_host_gpus, strict=True):
        host_tag = _sanitize_tag(h.host if "@" not in h.ssh_target else h.ssh_target.replace("@", "_"))
        shard_start = shard_offset
        shard_end = shard_offset + gpus_on_host - 1
        remote_bash = _build_remote_bash(
            workdir=args.workdir,
            python_bin=args.python,
            script_path=args.script,
            script_args=script_args,
            total_shards=total_shards,
            shard_offset=shard_offset,
            gpus_on_host=gpus_on_host,
            log_dir=args.log_dir,
            host_tag=host_tag,
            remote_setup=args.remote_setup,
        )

        print(f"[plan] {h.ssh_target}: shard_id {shard_start}..{shard_end} (gpus={gpus_on_host})")
        shard_offset += gpus_on_host

        if args.dry_run:
            continue

        launched.append(
            LaunchedHost(
                host=h,
                proc=_start_host_job(host=h, ssh_bin=args.ssh, ssh_base_args=ssh_base_args, remote_bash=remote_bash),
                host_tag=host_tag,
                shard_start=shard_start,
                shard_end=shard_end,
                gpus_on_host=gpus_on_host,
                log_dir=args.log_dir,
            )
        )

    if args.dry_run:
        print("[dry_run] done.")
        return 0

    exit_code = 0
    for item in launched:
        rc = item.proc.wait()
        if rc != 0:
            exit_code = 2
            example_log = item.log_dir / f"{item.host_tag}_shard{item.shard_start}_gpu0.log"
            print(
                f"[fail] host {item.host.ssh_target} exited with code {rc} "
                f"(shards {item.shard_start}..{item.shard_end}). "
                f"Check logs under {item.log_dir} (e.g. {example_log})."
            )
        else:
            print(f"[ok]   host {item.host.ssh_target}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
