#!/usr/bin/env python3
"""Filter a DPO JSONL file by keeping a fixed ratio of events per label.

默认用于 SoccerNet-ASR:
  input : ../dpo_annos/dpo_cross_event_15s.jsonl
  output: ../dpo_annos/dpo_cross_event_15s_label60.jsonl

The script samples unique events within each label using meta.event_key, then
writes all rows whose event was selected while preserving the original line
order. If event_key is missing for a row, that row is treated as its own event.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

keep_ratio=0.6
DEFAULT_INPUT = Path("/path/to/SoccerNet/asr/dpo_annos/dpo_cross_event_15s.jsonl")
DEFAULT_OUTPUT = Path(f"/path/to/SoccerNet/asr/dpo_annos/dpo_cross_event_15s_label{keep_ratio}.jsonl")


def _get_nested(record: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = record
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted_key)
        cur = cur[part]
    return cur


def _get_nested_optional(record: dict[str, Any], dotted_key: str) -> Any | None:
    cur: Any = record
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _target_count(total: int, ratio: float, rounding: str) -> int:
    raw = total * ratio
    if rounding == "ceil":
        count = math.ceil(raw)
    elif rounding == "round":
        count = math.floor(raw + 0.5)
    else:
        count = math.floor(raw)

    if total > 0 and ratio > 0:
        count = max(1, count)
    return min(total, count)


def _sample_sort_key(seed: int, label: str, event_id: str) -> str:
    payload = f"{seed}\0{label}\0{event_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep a fixed ratio of unique events for each label in a DPO JSONL file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=keep_ratio,
        help="Event ratio to keep per label. Default: 0.6",
    )
    parser.add_argument(
        "--label-key",
        default="meta.event_label",
        help="Dotted JSON key used as the label. Default: meta.event_label",
    )
    parser.add_argument(
        "--event-key",
        default="meta.event_key",
        help="Dotted JSON key used as the event id. Default: meta.event_key",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic hash sampling. Default: 42",
    )
    parser.add_argument(
        "--rounding",
        choices=("floor", "round", "ceil"),
        default="floor",
        help="How to convert label_count * keep_ratio to an integer. Default: floor",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-label keep counts without writing the output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.keep_ratio <= 1:
        raise ValueError(f"--keep-ratio must be in [0, 1], got {args.keep_ratio}")

    records: list[tuple[dict[str, Any], str, str]] = []
    events_by_label: dict[str, list[str]] = defaultdict(list)
    seen_events_by_label: dict[str, set[str]] = defaultdict(set)

    with args.input.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                label = str(_get_nested(record, args.label_key))
            except KeyError as exc:
                raise KeyError(f"Missing label key {args.label_key!r} at line {line_no}") from exc

            raw_event_id = _get_nested_optional(record, args.event_key)
            event_id = str(raw_event_id) if raw_event_id is not None else f"__line_{line_no}"

            records.append((record, label, event_id))
            if event_id not in seen_events_by_label[label]:
                seen_events_by_label[label].add(event_id)
                events_by_label[label].append(event_id)

    selected_events_by_label: dict[str, set[str]] = {}
    for label, event_ids in events_by_label.items():
        keep_count = _target_count(len(event_ids), args.keep_ratio, args.rounding)
        ranked_event_ids = sorted(
            event_ids,
            key=lambda event_id: _sample_sort_key(args.seed, label, event_id),
        )
        selected_events_by_label[label] = set(ranked_event_ids[:keep_count])

    kept_by_label: dict[str, int] = defaultdict(int)
    total_kept = 0
    for _, label, event_id in records:
        if event_id in selected_events_by_label[label]:
            kept_by_label[label] += 1
            total_kept += 1

    print(f"Input records : {len(records)}", file=sys.stderr)
    print(f"Output records: {total_kept}", file=sys.stderr)
    print("Per-label event counts:", file=sys.stderr)
    for label in sorted(events_by_label):
        total_events = len(events_by_label[label])
        kept_events = len(selected_events_by_label[label])
        kept_rows = kept_by_label[label]
        print(f"  {label}: keep {kept_events}/{total_events} events, {kept_rows} rows", file=sys.stderr)

    if args.dry_run:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record, label, event_id in records:
            if event_id in selected_events_by_label[label]:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"Wrote: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
