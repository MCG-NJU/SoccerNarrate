#!/usr/bin/env python3
"""
Sort per-label mention rates for each model.

Loads `event_mentions_summary.json` under `results/<model>/` (same discovery logic as stat.py),
then prints one table per model with labels sorted by mention_rate descending.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_FILENAME = "event_mentions_summary.json"


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        x = float(value)
        if not math.isfinite(x):
            return None
        return x
    try:
        x = float(str(value).strip())
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return 0
        return int(value)
    try:
        return int(float(str(value).strip()))
    except Exception:
        return 0


@dataclass
class LabelRow:
    label: str
    mention_rate: float
    total: int
    happened: float


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_model_summaries(results_dir: Path, *, models: Optional[Sequence[str]] = None) -> Iterable[Tuple[str, Path]]:
    if models:
        for model in models:
            model = str(model).strip()
            if not model:
                continue
            summary_path = results_dir / model / SUMMARY_FILENAME
            if summary_path.exists():
                yield model, summary_path
            else:
                raise FileNotFoundError(f"Missing summary: {summary_path}")
        return

    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        summary_path = child / SUMMARY_FILENAME
        if summary_path.exists():
            yield child.name, summary_path


def parse_label_rows(summary: Dict[str, Any]) -> List[LabelRow]:
    per_label = summary.get("per_label", {}) or {}
    if not isinstance(per_label, dict):
        return []

    rows: List[LabelRow] = []
    for label, raw in per_label.items():
        if not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(raw, dict):
            continue

        total = _safe_int(raw.get("total", 0))
        happened_int = _safe_int(raw.get("happened", raw.get("mentioned", 0)))

        rate = _safe_float(raw.get("mention_rate"))
        if rate is None:
            rate = _safe_float(raw.get("happened_rate"))
        if rate is None:
            rate = (float(happened_int) / float(total)) if total else 0.0

        happened = float(happened_int)
        if total > 0 and (raw.get("happened") is None) and _safe_float(raw.get("mention_rate")) is not None:
            happened = float(rate) * float(total)

        rows.append(LabelRow(label=label.strip(), mention_rate=float(rate), total=total, happened=happened))

    rows.sort(key=lambda r: (r.mention_rate, r.total, r.label), reverse=True)
    return rows


def print_model_table(model: str, rows: Sequence[LabelRow], *, top_k: Optional[int]) -> None:
    if top_k is not None:
        rows = list(rows)[: int(top_k)]

    if not rows:
        print(f"\n[{model}] (no labels)")
        return

    label_w = max(len("label"), max(len(r.label) for r in rows))
    rate_w = max(len("mention_rate"), len("0.000"))
    total_w = max(len("total"), max(len(str(r.total)) for r in rows))

    print(f"\n[{model}]")
    print(f"{'label':<{label_w}}  {'mention_rate':>{rate_w}}  {'total':>{total_w}}")
    print(f"{'-'*label_w}  {'-'*rate_w}  {'-'*total_w}")
    for r in rows:
        print(f"{r.label:<{label_w}}  {r.mention_rate:>{rate_w}.3f}  {r.total:>{total_w}d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort: per-label mention rates for each model")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory containing per-model subfolders")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model folder names under results-dir")
    parser.add_argument("--top-k", type=int, default=None, help="Only show top K labels per model")
    args = parser.parse_args()

    for model, summary_path in iter_model_summaries(args.results_dir, models=args.models):
        summary = load_json(summary_path)
        rows = parse_label_rows(summary)
        print_model_table(model, rows, top_k=args.top_k)


if __name__ == "__main__":
    main()
