#!/usr/bin/env python3
"""
Plot Off-timing Mention (OTM) time-diff distributions from *sliding_window_event_mentions*.jsonl.

Expected JSONL (per open/negative window) includes:
  row["otm_nearest_gt"] = [
    {
      "pred_label": "<SoccerNet label>",
      "nearest_gt_sec": <float>,
      "signed_time_diff_sec": <float>,
      "abs_time_diff_sec": <float>
    },
    ...
  ]

Signed time diff convention:
  + : nearest same-label GT event is before the window (late mention)
  - : nearest same-label GT event is after the window (early mention)

Example:
  python eval/llm_as_a_judge/stat/plot_otm_time_diffs.py \
    --jsonl /path/to/human_sliding_window_event_mentions.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        x = float(value)
        if not math.isfinite(x):
            return None
        return x
    if value is None:
        return None
    try:
        x = float(str(value).strip())
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def _iter_otm_offsets(jsonl_path: Path) -> Iterable[tuple[str, float]]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue

            offsets = row.get("otm_nearest_gt")
            if not isinstance(offsets, list) or not offsets:
                continue
            for item in offsets:
                if not isinstance(item, dict):
                    continue
                label = item.get("pred_label")
                if not isinstance(label, str) or not label.strip():
                    continue
                signed = _safe_float(item.get("signed_time_diff_sec"))
                if signed is None:
                    continue
                yield label.strip(), float(signed)


def _maybe_clip(values: np.ndarray, *, clip_seconds: Optional[float]) -> np.ndarray:
    if clip_seconds is None:
        return values
    c = float(clip_seconds)
    if not math.isfinite(c) or c <= 0:
        return values
    return np.clip(values, -c, c)


def plot_histogram(
    values: np.ndarray,
    *,
    title: str,
    out_path: Path,
    bins: int,
    clip_seconds: Optional[float],
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    values_plot = _maybe_clip(values, clip_seconds=clip_seconds)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(values_plot, bins=int(bins), color="#4C78A8", alpha=0.85)
    ax.axvline(0.0, color="#000000", linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("signed_time_diff_sec (window vs nearest same-label GT)")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.25)

    neg = int(np.sum(values < 0))
    pos = int(np.sum(values > 0))
    zero = int(np.sum(values == 0))
    ax.text(
        0.99,
        0.98,
        f"n={values.size}\nneg={neg} pos={pos} zero={zero}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_label_grid(
    by_label: Dict[str, np.ndarray],
    *,
    out_path: Path,
    bins: int,
    clip_seconds: Optional[float],
    min_count: int,
):
    labels = [k for k, v in sorted(by_label.items(), key=lambda kv: (-kv[1].size, kv[0])) if v.size >= int(min_count)]
    if not labels:
        return

    n = len(labels)
    ncols = 4 if n >= 4 else n
    nrows = int(math.ceil(n / float(ncols)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 2.6 * nrows), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    all_vals = np.concatenate([by_label[l] for l in labels], axis=0)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        return
    xmin = float(np.min(_maybe_clip(all_vals, clip_seconds=clip_seconds)))
    xmax = float(np.max(_maybe_clip(all_vals, clip_seconds=clip_seconds)))
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0

    for idx, label in enumerate(labels):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        vals = by_label[label]
        vals = vals[np.isfinite(vals)]
        vals_plot = _maybe_clip(vals, clip_seconds=clip_seconds)
        ax.hist(vals_plot, bins=int(bins), color="#F58518", alpha=0.85)
        ax.axvline(0.0, color="#000000", linewidth=1.0, alpha=0.7)
        ax.set_title(f"{label} (n={vals.size})", fontsize=10)
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_xlim(xmin, xmax)

    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    fig.suptitle("OTM signed_time_diff_sec by label", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OTM time diffs from sliding-window JSONL outputs")
    parser.add_argument("--jsonl", type=Path, required=True, help="Input JSONL (e.g., human_sliding_window_event_mentions.jsonl)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: <jsonl.parent>/otm_plots)")
    parser.add_argument("--bins", type=int, default=80, help="Histogram bins (default: 80)")
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=300.0,
        help="Clip signed_time_diff_sec to [-clip, clip] for plotting (default: 300; set <=0 to disable)",
    )
    parser.add_argument("--min-count-per-label", type=int, default=20, help="Min samples per label for grid plot (default: 20)")
    args = parser.parse_args()

    jsonl_path: Path = args.jsonl
    if not jsonl_path.exists():
        raise FileNotFoundError(str(jsonl_path))

    out_dir = args.out_dir if args.out_dir is not None else (jsonl_path.parent / "otm_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    by_label: DefaultDict[str, List[float]] = defaultdict(list)
    all_values: List[float] = []
    for label, signed in _iter_otm_offsets(jsonl_path):
        by_label[label].append(float(signed))
        all_values.append(float(signed))

    if not all_values:
        print(f"[skip] no otm_nearest_gt offsets found in: {jsonl_path}")
        return

    values = np.asarray(all_values, dtype=float)
    clip_seconds = float(args.clip_seconds) if (args.clip_seconds is not None and args.clip_seconds > 0) else None

    plot_histogram(
        values,
        title="OTM signed_time_diff_sec (all labels)",
        out_path=out_dir / "otm_signed_time_diff_hist_all.png",
        bins=int(args.bins),
        clip_seconds=clip_seconds,
    )

    by_label_np: Dict[str, np.ndarray] = {k: np.asarray(v, dtype=float) for k, v in by_label.items()}
    plot_per_label_grid(
        by_label_np,
        out_path=out_dir / "otm_signed_time_diff_hist_by_label.png",
        bins=int(max(args.bins // 2, 20)),
        clip_seconds=clip_seconds,
        min_count=int(args.min_count_per_label),
    )

    # Small JSON summary for quick inspection.
    summary: Dict[str, Any] = {
        "jsonl": str(jsonl_path),
        "total_offsets": int(values.size),
        "clip_seconds": clip_seconds,
        "per_label": {},
    }
    for label, vals in sorted(by_label_np.items(), key=lambda kv: (-kv[1].size, kv[0])):
        vals_f = vals[np.isfinite(vals)]
        if vals_f.size == 0:
            continue
        summary["per_label"][label] = {
            "count": int(vals_f.size),
            "mean_signed": float(np.mean(vals_f)),
            "median_signed": float(np.median(vals_f)),
            "mean_abs": float(np.mean(np.abs(vals_f))),
            "median_abs": float(np.median(np.abs(vals_f))),
        }

    (out_dir / "otm_time_diff_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
