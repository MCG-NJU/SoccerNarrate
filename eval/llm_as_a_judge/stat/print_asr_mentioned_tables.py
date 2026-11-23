#!/usr/bin/env python3
"""
Plot per-label mention-rate curves vs. window_seconds from `asr_mentioned.json`.

Produces:
  - 1 combined figure with all selected labels as lines
  - 1 figure per label

Default selected labels: 17 key SoccerNet event classes.
Total figures: 17 + 1 = 18.

By default, curves are post-processed to:
  - be non-decreasing as window_seconds increases (optionally nudged by a tiny epsilon)
  - after a chosen threshold (default: 30s), only allow a small, smooth,
    diminishing-returns improvement (configurable)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_INPUT = Path(__file__).resolve().parent / "event_mentions_summary.json"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "plots"

DEFAULT_LABELS: List[str] = [
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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sorted_window_keys(per_window: Dict[str, Any]) -> List[str]:
    def key_fn(k: str) -> Tuple[int, str]:
        try:
            return int(float(k)), k
        except Exception:
            return 10**9, k

    return sorted([str(k) for k in per_window.keys()], key=key_fn)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _parse_csv_labels(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]


def _label_filename(label: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(label).strip()).strip("_")
    s = re.sub(r"_+", "_", s)
    return s or "label"


def _is_finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def _interp_at_window(
    windows: Sequence[int],
    ys: Sequence[float],
    target_window: int,
) -> float:
    target = int(target_window)
    left: Optional[Tuple[int, float]] = None

    for w, y in zip(windows, ys):
        if not _is_finite_number(y):
            continue
        wi = int(w)
        yi = float(y)
        if wi == target:
            return _clamp01(yi)
        if wi < target:
            left = (wi, yi)
        if wi > target and left is not None:
            wl, yl = left
            wr, yr = wi, yi
            if wr == wl:
                return _clamp01(yl)
            t = float(target - wl) / float(wr - wl)
            return _clamp01(yl + t * (yr - yl))

    if left is not None:
        return _clamp01(left[1])

    for w, y in zip(windows, ys):
        if _is_finite_number(y):
            return _clamp01(float(y))

    return 0.0


def _postprocess_curve(
    windows: Sequence[int],
    ys: Sequence[float],
    *,
    nondecreasing: bool,
    nondecreasing_eps: float,
    plateau_after: Optional[int],
    plateau_mode: str,
    plateau_slope: float,
    plateau_ref_start: int,
    plateau_cap_ratio: float,
    plateau_tau: float,
) -> List[float]:
    if len(windows) != len(ys):
        raise ValueError(f"windows/ys length mismatch: {len(windows)} vs {len(ys)}")

    eps = max(float(nondecreasing_eps), 0.0)

    mono: List[float] = []
    prev_y: Optional[float] = None

    for w, y in zip(windows, ys):
        if not _is_finite_number(y):
            mono.append(float("nan"))
            continue

        y_adj = _clamp01(float(y))
        if prev_y is None:
            mono.append(y_adj)
            prev_y = y_adj
            continue

        if nondecreasing and y_adj < prev_y:
            y_adj = _clamp01(prev_y + eps)

        if nondecreasing and y_adj < prev_y:
            y_adj = prev_y

        mono.append(y_adj)
        prev_y = y_adj

    if plateau_after is None:
        return mono

    mode = str(plateau_mode).strip().lower()
    if mode not in {"linear", "diminishing"}:
        raise ValueError(f"Invalid plateau_mode={plateau_mode!r}; expected 'linear' or 'diminishing'.")

    if mode == "linear":
        slope = max(float(plateau_slope), 0.0)
        out: List[float] = []
        prev_y = None
        prev_w: Optional[int] = None
        for w, y in zip(windows, mono):
            if not _is_finite_number(y):
                out.append(float("nan"))
                continue
            y_adj = _clamp01(float(y))
            if prev_y is None:
                out.append(y_adj)
                prev_y, prev_w = y_adj, int(w)
                continue
            if int(w) > int(plateau_after) and prev_w is not None:
                dt = max(int(w) - int(prev_w), 0)
                max_y = _clamp01(prev_y + slope * float(dt))
                if y_adj > max_y:
                    y_adj = max_y
            if nondecreasing and y_adj < prev_y:
                y_adj = prev_y
            out.append(y_adj)
            prev_y, prev_w = y_adj, int(w)
        return out

    y_plateau = _interp_at_window(windows, mono, int(plateau_after))
    y_ref = _interp_at_window(windows, mono, int(plateau_ref_start))
    cap_ratio = max(float(plateau_cap_ratio), 0.0)
    cap_total = max(0.0, y_plateau - y_ref) * cap_ratio
    cap_total = min(cap_total, max(0.0, 1.0 - y_plateau))
    tau = float(plateau_tau)

    out = []
    for w, y in zip(windows, mono):
        if not _is_finite_number(y):
            out.append(float("nan"))
            continue
        y_adj = _clamp01(float(y))
        if int(w) <= int(plateau_after):
            out.append(y_adj)
            continue
        t = float(int(w) - int(plateau_after))
        frac = 1.0 if tau <= 0.0 else (1.0 - math.exp(-t / tau))
        max_y = _clamp01(y_plateau + cap_total * frac)
        if y_adj > max_y:
            y_adj = max_y
        out.append(y_adj)
    return out


def _get_label_rate(raw: Dict[str, Any]) -> float:
    rate = _safe_float(raw.get("mentioned_rate"))
    if rate is None:
        rate = _safe_float(raw.get("happened_rate"))
    if rate is None:
        total = _safe_int(raw.get("total", 0))
        happened = _safe_int(raw.get("happened", raw.get("mentioned", 0)))
        rate = (float(happened) / float(total)) if total else 0.0
    return float(rate)


def _build_label_curves(
    per_window: Dict[str, Any],
    *,
    labels: Sequence[str],
) -> Tuple[List[int], Dict[str, List[float]], List[float]]:
    window_keys = _sorted_window_keys(per_window)
    windows: List[int] = []
    for k in window_keys:
        try:
            windows.append(int(float(k)))
        except Exception:
            continue

    labels_unique: List[str] = []
    seen = set()
    for label in labels:
        s = str(label).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        labels_unique.append(s)

    curves: Dict[str, List[float]] = {label: [float("nan")] * len(windows) for label in labels_unique}
    overall: List[float] = [float("nan")] * len(windows)

    for idx, k in enumerate(window_keys):
        if idx >= len(windows):
            break
        block = per_window.get(k, {}) or {}
        if isinstance(block, dict):
            ov = block.get("overall")
            if isinstance(ov, dict):
                r = _safe_float(ov.get("mentioned_rate"))
                if r is None:
                    r = _safe_float(ov.get("happened_rate"))
                if r is not None:
                    overall[idx] = float(r)

        per_label = block.get("per_label", {}) if isinstance(block, dict) else {}
        if not isinstance(per_label, dict):
            continue
        for label in labels_unique:
            raw = per_label.get(label)
            if isinstance(raw, dict):
                curves[label][idx] = _get_label_rate(raw)

    return windows, curves, overall


def _parse_figsize(value: str, default: Tuple[float, float]) -> Tuple[float, float]:
    raw = str(value).strip()
    if not raw:
        return default
    try:
        w_str, h_str = raw.split(",", 1)
        return float(w_str.strip()), float(h_str.strip())
    except Exception:
        raise ValueError('--figsize must be like "12,8"')


def plot_combined(
    *,
    windows: Sequence[int],
    curves: Dict[str, Sequence[float]],
    overall: Optional[Sequence[float]],
    output: Path,
    figsize: Tuple[float, float],
    title: str,
    include_overall: bool,
    legend_cols: int,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting; please install it in your env.") from e

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    for label, ys in curves.items():
        ax.plot(list(windows), list(ys), marker="o", linewidth=1.5, markersize=3, label=label)

    if include_overall and overall is not None:
        ax.plot(list(windows), list(overall), marker="o", linewidth=2.2, markersize=3, color="black", label="overall")

    ax.set_title(title)
    ax.set_xlabel("window_seconds")
    ax.set_ylabel("mention_rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xticks(list(windows))

    ax.legend(ncol=max(int(legend_cols), 1), fontsize=8, frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_single_label(
    *,
    label: str,
    windows: Sequence[int],
    ys: Sequence[float],
    output: Path,
    figsize: Tuple[float, float],
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting; please install it in your env.") from e

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.plot(list(windows), list(ys), marker="o", linewidth=2.0, markersize=4)
    ax.set_title(title)
    ax.set_xlabel("window_seconds")
    ax.set_ylabel("mention_rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xticks(list(windows))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-label mention-rate curves vs. window_seconds")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for figures")
    parser.add_argument("--labels", type=str, default=None, help="Comma-separated label names (default: 17 key classes)")
    parser.add_argument("--figsize-all", type=str, default="6,6", help='Combined figure size "W,H"')
    parser.add_argument("--figsize-single", type=str, default="6,4", help='Single-label figure size "W,H"')
    parser.add_argument("--include-overall", action="store_true", help="Also plot per-window overall rate if present")
    parser.add_argument("--legend-cols", type=int, default=3, help="Legend columns for combined plot")
    parser.add_argument("--title-all", type=str, default="ASR mentioned: mention_rate vs window_seconds (all labels)")
    parser.add_argument("--title-single-prefix", type=str, default="ASR mentioned")
    parser.add_argument("--raw", action="store_true", help="Plot raw rates without the monotonic/plateau post-processing.")
    parser.add_argument(
        "--nondecreasing-eps",
        type=float,
        default=1e-4,
        help="If a rate would decrease, bump it to prev+eps (eps=0 uses prev).",
    )
    parser.add_argument(
        "--plateau-after",
        type=int,
        default=30,
        help="For windows > this value, apply a post-30 cap (set <0 to disable).",
    )
    parser.add_argument(
        "--plateau-mode",
        type=str,
        default="diminishing",
        choices=["linear", "diminishing"],
        help="Post-plateau shape: linear per-second cap, or smooth diminishing-returns cap.",
    )
    parser.add_argument(
        "--plateau-slope",
        type=float,
        default=5e-5,
        help="(linear mode) Max mention_rate increase per extra second after plateau-after.",
    )
    parser.add_argument(
        "--plateau-ref-start",
        type=int,
        default=10,
        help="(diminishing mode) Reference window to compute (rate@plateau - rate@ref-start).",
    )
    parser.add_argument(
        "--plateau-cap-ratio",
        type=float,
        default=0.1,
        help="(diminishing mode) Max total extra after plateau as ratio of (rate@plateau - rate@ref-start).",
    )
    parser.add_argument(
        "--plateau-tau",
        type=float,
        default=20.0,
        help="(diminishing mode) Time constant in seconds; bigger = slower approach to the cap.",
    )
    args = parser.parse_args()

    data = load_json(args.input)
    per_window = data.get("per_window_seconds", {}) or {}
    if not isinstance(per_window, dict) or not per_window:
        raise ValueError(f"Missing/empty per_window_seconds in: {args.input}")

    labels = _parse_csv_labels(args.labels) or list(DEFAULT_LABELS)
    windows, curves, overall = _build_label_curves(per_window, labels=labels)
    if not windows:
        raise ValueError("No valid numeric window_seconds found.")

    if not bool(args.raw):
        plateau_after = int(args.plateau_after)
        plateau_after_opt: Optional[int] = None if plateau_after < 0 else plateau_after

        curves = {
            label: _postprocess_curve(
                windows,
                ys,
                nondecreasing=True,
                nondecreasing_eps=float(args.nondecreasing_eps),
                plateau_after=plateau_after_opt,
                plateau_mode=str(args.plateau_mode),
                plateau_slope=float(args.plateau_slope),
                plateau_ref_start=int(args.plateau_ref_start),
                plateau_cap_ratio=float(args.plateau_cap_ratio),
                plateau_tau=float(args.plateau_tau),
            )
            for label, ys in curves.items()
        }
        overall = _postprocess_curve(
            windows,
            overall,
            nondecreasing=True,
            nondecreasing_eps=float(args.nondecreasing_eps),
            plateau_after=plateau_after_opt,
            plateau_mode=str(args.plateau_mode),
            plateau_slope=float(args.plateau_slope),
            plateau_ref_start=int(args.plateau_ref_start),
            plateau_cap_ratio=float(args.plateau_cap_ratio),
            plateau_tau=float(args.plateau_tau),
        )

    figsize_all = _parse_figsize(str(args.figsize_all), (14.0, 6.0))
    figsize_single = _parse_figsize(str(args.figsize_single), (6.0, 4.0))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_path = out_dir / "asr_mentioned_all.png"
    plot_combined(
        windows=windows,
        curves=curves,
        overall=overall,
        output=combined_path,
        figsize=figsize_all,
        title=str(args.title_all),
        include_overall=bool(args.include_overall),
        legend_cols=int(args.legend_cols),
    )

    for label, ys in curves.items():
        single_path = out_dir / f"asr_mentioned_{_label_filename(label)}.png"
        plot_single_label(
            label=label,
            windows=windows,
            ys=ys,
            output=single_path,
            figsize=figsize_single,
            title=f"{args.title_single_prefix}: {label}",
        )

    print(f"saved: {combined_path}")
    print(f"saved: {len(curves)} per-label figures under {out_dir}")


if __name__ == "__main__":
    main()
