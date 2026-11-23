#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 SoccerNet 的 Labels-v2.json 事件标注追加两个字段：
  - mentioned: bool
  - happened: bool

字段值参考 jsonl 标注文件（每行一个事件），例如：
  /path/to/codes/soccer-ac/text_cls/stat/event_mentions8_15.jsonl

匹配键（足够唯一，且与生成 jsonl 的事件信息一致）：
  (league_season, match, half, game_time, event_label, position_ms)

不会修改原始标注目录；会在 annos 下新建一个目录写入结果（只写 Labels-v2.json，不复制同目录下的 txt 文件）。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


EventKey = Tuple[str, str, int, str, str, int]


def parse_half_from_game_time(game_time: str) -> Optional[int]:
    # "1 - 13:10" -> 1
    if not game_time:
        return None
    if "-" not in game_time:
        return None
    left = game_time.split("-", 1)[0].strip()
    try:
        half = int(left)
    except Exception:
        return None
    if half not in (1, 2):
        return None
    return half


def to_position_ms(value) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if not s.isdigit():
        return None
    try:
        return int(s)
    except Exception:
        return None


def make_key(
    league_season: str,
    match: str,
    half: int,
    game_time: str,
    event_label: str,
    position_ms: int,
) -> EventKey:
    return (
        (league_season or "").strip(),
        (match or "").strip(),
        int(half),
        (game_time or "").strip(),
        (event_label or "").strip(),
        int(position_ms),
    )


def load_judge_map(jsonl_path: Path) -> Dict[EventKey, Tuple[bool, bool]]:
    """
    返回 {EventKey: (mentioned_bool, happened_bool)}
    """
    mapping: Dict[EventKey, Tuple[bool, bool]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ev = row.get("event") or {}
            judge = row.get("judge") or {}

            league_season = (ev.get("league_season") or "").strip()
            match = (ev.get("match") or "").strip()
            half = ev.get("half")
            game_time = (ev.get("game_time") or "").strip()
            event_label = (ev.get("event_label") or "").strip()
            position_sec = ev.get("position_sec")

            if not league_season or not match or not game_time or not event_label or half is None or position_sec is None:
                raise ValueError(f"jsonl line {line_idx}: missing event fields: {ev}")

            try:
                pos_ms = int(round(float(position_sec) * 1000.0))
            except Exception as e:
                raise ValueError(f"jsonl line {line_idx}: bad position_sec={position_sec}") from e

            mentioned_str = (judge.get("mentioned") or "").strip()
            happened_str = (judge.get("happened") or "").strip()
            if mentioned_str not in ("mentioned", "not_mentioned") or happened_str not in ("happened", "not_happened"):
                raise ValueError(f"jsonl line {line_idx}: bad judge fields: {judge}")

            mentioned = mentioned_str == "mentioned"
            happened = happened_str == "happened"

            key = make_key(league_season, match, int(half), game_time, event_label, pos_ms)
            if key in mapping:
                raise ValueError(f"jsonl line {line_idx}: duplicate key: {key}")
            mapping[key] = (mentioned, happened)

    return mapping


def iter_label_files(raw_root: Path) -> Iterable[Path]:
    yield from raw_root.rglob("Labels-v2.json")


def process_one_label_file(
    label_file: Path,
    *,
    raw_root: Path,
    dst_raw_root: Path,
    mapping: Dict[EventKey, Tuple[bool, bool]],
    missing_policy: str,
    overwrite: bool,
    missing_events_out: Optional[List[dict]],
    stats: Counter,
) -> None:
    rel = label_file.relative_to(raw_root)
    # raw/{league}/{season}/{match}/Labels-v2.json
    if len(rel.parts) < 4:
        raise ValueError(f"unexpected label path under raw root: {label_file}")
    league, season, match = rel.parts[0], rel.parts[1], rel.parts[2]
    league_season = f"{league}_{season}"

    dst_match_dir = dst_raw_root / rel.parent
    dst_label_file = dst_raw_root / rel

    if dst_label_file.exists() and not overwrite:
        raise FileExistsError(f"destination exists: {dst_label_file} (use --overwrite)")

    data = json.loads(label_file.read_text(encoding="utf-8"))
    annotations = data.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError(f"{label_file}: annotations is not a list")

    for ann in annotations:
        stats["total_events"] += 1
        if not isinstance(ann, dict):
            stats["bad_event"] += 1
            continue

        event_label = (ann.get("label") or "").strip()
        game_time = (ann.get("gameTime") or "").strip()
        half = parse_half_from_game_time(game_time)
        pos_ms = to_position_ms(ann.get("position"))

        if half is None or pos_ms is None or not event_label or not game_time:
            stats["bad_event_fields"] += 1
            if missing_policy == "error":
                raise ValueError(f"{label_file}: bad event fields: {ann}")
            ann["mentioned"] = False
            ann["happened"] = False
            continue

        key = make_key(league_season, match, half, game_time, event_label, pos_ms)
        hit = mapping.get(key)
        if hit is None:
            stats["missing_events"] += 1
            if missing_events_out is not None:
                missing_events_out.append(
                    {
                        "league_season": league_season,
                        "match": match,
                        "half": half,
                        "game_time": game_time,
                        "event_label": event_label,
                        "position_ms": pos_ms,
                        "label_file": str(label_file),
                        "visibility": ann.get("visibility"),
                        "team": ann.get("team"),
                    }
                )
            if missing_policy == "error":
                raise KeyError(f"missing in jsonl mapping: {key}")
            ann["mentioned"] = False
            ann["happened"] = False
            continue

        mentioned, happened = hit
        ann["mentioned"] = bool(mentioned)
        ann["happened"] = bool(happened)
        stats["matched_events"] += 1

    dst_match_dir.mkdir(parents=True, exist_ok=True)
    dst_label_file.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    stats["processed_label_files"] += 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="为 SoccerNet Labels-v2.json 的每个事件写入 mentioned/happened 字段（来自 jsonl）。"
    )
    parser.add_argument(
        "--mentions-jsonl",
        type=Path,
        default=Path("./text_cls/stat/event_mentions8_15.jsonl"),
        help="参考标注 jsonl（每行一个事件，包含 judge.mentioned/happened）。",
    )
    parser.add_argument(
        "--src-raw",
        type=Path,
        default=Path("/path/to/SoccerNet/SoccerNet-label/raw_annos"),
        help="原始 raw 标注目录（不会修改）。",
    )
    parser.add_argument(
        "--dst-raw",
        type=Path,
        default=Path("/path/to/SoccerNet/SoccerNet-label/annos_with_mentioned_happened"),
        help="输出目录（将创建新目录并写入修改后的 Labels-v2.json）。",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["error", "false"],
        default="false",
        help="jsonl 找不到对应事件时：error=直接报错；false=填充为 False 并继续。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖 dst-raw 下已有文件（不删除目录结构，只覆盖同名文件）。",
    )
    parser.add_argument(
        "--write-missing-jsonl",
        type=Path,
        default=None,
        help="可选：把缺失事件写到一个 jsonl（便于后续排查/补标）。",
    )
    parser.add_argument(
        "--write-summary-json",
        type=Path,
        default=None,
        help="可选：把统计信息写到 json 文件。",
    )
    args = parser.parse_args()

    if not args.mentions_jsonl.exists():
        raise FileNotFoundError(f"mentions jsonl not found: {args.mentions_jsonl}")
    if not args.src_raw.exists():
        raise FileNotFoundError(f"src raw not found: {args.src_raw}")

    if args.dst_raw.exists() and not args.overwrite:
        # 避免误覆盖
        raise FileExistsError(f"dst raw already exists: {args.dst_raw} (use --overwrite or change --dst-raw)")

    print(f"[1/3] loading mapping: {args.mentions_jsonl}")
    mapping = load_judge_map(args.mentions_jsonl)
    print(f"loaded {len(mapping):,} events from jsonl")

    print(f"[2/3] scanning label files: {args.src_raw}")
    label_files = sorted(iter_label_files(args.src_raw))
    print(f"found {len(label_files):,} Labels-v2.json files")

    missing_events_out: Optional[List[dict]] = [] if args.write_missing_jsonl else None
    stats: Counter = Counter()
    missing_matches: Counter = Counter()

    for idx, label_file in enumerate(label_files, start=1):
        try:
            process_one_label_file(
                label_file,
                raw_root=args.src_raw,
                dst_raw_root=args.dst_raw,
                mapping=mapping,
                missing_policy=args.missing_policy,
                overwrite=args.overwrite,
                missing_events_out=missing_events_out,
                stats=stats,
            )
        except KeyError:
            # 统计一下缺失匹配主要来自哪些比赛
            rel = label_file.relative_to(args.src_raw)
            if len(rel.parts) >= 3:
                league, season, match = rel.parts[0], rel.parts[1], rel.parts[2]
                missing_matches[f"{league}_{season}|{match}"] += 1
            raise

        if idx % 50 == 0:
            print(f"  processed {idx}/{len(label_files)} files...", flush=True)

    print(f"[3/3] done. dst={args.dst_raw}")
    print(
        "stats:",
        f"files={stats['processed_label_files']:,}",
        f"total_events={stats['total_events']:,}",
        f"matched={stats['matched_events']:,}",
        f"missing={stats['missing_events']:,}",
        f"bad_fields={stats['bad_event_fields']:,}",
        sep=" ",
    )

    if args.write_missing_jsonl and missing_events_out is not None:
        args.write_missing_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.write_missing_jsonl.open("w", encoding="utf-8") as f:
            for row in missing_events_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote missing events: {args.write_missing_jsonl} ({len(missing_events_out):,} lines)")

    if args.write_summary_json:
        args.write_summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "mentions_jsonl": str(args.mentions_jsonl),
            "src_raw": str(args.src_raw),
            "dst_raw": str(args.dst_raw),
            "missing_policy": args.missing_policy,
            "overwrite": bool(args.overwrite),
            "stats": dict(stats),
            "missing_matches_top20": missing_matches.most_common(20),
        }
        args.write_summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote summary: {args.write_summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
