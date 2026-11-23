#!/usr/bin/env python3
"""
Tag bracketed person names in SoccerNet ASR annotations.

For every processed annotation JSON under annos/{train,valid,test}, each segment contains
`corrected_name_text`, where person entities are written as bracketed spans, e.g.:

  "... [Ivanovic B.] ... [Mourinho J.] ... [Oliver M.] ..."

This script adds a new sibling field (default: `corrected_name_text_tagged`) where:
  - Players become:  Home_Player_{shirt_number} / Away_Player_{shirt_number}
  - Coaches become:  coach
  - Referees become: referee
  - Brackets are removed in all cases.

It edits files in-place (atomic write). Use --dry-run to preview without writing.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm


BRACKET_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def _normalize_key(text: str) -> str:
    """
    Normalize a human name (or alias) into a robust matching key.
    - casefold
    - strip diacritics
    - remove apostrophes
    - collapse punctuation/whitespace into single spaces
    """
    text = unicodedata.normalize("NFKD", text)
    out_chars: List[str] = []
    for ch in text:
        if unicodedata.combining(ch):
            continue
        if ch in {"’", "'"}:
            continue
        if ch.isalnum():
            out_chars.append(ch.casefold())
            continue
        if ch in {"-", "_"}:
            out_chars.append(" ")
            continue
        if ch.isspace():
            out_chars.append(" ")
            continue
        out_chars.append(" ")
    key = "".join(out_chars)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _iter_name_variants(obj: Any) -> Iterable[str]:
    if not isinstance(obj, dict):
        return
    for k in ("name", "short_name", "long_name"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            yield v.strip()


def _safe_shirt_number(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if s.isdigit():
        return str(int(s))
    return s


def _build_name_to_tokens(anno: Dict[str, Any]) -> Dict[str, Set[str]]:
    name_to_tokens: DefaultDict[str, Set[str]] = defaultdict(set)

    def add(name: str, token: str) -> None:
        key = _normalize_key(name)
        if key:
            name_to_tokens[key].add(token)

    lineup = anno.get("lineup")
    if isinstance(lineup, dict):
        for side, prefix in (("home", "Home_Player_"), ("away", "Away_Player_")):
            team = lineup.get(side)
            if not isinstance(team, dict):
                continue

            players = team.get("players") or []
            if isinstance(players, list):
                for p in players:
                    if not isinstance(p, dict):
                        continue
                    num = _safe_shirt_number(p.get("shirt_number"))
                    token = f"{prefix}{num}" if num else f"{prefix}UNK"
                    for n in _iter_name_variants(p):
                        add(n, token)

            coaches = team.get("coach") or []
            if isinstance(coaches, list):
                for c in coaches:
                    if not isinstance(c, dict):
                        continue
                    for n in _iter_name_variants(c):
                        add(n, "coach")

    refs = anno.get("referee")
    if isinstance(refs, str):
        add(refs, "referee")
    elif isinstance(refs, list):
        for r in refs:
            if isinstance(r, str) and r.strip():
                add(r.strip(), "referee")

    return dict(name_to_tokens)


def _resolve_token(name: str, name_to_tokens: Dict[str, Set[str]]) -> Optional[str]:
    raw = name.strip()
    if not raw:
        return None

    lowered = raw.casefold()
    if lowered in {"coach", "referee"}:
        return lowered

    key = _normalize_key(raw)
    tokens = name_to_tokens.get(key)
    if not tokens:
        return None
    if len(tokens) == 1:
        return next(iter(tokens))
    return None


def tag_corrected_name_text(
    text: str,
    name_to_tokens: Dict[str, Set[str]],
    unknown_names: Optional[List[str]] = None,
) -> Tuple[str, int]:
    if not text:
        return text, 0

    replaced = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal replaced
        inner = m.group(1).strip()
        token = _resolve_token(inner, name_to_tokens)
        if token is None:
            if unknown_names is not None and inner:
                unknown_names.append(inner)
            return inner  # drop brackets, keep original content
        replaced += 1
        return token

    out = BRACKET_PATTERN.sub(repl, text)
    return out, replaced


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        prefix=path.name + ".tmp.",
    ) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        tmp_path = Path(f.name)
    tmp_path.replace(path)


def _iter_anno_files(annos_root: Path, splits: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for split in splits:
        split_dir = annos_root / split
        if not split_dir.exists():
            continue
        for p in split_dir.rglob("*.json"):
            name = p.name
            if name.endswith("_processed.json") or name.endswith("_processed_clean.json"):
                out.append(p)
    return sorted(out)


def process_one_file(
    path: Path,
    field_in: str,
    field_out: str,
    overwrite: bool,
    dry_run: bool,
    verbose_unknown: bool,
) -> Dict[str, Any]:
    anno = json.loads(path.read_text(encoding="utf-8"))
    name_to_tokens = _build_name_to_tokens(anno)

    segs = anno.get("segments") or []
    if not isinstance(segs, list) or not segs:
        return {"path": str(path), "skipped": True, "reason": "no_segments"}

    unknown: List[str] = []
    replaced_total = 0
    modified = False
    segments_touched = 0

    for seg in segs:
        if not isinstance(seg, dict):
            continue
        if field_in not in seg:
            continue
        if (field_out in seg) and (not overwrite):
            continue

        raw = seg.get(field_in)
        text = raw if isinstance(raw, str) else ""
        tagged, replaced = tag_corrected_name_text(text, name_to_tokens, unknown_names=unknown)

        seg[field_out] = tagged
        replaced_total += replaced
        modified = True
        segments_touched += 1

    if modified and not dry_run:
        _write_json_atomic(path, anno)

    result: Dict[str, Any] = {
        "path": str(path),
        "skipped": False,
        "segments_touched": segments_touched,
        "replaced": replaced_total,
        "unknown": len(unknown),
    }
    if verbose_unknown and unknown:
        # Dedup while preserving order
        seen: Set[str] = set()
        uniq: List[str] = []
        for u in unknown:
            k = _normalize_key(u)
            if k and k not in seen:
                seen.add(k)
                uniq.append(u)
        result["unknown_examples"] = uniq[:50]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag bracketed person names in corrected_name_text into Home/Away_Player_{num}, coach, referee."
    )
    parser.add_argument(
        "--anno-file",
        type=Path,
        default=None,
        help="Process ONE annotation JSON file (in-place).",
    )
    parser.add_argument(
        "--annos-root",
        type=Path,
        default=Path("/path/to/SoccerNet/asr/annos"),
        help="Process ALL annotation JSON files under this root (train/valid/test).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
        help="Which splits under annos-root to process.",
    )
    parser.add_argument(
        "--field-in",
        type=str,
        default="corrected_name_text",
        help="Input text field in each segment.",
    )
    parser.add_argument(
        "--field-out",
        type=str,
        default="corrected_name_text_tagged",
        help="Output field to write in each segment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite field-out if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only report stats.",
    )
    parser.add_argument(
        "--try-one",
        action="store_true",
        help="When using --annos-root: only process the first file (quick sanity check).",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="When using --annos-root: limit number of files processed.",
    )
    parser.add_argument(
        "--verbose-unknown",
        action="store_true",
        help="Print some unknown bracket names (up to 50 per file in the final summary).",
    )
    args = parser.parse_args()

    if args.anno_file is not None:
        stats = process_one_file(
            args.anno_file,
            field_in=args.field_in,
            field_out=args.field_out,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            verbose_unknown=args.verbose_unknown,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    files = _iter_anno_files(args.annos_root, args.splits)
    if args.try_one:
        files = files[:1]
    if args.limit_files is not None:
        files = files[: max(0, args.limit_files)]

    total_files = len(files)
    total_segments_touched = 0
    total_replaced = 0
    total_unknown = 0

    unknown_files: List[Dict[str, Any]] = []

    for p in tqdm(files, desc="Tagging annos"):
        stats = process_one_file(
            p,
            field_in=args.field_in,
            field_out=args.field_out,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            verbose_unknown=args.verbose_unknown,
        )
        if stats.get("skipped"):
            continue
        total_segments_touched += int(stats.get("segments_touched", 0))
        total_replaced += int(stats.get("replaced", 0))
        total_unknown += int(stats.get("unknown", 0))
        if args.verbose_unknown and stats.get("unknown", 0):
            unknown_files.append(stats)

    summary = {
        "files": total_files,
        "segments_touched": total_segments_touched,
        "replaced": total_replaced,
        "unknown": total_unknown,
        "dry_run": bool(args.dry_run),
        "field_in": args.field_in,
        "field_out": args.field_out,
        "overwrite": bool(args.overwrite),
    }
    if args.verbose_unknown and unknown_files:
        summary["unknown_files"] = unknown_files[:20]

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

