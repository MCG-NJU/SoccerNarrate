#!/usr/bin/env python3

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SOCCERNET_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SPLIT_ROOT = SCRIPT_DIR
DEFAULT_VIDEOS_ROOT = SOCCERNET_ROOT / "videos"
DEFAULT_OUTPUT_ROOT = SOCCERNET_ROOT / "split_videos"


@dataclass
class Stats:
    created_links: int = 0
    kept_links: int = 0
    skipped_links: int = 0
    missing_games: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create split-aware SoccerNet directories backed by symlinks so videos "
            "do not need to be duplicated."
        )
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=DEFAULT_SPLIT_ROOT,
        help=(
            "Directory containing split folders such as train/valid/test. "
            f"Default: {DEFAULT_SPLIT_ROOT}"
        ),
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=DEFAULT_VIDEOS_ROOT,
        help=(
            "Directory containing the original SoccerNet videos organized as "
            "<league>/<season>/<game>. "
            f"Default: {DEFAULT_VIDEOS_ROOT}"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Directory where split-aware folders will be created. "
            f"Default: {DEFAULT_OUTPUT_ROOT}"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Split names to materialize. Default: train valid test",
    )
    parser.add_argument(
        "--label-filename",
        default="Labels-v2.json",
        help="Label file name used to enumerate games. Default: Labels-v2.json",
    )
    parser.add_argument(
        "--link-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to also symlink the split label file into each output game "
            "directory. Enabled by default."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing symlinks/files when they point somewhere else.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without writing symlinks.",
    )
    return parser.parse_args()


def game_dirs_from_split(split_dir: Path, label_filename: str) -> list[Path]:
    return sorted(label_path.parent for label_path in split_dir.rglob(label_filename))


def ensure_symlink(
    source: Path,
    destination: Path,
    *,
    force: bool,
    dry_run: bool,
) -> str:
    if destination.is_symlink():
        if destination.resolve(strict=False) == source.resolve(strict=False):
            return "kept"
        if not force:
            return "skipped"
        if not dry_run:
            destination.unlink()
    elif destination.exists():
        if destination.is_file() and force:
            if not dry_run:
                destination.unlink()
        else:
            return "skipped"

    if dry_run:
        return "created"

    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(source, start=destination.parent)
    destination.symlink_to(relative_target, target_is_directory=source.is_dir())
    return "created"


def update_stats(status: str, stats: Stats) -> None:
    if status == "created":
        stats.created_links += 1
    elif status == "kept":
        stats.kept_links += 1
    else:
        stats.skipped_links += 1


def materialize_split(
    split: str,
    split_root: Path,
    videos_root: Path,
    output_root: Path,
    label_filename: str,
    *,
    link_labels: bool,
    force: bool,
    dry_run: bool,
    stats: Stats,
) -> None:
    split_dir = split_root / split
    output_split_dir = output_root / split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    if not split_dir.exists():
        print(f"[warn] split directory not found: {split_dir}")
        return

    game_dirs = game_dirs_from_split(split_dir, label_filename)
    print(f"[info] split={split} games={len(game_dirs)}")

    for game_dir in game_dirs:
        relative_game_dir = game_dir.relative_to(split_dir)
        source_game_dir = videos_root / relative_game_dir
        destination_game_dir = output_split_dir / relative_game_dir
        destination_game_dir.mkdir(parents=True, exist_ok=True)

        if not source_game_dir.is_dir():
            stats.missing_games += 1
            print(f"[warn] source game directory missing: {source_game_dir}")
            continue

        for source_entry in sorted(source_game_dir.iterdir()):
            destination_entry = destination_game_dir / source_entry.name
            status = ensure_symlink(
                source_entry,
                destination_entry,
                force=force,
                dry_run=dry_run,
            )
            update_stats(status, stats)

        if link_labels:
            label_path = game_dir / label_filename
            destination_label = destination_game_dir / label_filename
            status = ensure_symlink(
                label_path,
                destination_label,
                force=force,
                dry_run=dry_run,
            )
            update_stats(status, stats)


def main() -> None:
    args = parse_args()

    split_root = args.split_root.resolve()
    videos_root = args.videos_root.resolve()
    output_root = args.output_root.resolve()

    print(f"[info] split_root={split_root}")
    print(f"[info] videos_root={videos_root}")
    print(f"[info] output_root={output_root}")
    print(f"[info] splits={', '.join(args.splits)}")
    if args.dry_run:
        print("[info] dry-run mode enabled")

    stats = Stats()

    for split in args.splits:
        materialize_split(
            split=split,
            split_root=split_root,
            videos_root=videos_root,
            output_root=output_root,
            label_filename=args.label_filename,
            link_labels=args.link_labels,
            force=args.force,
            dry_run=args.dry_run,
            stats=stats,
        )

    print("[done] summary")
    print(f"  created_links={stats.created_links}")
    print(f"  kept_links={stats.kept_links}")
    print(f"  skipped_links={stats.skipped_links}")
    print(f"  missing_games={stats.missing_games}")


if __name__ == "__main__":
    main()
