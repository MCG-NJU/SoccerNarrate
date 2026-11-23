#!/usr/bin/env python3
"""Download official SoccerNet full-match 720p videos and Labels-v2.json annotations.

Requires the official `SoccerNet` package (`pip install SoccerNet`).
To download 720p full-match videos, you must sign the SoccerNet NDA to obtain a password:
https://www.soccer-net.org/data
"""

import argparse
import os
from pathlib import Path
from SoccerNet.Downloader import SoccerNetDownloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SoccerNet 720p videos and/or Labels-v2.json annotations."
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("/path/to/SoccerNet/videos"),
        help="Target directory for downloaded SoccerNet files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test", "challenge"],
        help="SoccerNet splits to download.",
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Download 1_720p.mkv and 2_720p.mkv (requires NDA password via --password or SOCCERNET_PASSWORD).",
    )
    parser.add_argument(
        "--download-labels",
        action="store_true",
        help="Download Labels-v2.json action spotting event annotations.",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.environ.get("SOCCERNET_PASSWORD", ""),
        help="SoccerNet NDA password for 720p video downloads (or set SOCCERNET_PASSWORD env var).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.download_videos and not args.download_labels:
        raise SystemExit("Specify at least one of --download-videos or --download-labels.")

    args.local_dir.mkdir(parents=True, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=str(args.local_dir))

    files = []
    if args.download_labels:
        files.append("Labels-v2.json")
    if args.download_videos:
        if not args.password:
            raise SystemExit(
                "720p video download requires the SoccerNet NDA password "
                "(--password or SOCCERNET_PASSWORD). See https://www.soccer-net.org/data."
            )
        downloader.password = args.password
        files.extend(["1_720p.mkv", "2_720p.mkv"])

    downloader.downloadGames(files=files, split=args.splits)


if __name__ == "__main__":
    main()
