#!/usr/bin/env python3
"""
Collect video paths from the BBC News subset and export them to JSON.

The output JSON mirrors the structure of dataset/avcaption/filtered_videos.json
but retains only the `video_path` field for downstream consumption.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

DEFAULT_SOURCE_DIR = Path(
    os.environ.get("BBC_NEWS_SOURCE_DIR", "dataset/selected_100k/bbc_news")
)
DEFAULT_OUTPUT_PATH = Path(
    os.environ.get("BBC_NEWS_OUTPUT_PATH", "dataset/avcaption/bbc_news_video_paths.json")
)
VIDEO_EXTENSIONS = {".mp4"}


def find_videos(source_dir: Path) -> Iterable[Path]:
    """Yield all candidate video files under `source_dir`."""
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def build_manifest(source_dir: Path) -> List[dict[str, str]]:
    """Create the JSON-ready manifest with only `video_path` entries."""
    entries: List[dict[str, str]] = []

    for video_file in sorted(find_videos(source_dir), key=lambda p: p.as_posix()):
        try:
            rel_path = video_file.relative_to(source_dir)
        except ValueError:
            rel_path = video_file
        entries.append({"video_path": rel_path.as_posix()})

    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract BBC News video paths into a JSON manifest."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Directory to scan for videos (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination JSON file (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir: Path = args.source_dir.expanduser().resolve()
    output_path: Path = args.output.expanduser()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    manifest = build_manifest(source_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
        fp.write("\n")

    print(f"Wrote {len(manifest)} entries to {output_path}")


if __name__ == "__main__":
    main()
