#!/usr/bin/env python3
"""Fetch videos from a YouTube playlist and build/update data/links/video_links.csv."""

import argparse
import csv
import sys
from pathlib import Path

from utils import (
    get_playlist_video_id,
    get_video_description,
    load_env,
    parse_csv_lines,
    query_openrouter,
)

DEFAULT_PLAYLIST_ID = "PLsRNoUx8w3rPxNGCQYBPobGxNj1BfDT7P"
DEFAULT_NUM_VIDEOS = 3
DEFAULT_OUTPUT = "data/links/video_links.csv"


def _resolve_output_path(output: str) -> Path:
    p = Path(output)
    if p.is_absolute():
        return p
    # Keep output anchored to dataset_pipeline/ by default for consistent pipeline behavior.
    return (Path(__file__).resolve().parent / p).resolve()


def _fetch_videos(playlist_id: str, num_videos: int) -> list[tuple[int, str, str]]:
    videos = []
    for i in range(1, num_videos + 1):
        print(f"Fetching {i}/{num_videos}...")
        vid = get_playlist_video_id(playlist_id, i)
        if not vid:
            continue
        desc = get_video_description(vid)[:1000]
        videos.append((i, vid, desc))
    return videos


def _infer_speakers(videos: list[tuple[int, str, str]]) -> dict:
    if not videos:
        return {}

    prompt = "Extract FULL speaker names (first and last name). Output CSV with index,speaker_name:\\n"
    prompt += "\\n".join(f"{i},{d[:400]}" for i, _, d in videos)
    prompt += "\\nReturn FULL names only, no abbreviations."

    resp = query_openrouter(prompt)
    if not resp:
        return {}
    speakers = parse_csv_lines(resp)
    print(f"LLM returned {len(speakers)} names")
    return speakers


def _write_csv(videos: list[tuple[int, str, str]], speakers: dict, playlist_id: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "youtube_url", "speaker_name"])
        for i, vid, _ in videos:
            name = speakers.get(i, f"Speaker_{i:03d}")
            url = f"https://www.youtube.com/watch?v={vid}&list={playlist_id}&index={i}"
            w.writerow([f"spk_{i:03d}", url, name])
            print(f"  {i}: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch playlist entries into data/links/video_links.csv"
    )
    parser.add_argument("--playlist_id", default=DEFAULT_PLAYLIST_ID, help="YouTube playlist ID")
    parser.add_argument("--num_videos", type=int, default=DEFAULT_NUM_VIDEOS, help="Number of videos to fetch")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    args = parser.parse_args()

    load_env()
    output_path = _resolve_output_path(args.output)

    videos = _fetch_videos(args.playlist_id, args.num_videos)
    if not videos:
        print("ERROR: No videos fetched from playlist.")
        sys.exit(1)

    speakers = _infer_speakers(videos)
    _write_csv(videos, speakers, args.playlist_id, output_path)

    print(f"Saved {len(videos)} rows to {output_path}")
    if len(videos) < args.num_videos:
        print(f"WARNING: Requested {args.num_videos} videos but fetched {len(videos)}.")


if __name__ == "__main__":
    main()
