#!/usr/bin/env python3
"""
Step 1: Download YouTube Videos for Lip Reading Dataset
========================================================
Usage:
    python 01_download_videos.py --input data/links/video_links.csv --output_dir data/raw_videos/

Input CSV format (data/links/video_links.csv):
    speaker_id,youtube_url,speaker_name
    spk_001,https://www.youtube.com/watch?v=XXXXX,John_Doe
    spk_002,https://www.youtube.com/watch?v=YYYYY,Jane_Smith

Requirements:
    pip install yt-dlp imageio-ffmpeg opencv-python

Notes:
    - Downloads at 720p (sufficient for lip reading, 25fps)
    - Extracts audio separately as WAV (16kHz mono)
    - Idempotent: re-runs only missing/broken artifacts
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = "data/links/video_links.csv"
DEFAULT_OUTPUT_DIR = "data/raw_videos"


def _is_valid_file(path: str, min_size_bytes: int = 1024) -> bool:
    return os.path.exists(path) and os.path.getsize(path) >= min_size_bytes


def _resolve_local_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def _resolve_ffmpeg_bin() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _ensure_ffmpeg_shim(ffmpeg_bin: str) -> tuple[str, str]:
    """Create a stable `ffmpeg` command path for tools that expect that exact name."""
    if ffmpeg_bin == "ffmpeg":
        return "", "ffmpeg"

    shim_dir = Path.home() / ".cache" / "lipsynth" / "bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_ffmpeg = shim_dir / "ffmpeg"

    if shim_ffmpeg.exists() or shim_ffmpeg.is_symlink():
        try:
            if shim_ffmpeg.resolve() != Path(ffmpeg_bin):
                shim_ffmpeg.unlink()
                shim_ffmpeg.symlink_to(ffmpeg_bin)
        except Exception:
            shim_ffmpeg.unlink()
            shim_ffmpeg.symlink_to(ffmpeg_bin)
    else:
        shim_ffmpeg.symlink_to(ffmpeg_bin)

    return str(shim_dir), str(shim_ffmpeg)


def _get_video_duration_seconds(video_path: str) -> float:
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        if fps <= 0:
            return 0.0
        return float(frames / fps)
    except Exception:
        return 0.0


def download_video(url: str, output_dir: str, speaker_id: str, ffmpeg_cmd: str, ffmpeg_location: str) -> dict:
    """Download/recover one speaker's source video and audio."""

    video_dir = os.path.join(output_dir, speaker_id)
    os.makedirs(video_dir, exist_ok=True)

    video_path = os.path.join(video_dir, "full_video.mp4")
    audio_path = os.path.join(video_dir, "full_audio.wav")

    video_ok = _is_valid_file(video_path)
    audio_ok = _is_valid_file(audio_path)

    if video_ok and audio_ok:
        print(f"  [SKIP] {speaker_id} already complete")
        return {"status": "skipped", "speaker_id": speaker_id}

    if not video_ok:
        video_cmd = [
            "yt-dlp",
            "-f",
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]",
            "--merge-output-format",
            "mp4",
            "--ffmpeg-location",
            ffmpeg_location,
            "--output",
            video_path,
            "--no-playlist",
            "--retries",
            "3",
            url,
        ]

        print(f"  Downloading video for {speaker_id}...")
        result = subprocess.run(video_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = (result.stderr or result.stdout)[-500:]
            print(f"  [ERROR] Video download failed for {speaker_id}: {err[:200]}")
            return {"status": "error", "speaker_id": speaker_id, "error": err}

        video_ok = _is_valid_file(video_path)
        if not video_ok:
            return {
                "status": "error",
                "speaker_id": speaker_id,
                "error": "video_download_incomplete",
            }

    # Always (re)extract audio when missing/broken.
    if not audio_ok:
        audio_cmd = [
            ffmpeg_cmd,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_path,
        ]

        print(f"  Extracting audio for {speaker_id}...")
        result = subprocess.run(audio_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = (result.stderr or result.stdout)[-500:]
            print(f"  [ERROR] Audio extraction failed for {speaker_id}: {err[:200]}")
            return {"status": "error", "speaker_id": speaker_id, "error": err}

        audio_ok = _is_valid_file(audio_path)
        if not audio_ok:
            return {
                "status": "error",
                "speaker_id": speaker_id,
                "error": "audio_extraction_incomplete",
            }

    duration = _get_video_duration_seconds(video_path)
    print(f"  [OK] {speaker_id}: {duration:.1f}s ready")
    return {"status": "ok", "speaker_id": speaker_id, "duration": duration}


def main():
    parser = argparse.ArgumentParser(description="Download YouTube videos for lip reading dataset")
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="CSV file with video links")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    input_csv = _resolve_local_path(args.input)
    output_dir = _resolve_local_path(args.output_dir)

    if not input_csv.exists():
        print(f"[ERROR] Input CSV not found: {input_csv}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    ffmpeg_bin = _resolve_ffmpeg_bin()
    ffmpeg_shim_dir, ffmpeg_cmd = _ensure_ffmpeg_shim(ffmpeg_bin)
    if ffmpeg_shim_dir:
        os.environ["PATH"] = ffmpeg_shim_dir + os.pathsep + os.environ.get("PATH", "")
    ffmpeg_location = ffmpeg_shim_dir or str(Path(ffmpeg_bin).parent)
    print(f"Using ffmpeg: {ffmpeg_cmd}")
    print(f"Using input CSV: {input_csv}")
    print(f"Using output dir: {output_dir}")

    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        videos = list(reader)

    print(f"Found {len(videos)} videos to download/reconcile\n")

    results = []
    total_duration = 0.0

    for i, video in enumerate(videos, 1):
        speaker_id = video["speaker_id"]
        print(f"[{i}/{len(videos)}] Processing {speaker_id}...")
        result = download_video(
            url=video["youtube_url"],
            output_dir=str(output_dir),
            speaker_id=speaker_id,
            ffmpeg_cmd=ffmpeg_cmd,
            ffmpeg_location=ffmpeg_location,
        )
        results.append(result)
        if result.get("duration"):
            total_duration += float(result["duration"])

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")

    expected_speakers = [v["speaker_id"] for v in videos]
    missing = []
    for speaker_id in expected_speakers:
        speaker_dir = output_dir / speaker_id
        video_path = speaker_dir / "full_video.mp4"
        audio_path = speaker_dir / "full_audio.wav"
        if not (_is_valid_file(video_path) and _is_valid_file(audio_path)):
            missing.append(speaker_id)

    print(f"\n{'=' * 50}")
    print("Download Summary:")
    print(f"  Success: {ok}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")
    print(f"  Total duration observed: {total_duration / 3600:.1f} hours")
    if missing:
        print(f"  Missing complete artifacts for: {missing}")
    print(f"{'=' * 50}")

    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
