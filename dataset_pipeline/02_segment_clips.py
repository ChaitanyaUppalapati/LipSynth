#!/usr/bin/env python3
"""
Step 2: Segment Videos into Sentence-Level Clips
==================================================
Usage:
    python 02_segment_clips.py --input_dir raw_videos/ --output_dir segments/

This script:
    1. Uses Whisper to transcribe and get word-level timestamps
    2. Segments videos into 2-10 second clips aligned to sentence boundaries
    3. Outputs: video clip (.mp4) + audio clip (.wav) + transcript (.txt) + metadata (.json)
    4. Is idempotent and resumes missing clip artifacts only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MIN_CLIP_DURATION = 2.0
MAX_CLIP_DURATION = 10.0
TARGET_FPS = 25
MAX_CHARS = 100

_WHISPER_MODELS = {}


def _is_valid_file(path: str, min_size_bytes: int = 256) -> bool:
    return os.path.exists(path) and os.path.getsize(path) >= min_size_bytes


def _clip_complete(speaker_output: str, clip_id: str) -> bool:
    checks = [
        (os.path.join(speaker_output, f"{clip_id}.mp4"), 1024),
        (os.path.join(speaker_output, f"{clip_id}.wav"), 1024),
        (os.path.join(speaker_output, f"{clip_id}.txt"), 1),
        (os.path.join(speaker_output, f"{clip_id}.json"), 16),
    ]
    return all(_is_valid_file(path, min_size) for path, min_size in checks)


def _resolve_ffmpeg_bin() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _ensure_ffmpeg_shim(ffmpeg_bin: str) -> tuple[str, str]:
    """Create a stable `ffmpeg` command path for tools that require that exact name."""
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


def transcribe_with_whisper(audio_path: str, model_size: str = "base", device: str = "auto") -> dict:
    """Transcribe audio using Whisper with word-level timestamps."""
    import whisper
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _WHISPER_MODELS.get((model_size, device))
    if model is None:
        print(f"    Loading Whisper model: {model_size} (device={device})")
        model = whisper.load_model(model_size, device=device)
        _WHISPER_MODELS[(model_size, device)] = model

    print("    Transcribing audio...")
    return model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
        verbose=False,
    )


def get_sentence_segments(whisper_result: dict) -> list:
    """Extract sentence-level segments from Whisper output."""
    segments = []

    for seg in whisper_result.get("segments", []):
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        duration = end - start

        if duration < MIN_CLIP_DURATION:
            continue

        if duration > MAX_CLIP_DURATION:
            words = seg.get("words", [])
            if words:
                segments.extend(split_long_segment(words))
            continue

        if len(text) > MAX_CHARS:
            continue

        if len(text.split()) < 2:
            continue

        segments.append(
            {
                "start": start,
                "end": end,
                "duration": duration,
                "text": text,
                "words": seg.get("words", []),
            }
        )

    return segments


def split_long_segment(words: list) -> list:
    """Split long segment into natural chunks."""
    segments = []
    current_words = []
    current_start = None

    for word in words:
        if current_start is None:
            current_start = word["start"]

        current_words.append(word)
        current_duration = word["end"] - current_start
        current_text = " ".join(w["word"].strip() for w in current_words).strip()

        should_split = False
        token = word["word"].strip()
        if any(token.endswith(p) for p in [".", "!", "?", ","]) and current_duration >= MIN_CLIP_DURATION:
            should_split = True
        if current_duration >= MAX_CLIP_DURATION * 0.9:
            should_split = True

        if should_split and len(current_text) <= MAX_CHARS:
            segments.append(
                {
                    "start": current_start,
                    "end": word["end"],
                    "duration": word["end"] - current_start,
                    "text": current_text,
                    "words": current_words.copy(),
                }
            )
            current_words = []
            current_start = None

    if current_words and current_start is not None:
        end = current_words[-1]["end"]
        duration = end - current_start
        text = " ".join(w["word"].strip() for w in current_words).strip()
        if MIN_CLIP_DURATION <= duration <= MAX_CLIP_DURATION and len(text) <= MAX_CHARS and len(text.split()) >= 2:
            segments.append(
                {
                    "start": current_start,
                    "end": end,
                    "duration": duration,
                    "text": text,
                    "words": current_words.copy(),
                }
            )

    return segments


def extract_clip(
    video_path: str,
    audio_path: str,
    segment: dict,
    output_video: str,
    output_audio: str,
    ffmpeg_bin: str,
) -> bool:
    """Extract one video+audio clip."""
    start = segment["start"]
    duration = segment["duration"]

    video_cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        str(start),
        "-i",
        video_path,
        "-t",
        str(duration),
        "-vf",
        f"fps={TARGET_FPS},scale=-1:720",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        output_video,
    ]
    result = subprocess.run(video_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False

    audio_cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        str(start),
        "-i",
        audio_path,
        "-t",
        str(duration),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_audio,
    ]
    result = subprocess.run(audio_cmd, capture_output=True, text=True)
    return result.returncode == 0


def process_speaker(speaker_dir: str, output_dir: str, whisper_model: str, ffmpeg_bin: str, device: str = "auto") -> dict:
    """Process one speaker idempotently and verify completeness."""
    speaker_id = os.path.basename(speaker_dir)
    video_path = os.path.join(speaker_dir, "full_video.mp4")
    audio_path = os.path.join(speaker_dir, "full_audio.wav")

    if not (_is_valid_file(video_path) and _is_valid_file(audio_path)):
        return {
            "status": "error",
            "speaker_id": speaker_id,
            "clips": 0,
            "duration": 0.0,
            "reason": "missing_raw_video_or_audio",
        }

    speaker_output = os.path.join(output_dir, speaker_id)
    os.makedirs(speaker_output, exist_ok=True)
    transcript_path = os.path.join(speaker_output, "full_transcript.json")

    if _is_valid_file(transcript_path):
        with open(transcript_path, "r") as f:
            whisper_result = json.load(f)
    else:
        whisper_result = transcribe_with_whisper(audio_path, whisper_model, device=device)
        with open(transcript_path, "w") as f:
            json.dump(whisper_result, f, indent=2)

    segments = get_sentence_segments(whisper_result)
    print(f"    Expected segments: {len(segments)}")
    if not segments:
        return {
            "status": "error",
            "speaker_id": speaker_id,
            "clips": 0,
            "duration": 0.0,
            "reason": "no_valid_segments",
        }

    clip_count = 0
    skipped_existing = 0
    total_duration = 0.0
    missing_ids = []

    for i, segment in enumerate(segments):
        clip_id = f"{speaker_id}_{i + 1:04d}"
        output_video = os.path.join(speaker_output, f"{clip_id}.mp4")
        output_audio = os.path.join(speaker_output, f"{clip_id}.wav")
        output_text = os.path.join(speaker_output, f"{clip_id}.txt")
        output_meta = os.path.join(speaker_output, f"{clip_id}.json")

        if _clip_complete(speaker_output, clip_id):
            clip_count += 1
            skipped_existing += 1
            total_duration += segment["duration"]
            continue

        success = extract_clip(video_path, audio_path, segment, output_video, output_audio, ffmpeg_bin)
        if not success:
            missing_ids.append(clip_id)
            continue

        with open(output_text, "w") as f:
            f.write(segment["text"])

        metadata = {
            "clip_id": clip_id,
            "speaker_id": speaker_id,
            "start": segment["start"],
            "end": segment["end"],
            "duration": segment["duration"],
            "text": segment["text"],
            "num_words": len(segment["text"].split()),
            "num_chars": len(segment["text"]),
            "words": segment.get("words", []),
        }
        with open(output_meta, "w") as f:
            json.dump(metadata, f, indent=2)

        if _clip_complete(speaker_output, clip_id):
            clip_count += 1
            total_duration += segment["duration"]
        else:
            missing_ids.append(clip_id)

    # Final completeness check against expected clip IDs.
    expected_ids = [f"{speaker_id}_{i + 1:04d}" for i in range(len(segments))]
    unresolved = [cid for cid in expected_ids if not _clip_complete(speaker_output, cid)]
    if unresolved:
        missing_ids = sorted(set(missing_ids + unresolved))
        return {
            "status": "error",
            "speaker_id": speaker_id,
            "clips": clip_count,
            "duration": total_duration,
            "reason": "incomplete_clip_artifacts",
            "missing_clip_ids": missing_ids,
        }

    print(f"    Extracted/verified {clip_count} clips ({skipped_existing} already complete)")
    return {
        "status": "ok",
        "speaker_id": speaker_id,
        "clips": clip_count,
        "duration": total_duration,
        "skipped_existing": skipped_existing,
    }


def main():
    parser = argparse.ArgumentParser(description="Segment videos into sentence-level clips")
    parser.add_argument("--input_dir", default="raw_videos", help="Directory with raw videos")
    parser.add_argument("--output_dir", default="segments", help="Output directory for clips")
    parser.add_argument(
        "--whisper_model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (larger = more accurate but slower)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use for Whisper (auto, cpu, cuda, cuda:0, etc.)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ffmpeg_bin = _resolve_ffmpeg_bin()
    ffmpeg_shim_dir, ffmpeg_cmd = _ensure_ffmpeg_shim(ffmpeg_bin)
    if ffmpeg_shim_dir:
        os.environ["PATH"] = ffmpeg_shim_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"Using ffmpeg: {ffmpeg_cmd}")

    speaker_dirs = sorted(
        [
            os.path.join(args.input_dir, d)
            for d in os.listdir(args.input_dir)
            if os.path.isdir(os.path.join(args.input_dir, d))
        ]
    )
    print(f"Found {len(speaker_dirs)} speakers to process\n")

    if not speaker_dirs:
        print("ERROR: No speaker folders found in input_dir.")
        sys.exit(1)

    all_results = []
    for i, speaker_dir in enumerate(speaker_dirs, 1):
        speaker_id = os.path.basename(speaker_dir)
        print(f"[{i}/{len(speaker_dirs)}] Processing {speaker_id}...")
        try:
            result = process_speaker(
            speaker_dir,
            args.output_dir,
            args.whisper_model,
            ffmpeg_cmd,
            device=args.device,
        )
        except Exception as e:
            result = {"status": "error", "speaker_id": speaker_id, "clips": 0, "duration": 0.0, "reason": str(e)}
        all_results.append(result)

    total_clips = sum(r.get("clips", 0) for r in all_results)
    total_duration = sum(float(r.get("duration", 0.0)) for r in all_results)
    failures = [r for r in all_results if r.get("status") != "ok"]

    print(f"\n{'=' * 50}")
    print("Segmentation Summary:")
    print(f"  Total speakers: {len(all_results)}")
    print(f"  Total clips: {total_clips}")
    print(f"  Total duration: {total_duration / 3600:.2f} hours")
    if failures:
        print(f"  Failed speakers: {[r['speaker_id'] for r in failures]}")
    print(f"{'=' * 50}")

    summary_path = os.path.join(args.output_dir, "segmentation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total_speakers": len(all_results),
                "total_clips": total_clips,
                "total_duration_hours": round(total_duration / 3600, 2),
                "failures": failures,
                "speakers": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to {summary_path}")

    if failures or total_clips == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
