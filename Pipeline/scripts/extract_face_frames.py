#!/usr/bin/env python3
"""Extract one or more face-conditioning frames per clip for Stage 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def find_pipe_root(start: Path) -> Path:
    for cand in [start.resolve(), *start.resolve().parents]:
        if (cand / "pyproject.toml").exists() and (cand / "third_party" / "LipVoicer").exists():
            return cand
    raise FileNotFoundError("Could not locate the Pipeline root from the current working directory.")


PIPE_ROOT = find_pipe_root(Path(__file__).resolve())
PROJECT_ROOT = PIPE_ROOT.parent


def discover_data_root(pipe_root: Path) -> Path:
    candidates = [
        pipe_root / "data" / "custom_data",
        PROJECT_ROOT / "data" / "custom_data",
    ]
    for candidate in candidates:
        if (candidate / "dataset_final" / "train.tsv").exists():
            return candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DATA_ROOT = discover_data_root(PIPE_ROOT)
MANIFEST_DIR = DATA_ROOT / "dataset_final"
VIDEO_DIR = MANIFEST_DIR / "videos"
SEGMENTS_DIR = DATA_ROOT / "segments"
FACE_DIR = DATA_ROOT / "faces"


def clip_video_path(clip_id: str, speaker_id: str) -> Path:
    candidates = [
        VIDEO_DIR / f"{clip_id}.mp4",
        SEGMENTS_DIR / speaker_id / f"{clip_id}.mp4",
    ]
    return next((p for p in candidates if p.exists()), candidates[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--frames-per-clip", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--face-dir", type=Path, default=FACE_DIR)
    return parser.parse_args()


def read_manifests(manifest_dir: Path, split: str) -> pd.DataFrame:
    splits = ("train", "val", "test") if split == "all" else (split,)
    frames = [pd.read_csv(manifest_dir / f"{name}.tsv", sep="\t") for name in splits]
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["clip_id", "speaker_id"])


def sample_frame_indices(frame_count: int, frames_per_clip: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frames_per_clip <= 1:
        return [max(0, frame_count // 2)]
    if frame_count == 1:
        return [0]
    positions = []
    for idx in range(frames_per_clip):
        frac = (idx + 1) / (frames_per_clip + 1)
        positions.append(min(frame_count - 1, max(0, int(round(frac * (frame_count - 1))))))
    deduped = []
    for pos in positions:
        if pos not in deduped:
            deduped.append(pos)
    return deduped


def extract_frames(video_path: Path, output_prefix: Path, frames_per_clip: int, force: bool) -> bool:
    legacy_path = output_prefix.with_name(f"{output_prefix.name}.jpg")
    multi_paths = [output_prefix.with_name(f"{output_prefix.name}_{idx}.jpg") for idx in range(frames_per_clip)]
    if not force and legacy_path.exists() and all(path.exists() for path in multi_paths):
        return True

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(frame_count, frames_per_clip)
    if not indices:
        cap.release()
        return False

    frames = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        return False

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    center_frame = frames[len(frames) // 2]
    center_frame.save(str(legacy_path))
    for idx, frame in enumerate(frames):
        frame.save(str(output_prefix.with_name(f"{output_prefix.name}_{idx}.jpg")))
    return True


def main() -> None:
    args = parse_args()
    args.face_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifests(args.manifest_dir, args.split)

    generated = 0
    skipped = 0
    failed = 0
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="face-frames"):
        clip_id = str(row["clip_id"]).strip()
        speaker_id = str(row["speaker_id"]).strip()
        video_path = clip_video_path(clip_id, speaker_id)
        if not video_path.exists():
            failed += 1
            continue
        prefix = args.face_dir / f"{clip_id}_face"
        legacy_path = args.face_dir / f"{clip_id}_face.jpg"
        multi_paths = [args.face_dir / f"{clip_id}_face_{idx}.jpg" for idx in range(args.frames_per_clip)]
        if not args.force and legacy_path.exists() and all(path.exists() for path in multi_paths):
            skipped += 1
            continue
        if extract_frames(video_path, prefix, args.frames_per_clip, args.force):
            generated += 1
        else:
            failed += 1

    print(f"Generated:{generated}  Skipped:{skipped}  Failed:{failed}")


if __name__ == "__main__":
    main()
