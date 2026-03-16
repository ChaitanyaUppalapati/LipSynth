#!/usr/bin/env python3
"""
Step 4: Finalize Dataset — Quality Checks, Splits, Manifest
==============================================================
Usage:
    python 04_finalize_dataset.py \
        --segments_dir segments/ \
        --lip_rois_dir lip_rois/ \
        --output_dir dataset_final/

This script:
    1. Cross-references segments and lip ROIs
    2. Filters out clips that failed any stage
    3. Creates train/val/test splits (speaker-disjoint!)
    4. Generates manifest files compatible with LipVoicer
    5. Computes and reports dataset statistics

Output structure (LipVoicer-compatible):
    dataset_final/
    ├── train.tsv          # manifest: clip_id \t text \t speaker_id \t duration
    ├── val.tsv
    ├── test.tsv
    ├── videos/            # symlinks or copies of clip .mp4 files
    ├── audios/            # symlinks or copies of clip .wav files
    ├── mouths/            # symlinks or copies of mouth .npz files
    ├── transcripts/       # .txt files
    └── dataset_stats.json
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


# Split ratios (by speaker count, speaker-disjoint)
TRAIN_RATIO = 0.80   # 40 speakers
VAL_RATIO = 0.10     # 5 speakers
TEST_RATIO = 0.10    # 5 speakers

random.seed(42)


def gather_valid_clips(segments_dir: str, lip_rois_dir: str) -> list:
    """Find clips that have all required components."""
    
    valid_clips = []
    
    # Iterate through speakers
    for speaker_id in sorted(os.listdir(segments_dir)):
        seg_speaker_dir = os.path.join(segments_dir, speaker_id)
        roi_speaker_dir = os.path.join(lip_rois_dir, speaker_id)
        
        if not os.path.isdir(seg_speaker_dir):
            continue
        
        # Find all clip IDs in segments
        clip_files = [f for f in os.listdir(seg_speaker_dir) if f.endswith(".mp4")]
        
        for clip_file in sorted(clip_files):
            clip_id = Path(clip_file).stem
            
            # Check all required files exist
            video_path = os.path.join(seg_speaker_dir, f"{clip_id}.mp4")
            audio_path = os.path.join(seg_speaker_dir, f"{clip_id}.wav")
            text_path = os.path.join(seg_speaker_dir, f"{clip_id}.txt")
            meta_path = os.path.join(seg_speaker_dir, f"{clip_id}.json")
            roi_path = os.path.join(roi_speaker_dir, f"{clip_id}.npz")
            
            if not all(os.path.exists(p) for p in [video_path, audio_path, text_path, roi_path]):
                continue
            
            # Read transcript
            with open(text_path, "r") as f:
                text = f.read().strip()
            
            # Read metadata if available
            metadata = {}
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
            
            valid_clips.append({
                "clip_id": clip_id,
                "speaker_id": speaker_id,
                "text": text,
                "duration": metadata.get("duration", 0),
                "num_words": metadata.get("num_words", len(text.split())),
                "video_path": video_path,
                "audio_path": audio_path,
                "text_path": text_path,
                "roi_path": roi_path,
            })
    
    return valid_clips


def find_incomplete_clips(segments_dir: str, lip_rois_dir: str) -> list:
    """Return clips that are missing required artifacts."""
    incomplete = []
    for speaker_id in sorted(os.listdir(segments_dir)):
        seg_speaker_dir = os.path.join(segments_dir, speaker_id)
        roi_speaker_dir = os.path.join(lip_rois_dir, speaker_id)
        if not os.path.isdir(seg_speaker_dir):
            continue

        clip_files = [f for f in os.listdir(seg_speaker_dir) if f.endswith(".mp4")]
        for clip_file in sorted(clip_files):
            clip_id = Path(clip_file).stem
            expected = {
                "video": os.path.join(seg_speaker_dir, f"{clip_id}.mp4"),
                "audio": os.path.join(seg_speaker_dir, f"{clip_id}.wav"),
                "text": os.path.join(seg_speaker_dir, f"{clip_id}.txt"),
                "roi": os.path.join(roi_speaker_dir, f"{clip_id}.npz"),
            }
            missing = [name for name, p in expected.items() if not os.path.exists(p)]
            if missing:
                incomplete.append({"clip_id": clip_id, "speaker_id": speaker_id, "missing": missing})
    return incomplete


def create_splits(clips: list) -> dict:
    """Create speaker-disjoint train/val/test splits."""
    
    # Group by speaker
    speaker_clips = defaultdict(list)
    for clip in clips:
        speaker_clips[clip["speaker_id"]].append(clip)
    
    speakers = sorted(speaker_clips.keys())
    random.shuffle(speakers)
    
    n_speakers = len(speakers)
    n_val = max(1, int(n_speakers * VAL_RATIO))
    n_test = max(1, int(n_speakers * TEST_RATIO))
    n_train = n_speakers - n_val - n_test
    
    train_speakers = speakers[:n_train]
    val_speakers = speakers[n_train:n_train + n_val]
    test_speakers = speakers[n_train + n_val:]
    
    splits = {
        "train": [],
        "val": [],
        "test": []
    }
    
    for spk in train_speakers:
        splits["train"].extend(speaker_clips[spk])
    for spk in val_speakers:
        splits["val"].extend(speaker_clips[spk])
    for spk in test_speakers:
        splits["test"].extend(speaker_clips[spk])
    
    print(f"  Split by speaker:")
    print(f"    Train: {n_train} speakers, {len(splits['train'])} clips")
    print(f"    Val:   {n_val} speakers, {len(splits['val'])} clips")
    print(f"    Test:  {n_test} speakers, {len(splits['test'])} clips")
    
    return splits


def write_manifest(clips: list, output_path: str):
    """Write a TSV manifest file."""
    with open(output_path, "w") as f:
        # Header
        f.write("clip_id\ttext\tspeaker_id\tduration\tnum_words\n")
        for clip in sorted(clips, key=lambda x: x["clip_id"]):
            f.write(f"{clip['clip_id']}\t{clip['text']}\t{clip['speaker_id']}\t"
                    f"{clip['duration']:.2f}\t{clip['num_words']}\n")


def organize_files(clips: list, output_dir: str, use_symlinks: bool = True):
    """Copy or symlink files into the final dataset structure."""
    
    dirs = ["videos", "audios", "mouths", "transcripts"]
    for d in dirs:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)
    
    link_fn = os.symlink if use_symlinks else shutil.copy2
    
    for clip in clips:
        clip_id = clip["clip_id"]
        
        targets = [
            (clip["video_path"], os.path.join(output_dir, "videos", f"{clip_id}.mp4")),
            (clip["audio_path"], os.path.join(output_dir, "audios", f"{clip_id}.wav")),
            (clip["roi_path"],   os.path.join(output_dir, "mouths", f"{clip_id}.npz")),
            (clip["text_path"],  os.path.join(output_dir, "transcripts", f"{clip_id}.txt")),
        ]
        
        for src, dst in targets:
            if not os.path.exists(dst):
                try:
                    if use_symlinks:
                        link_fn(os.path.abspath(src), dst)
                    else:
                        link_fn(src, dst)
                except OSError:
                    # Fallback to copy if symlinks fail
                    shutil.copy2(src, dst)


def compute_stats(splits: dict) -> dict:
    """Compute dataset statistics."""
    stats = {}
    
    for split_name, clips in splits.items():
        durations = [c["duration"] for c in clips]
        word_counts = [c["num_words"] for c in clips]
        speakers = set(c["speaker_id"] for c in clips)
        
        stats[split_name] = {
            "num_clips": len(clips),
            "num_speakers": len(speakers),
            "total_duration_hours": round(sum(durations) / 3600, 3),
            "avg_duration_sec": round(sum(durations) / max(len(durations), 1), 2),
            "min_duration_sec": round(min(durations) if durations else 0, 2),
            "max_duration_sec": round(max(durations) if durations else 0, 2),
            "total_words": sum(word_counts),
            "avg_words_per_clip": round(sum(word_counts) / max(len(word_counts), 1), 1),
            "speakers": sorted(list(speakers))
        }
    
    # Overall
    all_clips = sum(len(v) for v in splits.values())
    all_duration = sum(s["total_duration_hours"] for s in stats.values())
    all_speakers = set()
    for s in stats.values():
        all_speakers.update(s["speakers"])
    
    stats["overall"] = {
        "total_clips": all_clips,
        "total_speakers": len(all_speakers),
        "total_duration_hours": round(all_duration, 3),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Finalize dataset with splits and manifests")
    parser.add_argument("--segments_dir", default="segments", help="Segmented clips directory")
    parser.add_argument("--lip_rois_dir", default="lip_rois", help="Lip ROI directory")
    parser.add_argument("--output_dir", default="dataset_final", help="Final dataset directory")
    parser.add_argument("--copy_files", action="store_true", help="Copy files instead of symlinks")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    print("Checking for missing artifacts...")
    incomplete = find_incomplete_clips(args.segments_dir, args.lip_rois_dir)
    if incomplete:
        print(f"  Found {len(incomplete)} incomplete clips before finalization.")
        print("  These clips will be excluded from the final dataset.")
        print(f"  Example: {incomplete[:5]}")

    # Step 1: Gather valid clips
    print("Gathering valid clips...")
    valid_clips = gather_valid_clips(args.segments_dir, args.lip_rois_dir)
    print(f"  Found {len(valid_clips)} valid clips\n")

    if not valid_clips:
        print("ERROR: No valid clips found! Check your pipeline outputs.")
        sys.exit(1)
    
    # Step 2: Create splits
    print("Creating speaker-disjoint splits...")
    splits = create_splits(valid_clips)
    
    # Step 3: Write manifests
    print("\nWriting manifest files...")
    for split_name, clips in splits.items():
        manifest_path = os.path.join(args.output_dir, f"{split_name}.tsv")
        write_manifest(clips, manifest_path)
        print(f"  {manifest_path}")
    
    # Step 4: Organize files
    print("\nOrganizing files...")
    all_clips = []
    for clips in splits.values():
        all_clips.extend(clips)
    organize_files(all_clips, args.output_dir, use_symlinks=not args.copy_files)
    print(f"  Files {'copied' if args.copy_files else 'symlinked'} to {args.output_dir}/")
    
    # Step 5: Statistics
    stats = compute_stats(splits)
    stats_path = os.path.join(args.output_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total clips:    {stats['overall']['total_clips']}")
    print(f"  Total speakers: {stats['overall']['total_speakers']}")
    print(f"  Total duration: {stats['overall']['total_duration_hours']:.2f} hours")
    print(f"")
    for split_name in ["train", "val", "test"]:
        s = stats[split_name]
        print(f"  {split_name:5s}: {s['num_clips']:5d} clips | "
              f"{s['num_speakers']:2d} speakers | "
              f"{s['total_duration_hours']:.2f}h | "
              f"avg {s['avg_duration_sec']:.1f}s/clip")
    print(f"{'='*60}")
    print(f"\nDataset ready at: {args.output_dir}/")
    print(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
