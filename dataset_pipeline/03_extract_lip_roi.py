#!/usr/bin/env python3
"""
Step 3: Face Detection + Lip Region-of-Interest (ROI) Extraction
=================================================================
Usage:
    python 03_extract_lip_roi.py --input_dir segments/ --output_dir lip_rois/

This script:
    1. Detects faces and landmarks using a PyTorch model (face-alignment)
    2. Extracts mouth/lip landmarks
    3. Crops lip region (96x96 grayscale - LipVoicer standard)
    4. Validates single-speaker constraint
    5. Filters out bad clips (no face, multiple faces, occlusions)

Requirements:
    pip install face-alignment torch opencv-python numpy

Output structure:
    lip_rois/
    ├── spk_001/
    │   ├── spk_001_0001/
    │   │   ├── mouth_000.png    # 96x96 grayscale mouth crop per frame
    │   │   ├── mouth_001.png
    │   │   └── ...
    │   ├── spk_001_0001.npz     # numpy array of all frames (T, 96, 96)
    │   └── ...
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# -- Configuration ------------------------------------------------------------
MOUTH_ROI_SIZE = 96          # LipVoicer standard: 96x96
MOUTH_ROI_PADDING = 1.5      # Padding factor around mouth landmarks
MIN_FRAMES_RATIO = 0.85      # At least 85% of frames must have a detected face
MOUTH_LM_START = 48          # 68-point landmarks: mouth starts at index 48
MOUTH_LM_END = 68            # 68-point landmarks: mouth ends at index 67


def _get_landmark_type(face_alignment):
    """Return 2D landmark enum across face-alignment versions."""
    if hasattr(face_alignment.LandmarksType, "TWO_D"):
        return face_alignment.LandmarksType.TWO_D
    return face_alignment.LandmarksType._2D


def init_face_detector(device: str = "auto") -> dict:
    """Initialize PyTorch face landmark detector."""
    try:
        import torch
        import face_alignment
    except ImportError as e:
        raise RuntimeError(
            "PyTorch face detector requires 'face-alignment' and 'torch'. "
            "Install with: pip install face-alignment torch"
        ) from e

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARNING] CUDA requested ('{device}') but not available. Falling back to CPU.")
        device = "cpu"

    landmark_type = _get_landmark_type(face_alignment)
    fa = face_alignment.FaceAlignment(
        landmark_type,
        flip_input=False,
        device=device,
    )
    return {"type": "pytorch", "detector": fa, "device": device}


def _predict_landmarks(detector, rgb_frame):
    """Handle API differences across face-alignment versions."""
    if hasattr(detector, "get_landmarks_from_image"):
        return detector.get_landmarks_from_image(rgb_frame)
    return detector.get_landmarks(rgb_frame)


def detect_mouth_pytorch(frame, detector):
    """Detect mouth region from PyTorch landmarks."""
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictions = _predict_landmarks(detector, rgb_frame)

    if not predictions:
        return None, 0

    num_faces = len(predictions)

    # Use first (most prominent) face.
    landmarks = np.asarray(predictions[0], dtype=np.float32)
    if landmarks.shape[0] < MOUTH_LM_END:
        return None, num_faces

    mouth_points = landmarks[MOUTH_LM_START:MOUTH_LM_END]

    x_min, y_min = mouth_points.min(axis=0)
    x_max, y_max = mouth_points.max(axis=0)

    mouth_w = x_max - x_min
    mouth_h = y_max - y_min
    cx = int((x_min + x_max) / 2.0)
    cy = int((y_min + y_max) / 2.0)

    side = max(1, int(max(mouth_w, mouth_h) * MOUTH_ROI_PADDING))

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    if x2 <= x1 or y2 <= y1:
        return None, num_faces

    return (x1, y1, x2, y2), num_faces


def extract_mouth_roi(frame, bbox):
    """Crop and resize mouth ROI to standard size."""
    x1, y1, x2, y2 = bbox
    mouth_crop = frame[y1:y2, x1:x2]

    if mouth_crop.size == 0:
        return None

    if len(mouth_crop.shape) == 3:
        mouth_crop = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)

    mouth_roi = cv2.resize(
        mouth_crop,
        (MOUTH_ROI_SIZE, MOUTH_ROI_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return mouth_roi


def _is_valid_npz(npz_path: str) -> bool:
    if not os.path.exists(npz_path) or os.path.getsize(npz_path) < 256:
        return False
    try:
        with np.load(npz_path) as data:
            arr = data.get("mouth_rois")
            return arr is not None and arr.ndim == 3 and arr.shape[1:] == (MOUTH_ROI_SIZE, MOUTH_ROI_SIZE)
    except Exception:
        return False


def process_clip(video_path: str, output_dir: str, detector: dict,
                 save_frames: bool = False) -> dict:
    """Process a single video clip: detect face, extract mouth ROI per frame."""
    clip_name = Path(video_path).stem
    clip_output_dir = os.path.join(output_dir, clip_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "reason": "cannot_open_video"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    mouth_rois = []
    detected_frames = 0
    multi_face_frames = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox, num_faces = detect_mouth_pytorch(frame, detector["detector"])

        if num_faces > 1:
            multi_face_frames += 1

        roi = None
        if bbox is not None:
            roi = extract_mouth_roi(frame, bbox)

        if roi is not None:
            mouth_rois.append(roi)
            detected_frames += 1
            if save_frames:
                os.makedirs(clip_output_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(clip_output_dir, f"mouth_{frame_idx:03d}.png"),
                    roi,
                )
        else:
            # Keep temporal alignment with original frame count.
            mouth_rois.append(np.zeros((MOUTH_ROI_SIZE, MOUTH_ROI_SIZE), dtype=np.uint8))

        frame_idx += 1

    cap.release()

    if total_frames == 0:
        return {"status": "error", "reason": "empty_video"}

    detection_ratio = detected_frames / total_frames
    multi_face_ratio = multi_face_frames / total_frames

    if detection_ratio < MIN_FRAMES_RATIO:
        return {
            "status": "rejected",
            "reason": f"low_detection_rate ({detection_ratio:.2f})",
            "detection_ratio": detection_ratio,
        }

    if multi_face_ratio > 0.3:
        return {
            "status": "rejected",
            "reason": f"multiple_faces ({multi_face_ratio:.2f})",
            "multi_face_ratio": multi_face_ratio,
        }

    mouth_array = np.stack(mouth_rois, axis=0)
    npz_path = os.path.join(output_dir, f"{clip_name}.npz")
    np.savez_compressed(npz_path, mouth_rois=mouth_array)

    return {
        "status": "ok",
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detection_ratio": round(detection_ratio, 3),
        "multi_face_ratio": round(multi_face_ratio, 3),
        "shape": mouth_array.shape,
        "fps": fps,
    }


def process_speaker(segments_dir: str, output_dir: str, detector: dict,
                    save_frames: bool = False) -> dict:
    """Process all clips for a speaker."""
    speaker_id = os.path.basename(segments_dir)
    speaker_output = os.path.join(output_dir, speaker_id)
    os.makedirs(speaker_output, exist_ok=True)

    video_files = sorted([f for f in os.listdir(segments_dir) if f.endswith(".mp4")])

    results = {"ok": 0, "rejected": 0, "error": 0, "skipped": 0}
    clip_results = []

    clip_iter = video_files
    if tqdm is not None:
        clip_iter = tqdm(video_files, desc=f"{speaker_id}", unit="clip", leave=False)

    for vf in clip_iter:
        clip_id = Path(vf).stem
        video_path = os.path.join(segments_dir, vf)
        npz_path = os.path.join(speaker_output, f"{clip_id}.npz")

        if _is_valid_npz(npz_path):
            clip_result = {"status": "ok", "reason": "already_complete", "clip_id": clip_id}
            clip_results.append(clip_result)
            results["ok"] += 1
            results["skipped"] += 1
            continue

        clip_result = process_clip(video_path, speaker_output, detector, save_frames)
        clip_result["clip_id"] = clip_id
        clip_results.append(clip_result)
        results[clip_result["status"]] = results.get(clip_result["status"], 0) + 1

    expected_ids = [Path(vf).stem for vf in video_files]
    status_by_clip = {r["clip_id"]: r.get("status", "error") for r in clip_results}
    missing_ids = []
    for clip_id in expected_ids:
        # Rejected clips are intentionally filtered and may not have ROI outputs.
        if status_by_clip.get(clip_id) == "rejected":
            continue
        if not _is_valid_npz(os.path.join(speaker_output, f"{clip_id}.npz")):
            missing_ids.append(clip_id)

    if missing_ids and results.get("error", 0) == 0:
        results["error"] = len(missing_ids)

    status = "ok" if not missing_ids else "error"

    report_path = os.path.join(speaker_output, "extraction_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "speaker_id": speaker_id,
                "status": status,
                "summary": results,
                "missing_clip_ids": missing_ids,
                "clips": clip_results,
            },
            f,
            indent=2,
        )

    return {"speaker_id": speaker_id, "status": status, "missing_clip_ids": missing_ids, **results, "total": len(video_files)}


def main():
    parser = argparse.ArgumentParser(description="Extract lip ROIs from video clips")
    parser.add_argument("--input_dir", default="segments", help="Directory with segmented clips")
    parser.add_argument("--output_dir", default="lip_rois", help="Output directory for mouth ROIs")
    parser.add_argument("--save_frames", action="store_true", help="Save individual frame PNGs")
    parser.add_argument("--detector", default="pytorch", choices=["pytorch"], help="Face detector backend")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, cuda, cuda:0, ...")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    detector = init_face_detector(device=args.device)
    print(f"Using {args.detector} face detector on device={detector['device']}")

    speaker_dirs = sorted([
        os.path.join(args.input_dir, d)
        for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])

    print(f"Found {len(speaker_dirs)} speakers to process\n")

    all_results = []
    speaker_iter = speaker_dirs
    if tqdm is not None:
        speaker_iter = tqdm(speaker_dirs, desc="Step 3 speakers", unit="speaker")

    for i, speaker_dir in enumerate(speaker_iter, 1):
        speaker_id = os.path.basename(speaker_dir)
        print(f"[{i}/{len(speaker_dirs)}] Processing {speaker_id}...")
        try:
            result = process_speaker(speaker_dir, args.output_dir, detector, args.save_frames)
            print(f"    OK: {result['ok']}, Rejected: {result['rejected']}, Error: {result['error']}")
            if hasattr(speaker_iter, "set_postfix_str"):
                speaker_iter.set_postfix_str(
                    f"ok={result['ok']} rejected={result['rejected']} error={result['error']}"
                )
            all_results.append(result)
        except Exception as e:
            print(f"    [ERROR] {e}")
            traceback.print_exc()
            all_results.append({"speaker_id": speaker_id, "status": "error", "ok": 0, "rejected": 0, "error": -1})

    total_ok = sum(r.get("ok", 0) for r in all_results)
    total_rejected = sum(r["rejected"] for r in all_results)
    total_error = sum(r["error"] for r in all_results if r["error"] > 0)
    failed_speakers = [r["speaker_id"] for r in all_results if r.get("status") == "error"]

    print(f"\n{'=' * 50}")
    print("Lip ROI Extraction Summary:")
    print(f"  Total speakers: {len(all_results)}")
    print(f"  Clips accepted: {total_ok}")
    print(f"  Clips rejected: {total_rejected}")
    print(f"  Clips errored:  {total_error}")
    print(f"  Accept rate: {total_ok / max(total_ok + total_rejected + total_error, 1) * 100:.1f}%")
    if failed_speakers:
        print(f"  Failed speakers: {failed_speakers}")
    print(f"{'=' * 50}")

    if failed_speakers or total_ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
