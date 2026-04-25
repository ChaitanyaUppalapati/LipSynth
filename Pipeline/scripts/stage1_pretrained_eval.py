#!/usr/bin/env python3
"""Evaluate Stage 1 with official LipVoicer lip-reading weights.

This script makes the pretrained LipVoicer visual speech recognizer the
primary Stage 1 baseline for this repository. It reads the repository's
speaker-disjoint manifests, loads pre-extracted mouth ROIs from `.npz`
artifacts, applies the official LipVoicer video transform, runs beam-search
decoding with the benchmark language model, and exports both evaluation
predictions and a Stage 2 handoff file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


REQUIRED_MANIFEST_COLUMNS = ("clip_id", "text", "speaker_id")
REQUIRED_BENCHMARK_KEYS = ("model_path", "model_conf", "rnnlm", "rnnlm_conf")


def find_pipe_root(start: Path) -> Path:
    for cand in [start.resolve(), *start.resolve().parents]:
        if (cand / "pyproject.toml").exists() and (cand / "third_party" / "LipVoicer").exists():
            return cand
    raise FileNotFoundError(
        "Could not locate the Pipeline root. Run this from the repository or pass explicit paths."
    )


PIPE_ROOT = find_pipe_root(Path(__file__).resolve())
PROJECT_ROOT = PIPE_ROOT.parent
LIPVOICER_ROOT = PIPE_ROOT / "third_party" / "LipVoicer"


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
DEFAULT_BENCHMARK_CONFIG = (
    LIPVOICER_ROOT / "mouthroi_processing" / "configs" / "LRS3_V_WER19.1.ini"
)
DEFAULT_OUTPUT_DIR = PIPE_ROOT / "outputs" / "stage1_eval" / "pretrained"
DEFAULT_BASELINE_PATH = PIPE_ROOT / "outputs" / "stage1_eval" / "val_predictions_stage1.csv"


@dataclass(frozen=True)
class BenchmarkAssets:
    lipvoicer_root: Path
    config_path: Path
    model_path: Path
    model_conf: Path
    rnnlm_path: Path
    rnnlm_conf: Path
    beam_size: int
    penalty: float
    ctc_weight: float
    lm_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 evaluation with the official LipVoicer lip-reading model."
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--manifest-tsv",
        type=Path,
        default=None,
        help="Optional manifest override. Defaults to data/dataset_final/<split>.tsv.",
    )
    parser.add_argument(
        "--lip-roi-root",
        type=Path,
        default=DATA_ROOT / "lip_rois",
        help="Root containing speaker subdirectories of mouth ROI .npz files.",
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=DEFAULT_BENCHMARK_CONFIG,
        help="LipVoicer benchmark config (.ini) that points at model + LM assets.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use. Defaults to auto-detecting cuda, otherwise cpu.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write predictions, summary, and Stage 2 handoff artifacts.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of manifest rows to evaluate.",
    )
    parser.add_argument(
        "--sample-clip-id",
        default=None,
        help="Only evaluate a single clip_id for smoke testing.",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Optional existing Stage 1 baseline CSV for comparison.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort immediately on the first malformed sample instead of collecting errors.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{requested_device}' but CUDA is not available in this environment."
        )
    return requested_device


def read_manifest(tsv_path: Path, max_samples: int | None, sample_clip_id: str | None) -> pd.DataFrame:
    if not tsv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t", engine="python")
    df.columns = [str(col).strip() for col in df.columns]
    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest {tsv_path} is missing required columns: {missing}")

    for column in REQUIRED_MANIFEST_COLUMNS:
        df[column] = df[column].astype(str).str.strip()
    df["text"] = df["text"].map(normalize_text)

    if sample_clip_id:
        df = df[df["clip_id"] == sample_clip_id].copy()
        if df.empty:
            raise ValueError(f"Clip '{sample_clip_id}' was not found in manifest: {tsv_path}")

    if max_samples is not None:
        df = df.head(max_samples).copy()

    if df.empty:
        raise ValueError(f"Manifest {tsv_path} produced no rows after filtering.")
    return df.reset_index(drop=True)


def _resolve_asset_path(lipvoicer_root: Path, raw_value: str) -> Path:
    raw_path = Path(raw_value)
    return raw_path if raw_path.is_absolute() else lipvoicer_root / raw_path


def load_benchmark_assets(config_path: Path) -> BenchmarkAssets:
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {config_path}")

    config = ConfigParser()
    config.read(config_path)
    lipvoicer_root = config_path.parents[2]

    missing_sections = [name for name in ("model", "decode") if not config.has_section(name)]
    if missing_sections:
        raise ValueError(f"Benchmark config {config_path} is missing sections: {missing_sections}")

    resolved: dict[str, Path] = {}
    missing_assets: list[str] = []
    for key in REQUIRED_BENCHMARK_KEYS:
        if not config.has_option("model", key):
            missing_assets.append(f"config:model.{key} (missing key)")
            continue
        resolved_path = _resolve_asset_path(lipvoicer_root, config.get("model", key))
        resolved[key] = resolved_path
        if not resolved_path.exists():
            missing_assets.append(f"config:model.{key} -> {resolved_path}")

    if missing_assets:
        joined = "\n".join(f"- {item}" for item in missing_assets)
        raise FileNotFoundError(
            "Official LipVoicer benchmark assets are missing.\n"
            "Expected the following files to exist:\n"
            f"{joined}\n"
            "Download the official checkpoints into third_party/LipVoicer/mouthroi_processing/benchmarks "
            "before running pretrained Stage 1 evaluation."
        )

    return BenchmarkAssets(
        lipvoicer_root=lipvoicer_root,
        config_path=config_path,
        model_path=resolved["model_path"],
        model_conf=resolved["model_conf"],
        rnnlm_path=resolved["rnnlm"],
        rnnlm_conf=resolved["rnnlm_conf"],
        beam_size=config.getint("decode", "beam_size", fallback=40),
        penalty=config.getfloat("decode", "penalty", fallback=0.0),
        ctc_weight=config.getfloat("decode", "ctc_weight", fallback=0.1),
        lm_weight=config.getfloat("decode", "lm_weight", fallback=0.3),
    )


def ensure_lipvoicer_imports(lipvoicer_root: Path) -> tuple[Any, Any, Any]:
    root_str = str(lipvoicer_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from mouthroi_processing.espnet.asr.asr_utils import add_results_to_json
    from mouthroi_processing.pipelines.data.transforms import VideoTransform
    from mouthroi_processing.pipelines.model import AVSR

    return AVSR, VideoTransform, add_results_to_json


def load_mouth_rois(npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"ROI file missing: {npz_path}")

    payload = np.load(npz_path)
    if "mouth_rois" not in payload.files:
        raise KeyError(f"'mouth_rois' key missing in {npz_path}")

    mouth_rois = payload["mouth_rois"]
    if mouth_rois.ndim != 3:
        raise ValueError(f"Expected mouth_rois to have shape (T,96,96), got {mouth_rois.shape} in {npz_path}")
    if mouth_rois.shape[1:] != (96, 96):
        raise ValueError(f"Expected mouth_rois to have shape (T,96,96), got {mouth_rois.shape} in {npz_path}")
    if mouth_rois.dtype != np.uint8:
        mouth_rois = mouth_rois.astype(np.uint8)
    return mouth_rois


class PretrainedStage1Runner:
    """Thin wrapper around LipVoicer's pretrained AVSR stack."""

    def __init__(self, assets: BenchmarkAssets, device: str) -> None:
        avsr_cls, video_transform_cls, add_results_to_json = ensure_lipvoicer_imports(
            assets.lipvoicer_root
        )
        self.assets = assets
        self.device = device
        self.video_transform = video_transform_cls(speed_rate=1)
        self.add_results_to_json = add_results_to_json
        self.decoder_name = "batch_beam_search_ctc_lm"
        self.confidence_type = "beam_score_proxy"
        self.model = avsr_cls(
            modality="video",
            model_path=str(assets.model_path),
            model_conf=str(assets.model_conf),
            rnnlm=str(assets.rnnlm_path),
            rnnlm_conf=str(assets.rnnlm_conf),
            penalty=assets.penalty,
            ctc_weight=assets.ctc_weight,
            lm_weight=assets.lm_weight,
            beam_size=assets.beam_size,
            device=device,
        )

    def preprocess(self, mouth_rois: np.ndarray) -> torch.Tensor:
        transformed = self.video_transform(torch.from_numpy(mouth_rois))
        expected = (1, mouth_rois.shape[0], 88, 88)
        if tuple(transformed.shape) != expected:
            raise ValueError(
                f"Unexpected transformed ROI shape. Expected {expected}, got {tuple(transformed.shape)}."
            )
        return transformed

    def _decode_hypotheses(self, hypotheses: list[Any]) -> tuple[str, float, float]:
        if not hypotheses:
            return "", 0.0, float("-inf")

        nbest_limit = min(5, len(hypotheses))
        nbest_dicts = [hyp.asdict() for hyp in hypotheses[:nbest_limit]]
        transcription = self.add_results_to_json(nbest_dicts[:1], self.model.token_list)
        transcription = transcription.replace("▁", " ").replace("<eos>", "").strip()
        transcription = normalize_text(transcription)

        scores = torch.tensor(
            [float(hyp.score) for hyp in hypotheses[:nbest_limit]],
            dtype=torch.float32,
        )
        proxy = float(torch.softmax(scores, dim=0)[0].item())
        return transcription, proxy, float(scores[0].item())

    @torch.inference_mode()
    def predict_from_npz(self, npz_path: Path) -> dict[str, Any]:
        mouth_rois = load_mouth_rois(npz_path)
        transformed = self.preprocess(mouth_rois).to(self.device)
        encoded = self.model.model.encode(transformed)
        hypotheses = self.model.beam_search(encoded)
        pred_text, sequence_confidence, beam_score = self._decode_hypotheses(hypotheses)
        return {
            "pred_text": pred_text,
            "sequence_confidence": sequence_confidence,
            "beam_score": beam_score,
            "raw_roi_shape": list(mouth_rois.shape),
            "transformed_roi_shape": list(transformed.shape),
        }


def evaluate_manifest(
    manifest: pd.DataFrame,
    lip_roi_root: Path,
    runner: PretrainedStage1Runner,
    fail_fast: bool,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for record in tqdm(manifest.to_dict(orient="records"), desc="Stage1 pretrained inference"):
        clip_id = record["clip_id"]
        speaker_id = record["speaker_id"]
        roi_path = lip_roi_root / speaker_id / f"{clip_id}.npz"

        try:
            prediction = runner.predict_from_npz(roi_path)
        except Exception as exc:  # noqa: BLE001 - want actionable sample errors
            error = {
                "clip_id": clip_id,
                "speaker_id": speaker_id,
                "roi_path": str(roi_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            if fail_fast:
                raise RuntimeError(json.dumps(error, indent=2)) from exc
            errors.append(error)
            continue

        gt_text = normalize_text(record["text"])
        pred_text = prediction["pred_text"]
        rows.append(
            {
                "clip_id": clip_id,
                "speaker_id": speaker_id,
                "roi_path": str(roi_path),
                "gt_text": gt_text,
                "pred_text": pred_text,
                "sequence_confidence": prediction["sequence_confidence"],
                "confidence_type": runner.confidence_type,
                "decoder_name": runner.decoder_name,
                "beam_score": prediction["beam_score"],
                "raw_roi_shape": prediction["raw_roi_shape"],
                "transformed_roi_shape": prediction["transformed_roi_shape"],
            }
        )

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        raise RuntimeError(
            "No predictions were produced. "
            "Check the collected errors or verify that benchmark assets are available."
        )
    return pred_df, errors


def decide_action(overall_wer: float) -> tuple[str, str]:
    if overall_wer <= 0.60:
        return "reuse_pretrained", "Zero-shot validation WER is strong enough to freeze Stage 1 for the initial Stage 2 handoff."
    if overall_wer <= 0.85:
        return "finetune", "Pretrained Stage 1 is partially useful, but should get a short sanity-run fine-tune before becoming the mainline model."
    return "retrain", "Pretrained Stage 1 is not strong enough on this domain, so the repo-native training path should become the primary Stage 1 track."


def compare_with_existing_baseline(baseline_path: Path, pred_df: pd.DataFrame) -> dict[str, Any] | None:
    if not baseline_path.exists():
        return None

    baseline_df = pd.read_csv(baseline_path)
    if not {"gt_text", "pred_text"}.issubset(set(baseline_df.columns)):
        return None

    baseline_gt = baseline_df["gt_text"].fillna("").astype(str).map(normalize_text).tolist()
    baseline_pred = baseline_df["pred_text"].fillna("").astype(str).map(normalize_text).tolist()
    current_gt = pred_df["gt_text"].fillna("").astype(str).map(normalize_text).tolist()
    current_pred = pred_df["pred_text"].fillna("").astype(str).map(normalize_text).tolist()
    baseline_wer = float(jiwer.wer(baseline_gt, baseline_pred))
    baseline_cer = float(jiwer.cer(baseline_gt, baseline_pred))
    current_wer = float(jiwer.wer(current_gt, current_pred))
    current_cer = float(jiwer.cer(current_gt, current_pred))
    return {
        "baseline_path": str(baseline_path),
        "baseline_wer": baseline_wer,
        "baseline_cer": baseline_cer,
        "current_wer": current_wer,
        "current_cer": current_cer,
        "wer_delta": current_wer - baseline_wer,
        "cer_delta": current_cer - baseline_cer,
    }


def build_prediction_summary(pred_df: pd.DataFrame) -> dict[str, Any]:
    pred_text = pred_df["pred_text"].fillna("").astype(str)
    gt_text = pred_df["gt_text"].fillna("").astype(str)
    sample_columns = [
        "clip_id",
        "speaker_id",
        "gt_text",
        "pred_text",
        "wer",
        "cer",
        "sequence_confidence",
    ]
    hardest_examples = (
        pred_df.sort_values(["wer", "cer", "sequence_confidence"], ascending=[False, False, True])[sample_columns]
        .head(5)
        .to_dict(orient="records")
    )
    lowest_confidence_examples = (
        pred_df.sort_values(["sequence_confidence", "wer", "cer"], ascending=[True, False, False])[sample_columns]
        .head(5)
        .to_dict(orient="records")
    )
    sampled_examples = pred_df[sample_columns].head(5).to_dict(orient="records")
    return {
        "prediction_length": {
            "mean_pred_chars": float(pred_text.str.len().mean()),
            "mean_gt_chars": float(gt_text.str.len().mean()),
            "empty_predictions": int((pred_text.str.len() == 0).sum()),
            "unique_prediction_ratio": float(pred_text.nunique() / max(len(pred_df), 1)),
        },
        "confidence": {
            "mean": float(pred_df["sequence_confidence"].mean()),
            "std": float(pred_df["sequence_confidence"].std()) if len(pred_df) > 1 else 0.0,
            "min": float(pred_df["sequence_confidence"].min()),
            "max": float(pred_df["sequence_confidence"].max()),
        },
        "speaker_metrics": (
            pred_df.groupby("speaker_id", as_index=False)
            .agg(
                n=("clip_id", "count"),
                wer=("wer", "mean"),
                cer=("cer", "mean"),
                confidence=("sequence_confidence", "mean"),
            )
            .sort_values(["wer", "cer"], ascending=[False, False])
            .to_dict(orient="records")
        ),
        "sample_predictions": sampled_examples,
        "hardest_examples": hardest_examples,
        "lowest_confidence_examples": lowest_confidence_examples,
    }


def export_results(
    pred_df: pd.DataFrame,
    errors: list[dict[str, str]],
    output_dir: Path,
    split: str,
    assets: BenchmarkAssets,
    decision: str,
    rationale: str,
    baseline_comparison: dict[str, Any] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / f"{split}_predictions_stage1_pretrained.csv"
    handoff_path = output_dir / f"{split}_stage2_handoff_pretrained.csv"
    summary_path = output_dir / f"{split}_summary_stage1_pretrained.json"
    errors_path = output_dir / f"{split}_errors_stage1_pretrained.json"

    pred_df.to_csv(pred_path, index=False)

    handoff_df = pred_df[
        ["clip_id", "pred_text", "sequence_confidence", "confidence_type"]
    ].copy()
    handoff_df["visual_embedding_path"] = ""
    handoff_df.to_csv(handoff_path, index=False)

    gt_text = pred_df["gt_text"].fillna("").astype(str).map(normalize_text).tolist()
    pred_text = pred_df["pred_text"].fillna("").astype(str).map(normalize_text).tolist()
    overall_wer = float(jiwer.wer(gt_text, pred_text))
    overall_cer = float(jiwer.cer(gt_text, pred_text))
    prediction_summary = build_prediction_summary(pred_df)
    summary = {
        "split": split,
        "num_predictions": int(len(pred_df)),
        "num_errors": int(len(errors)),
        "overall_wer": overall_wer,
        "overall_cer": overall_cer,
        "mean_sequence_confidence": float(pred_df["sequence_confidence"].mean()),
        "confidence_type": pred_df["confidence_type"].iloc[0],
        "decoder_name": pred_df["decoder_name"].iloc[0],
        "decision": decision,
        "rationale": rationale,
        "benchmark_config": str(assets.config_path),
        "model_path": str(assets.model_path),
        "model_conf": str(assets.model_conf),
        "rnnlm_path": str(assets.rnnlm_path),
        "rnnlm_conf": str(assets.rnnlm_conf),
        "baseline_comparison": baseline_comparison,
        "prediction_summary": prediction_summary,
        "outputs": {
            "predictions_csv": str(pred_path),
            "stage2_handoff_csv": str(handoff_path),
            "errors_json": str(errors_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    errors_path.write_text(json.dumps(errors, indent=2))


def attach_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.copy()
    pred_df["gt_text"] = pred_df["gt_text"].fillna("").astype(str).map(normalize_text)
    pred_df["pred_text"] = pred_df["pred_text"].fillna("").astype(str).map(normalize_text)
    pred_df["cer"] = [
        float(jiwer.cer(ref, hyp)) for ref, hyp in zip(pred_df["gt_text"], pred_df["pred_text"])
    ]
    pred_df["wer"] = [
        float(jiwer.wer(ref, hyp)) for ref, hyp in zip(pred_df["gt_text"], pred_df["pred_text"])
    ]
    return pred_df


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    split = args.split
    manifest_tsv = args.manifest_tsv or (DATA_ROOT / "dataset_final" / f"{split}.tsv")
    manifest = read_manifest(manifest_tsv, args.max_samples, args.sample_clip_id)
    assets = load_benchmark_assets(args.benchmark_config)
    runner = PretrainedStage1Runner(assets, device=device)

    pred_df, errors = evaluate_manifest(
        manifest=manifest,
        lip_roi_root=args.lip_roi_root,
        runner=runner,
        fail_fast=args.fail_fast,
    )
    pred_df = attach_metrics(pred_df)

    overall_wer = float(jiwer.wer(pred_df["gt_text"].tolist(), pred_df["pred_text"].tolist()))
    overall_cer = float(jiwer.cer(pred_df["gt_text"].tolist(), pred_df["pred_text"].tolist()))
    decision, rationale = decide_action(overall_wer)
    baseline_comparison = compare_with_existing_baseline(args.baseline_path, pred_df)

    export_results(
        pred_df=pred_df,
        errors=errors,
        output_dir=args.output_dir,
        split=split,
        assets=assets,
        decision=decision,
        rationale=rationale,
        baseline_comparison=baseline_comparison,
    )

    print(f"Device: {device}")
    print(f"Manifest: {manifest_tsv}")
    print(f"Predictions: {len(pred_df)}")
    print(f"Errors: {len(errors)}")
    print(f"Overall WER: {overall_wer:.4f}")
    print(f"Overall CER: {overall_cer:.4f}")
    print(f"Decision: {decision}")
    print(f"Outputs written to: {args.output_dir}")
    if baseline_comparison:
        print(
            "Baseline comparison:"
            f" current WER {baseline_comparison['current_wer']:.4f} vs"
            f" baseline WER {baseline_comparison['baseline_wer']:.4f}"
        )


if __name__ == "__main__":
    main()
