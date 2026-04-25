#!/usr/bin/env python3
"""Reproducible Stage 2 evaluation and guidance ablation entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchvision.transforms as Tv
from PIL import Image
from tqdm.auto import tqdm

try:
    from pesq import pesq
except ImportError:
    pesq = None

from pystoi import stoi


def find_pipe_root(start: Path) -> Path:
    for cand in [start.resolve(), *start.resolve().parents]:
        if (cand / "pyproject.toml").exists() and (cand / "third_party" / "LipVoicer").exists():
            return cand
    raise FileNotFoundError(
        "Could not locate the Pipeline root. Run this from the repository or pass explicit paths."
    )


PIPE_ROOT = find_pipe_root(Path(__file__).resolve())
PROJECT_ROOT = PIPE_ROOT.parent
LV_ROOT = PIPE_ROOT / "third_party" / "LipVoicer"

for path in (PROJECT_ROOT, PIPE_ROOT, LV_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


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
ROI_DIR = DATA_ROOT / "lip_rois"
FACE_DIR = DATA_ROOT / "faces"
MEL_DIR = DATA_ROOT / "mel_specs"
OUTPUT_DIR = PIPE_ROOT / "outputs" / "stage2_finetune"
STAGE1_PRETRAINED_DIR = PIPE_ROOT / "outputs" / "stage1_eval" / "pretrained"
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio_samples"


CFG = {
    "sr": 16000,
    "filter_length": 640,
    "hop_length": 160,
    "win_length": 640,
    "mel_fmin": 20.0,
    "mel_fmax": 8000.0,
    "n_mels": 80,
    "fps": 25,
    "video_window": 25,
    "T": 400,
    "beta_0": 0.0001,
    "beta_T": 0.02,
    "w_video": 2.0,
    "s2_cond_drop_prob": 0.2,
}
CFG["vid_2_aud"] = CFG["sr"] / CFG["fps"] / CFG["hop_length"]


DEFAULT_MELGEN_CKPT = (
    LV_ROOT / "exp" / "LRS2" / "wnet_h512_d12_T400_betaT0.02" / "checkpoint" / "1000000.pkl"
)
DEFAULT_HIFIGAN_CKPT = LV_ROOT / "hifi_gan" / "g_02400000"
DEFAULT_STAGE2_CKPT = OUTPUT_DIR / "stage2_ft_best.pkl"


@dataclass
class Stage1Artifacts:
    summary: dict[str, Any] | None
    predictions: pd.DataFrame | None
    credibility_ok: bool
    credibility_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible Stage 2 evaluation and ablations.")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split for the full evaluation pass.",
    )
    parser.add_argument(
        "--demo-count",
        type=int,
        default=10,
        help="Number of demo clips from the evaluation split to render and score.",
    )
    parser.add_argument(
        "--full-eval-limit",
        type=int,
        default=None,
        help="Optional limit on the number of clips in the full evaluation pass.",
    )
    parser.add_argument(
        "--ablation-count",
        type=int,
        default=20,
        help="Number of validation clips to use for the guidance ablation.",
    )
    parser.add_argument(
        "--guidance-weights",
        default="0.0,1.0,2.0,4.0",
        help="Comma-separated fixed guidance weights for the ablation.",
    )
    parser.add_argument(
        "--stage2-ckpt",
        type=Path,
        default=DEFAULT_STAGE2_CKPT,
        help="Fine-tuned Stage 2 checkpoint to evaluate. Falls back to the pretrained MelGen checkpoint if absent.",
    )
    parser.add_argument(
        "--melgen-ckpt",
        type=Path,
        default=DEFAULT_MELGEN_CKPT,
        help="Base LipVoicer MelGen checkpoint used to initialize the model.",
    )
    parser.add_argument(
        "--hifigan-ckpt",
        type=Path,
        default=DEFAULT_HIFIGAN_CKPT,
        help="HiFi-GAN checkpoint used for waveform synthesis.",
    )
    parser.add_argument(
        "--stage1-summary",
        type=Path,
        default=STAGE1_PRETRAINED_DIR / "val_summary_stage1_pretrained.json",
        help="Stage 1 summary JSON used to gate confidence-weighted guidance.",
    )
    parser.add_argument(
        "--stage1-predictions",
        type=Path,
        default=STAGE1_PRETRAINED_DIR / "val_predictions_stage1_pretrained.csv",
        help="Stage 1 predictions CSV used to source per-clip confidences.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for metrics, plots, and reports.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use. Defaults to auto-detecting cuda, otherwise cpu.",
    )
    parser.add_argument(
        "--use-fast-inference",
        action="store_true",
        help="Use LipVoicer's reduced diffusion schedule for a quicker, lower-quality smoke run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate WAV artifacts even if they already exist.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{requested_device}' but CUDA is not available in this environment."
        )
    return torch.device(requested_device)


def parse_guidance_weights(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def read_manifest(split: str, limit: int | None = None) -> pd.DataFrame:
    manifest_path = MANIFEST_DIR / f"{split}.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path, sep="\t")
    if limit is not None:
        df = df.head(limit).copy()
    return df.reset_index(drop=True)


def roi_tfm(split: str):
    from Pipeline.third_party.LipVoicer.dataloaders.lipreading_utils import (
        CenterCrop,
        Compose,
        HorizontalFlip,
        Normalize as LVNorm,
        RandomCrop,
    )

    crop = (88, 88)
    mean, std = 0.421, 0.165
    base = [LVNorm(0.0, 255.0)]
    aug = [RandomCrop(crop), HorizontalFlip(0.5)] if split == "train" else [CenterCrop(crop)]
    return Compose(base + aug + [LVNorm(mean, std)])


def face_tfm():
    return Tv.Compose(
        [
            Tv.Resize(224),
            Tv.CenterCrop(224),
            Tv.ToTensor(),
            Tv.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_torch_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_hifigan_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text())


def maybe_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def compute_metrics(gt_wav_path: Path, gen_wav_path: Path, sr: int = 16000) -> dict[str, float] | None:
    gt, gt_sr = torchaudio.load(str(gt_wav_path))
    gen, gen_sr = torchaudio.load(str(gen_wav_path))
    if gt_sr != sr:
        gt = torchaudio.functional.resample(gt, gt_sr, sr)
    if gen_sr != sr:
        gen = torchaudio.functional.resample(gen, gen_sr, sr)
    gt_np = gt.mean(0).numpy()
    gen_np = gen.mean(0).numpy()
    length = min(len(gt_np), len(gen_np))
    gt_np = gt_np[:length]
    gen_np = gen_np[:length]
    if length < sr // 4:
        return None
    try:
        result = {"stoi": float(stoi(gt_np, gen_np, sr, extended=False))}
        result["pesq"] = float(pesq(sr, gt_np, gen_np, "wb")) if pesq else None
    except Exception:
        return None
    return result


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"clip_count": 0, "stoi_mean": None, "stoi_std": None, "pesq_mean": None, "pesq_std": None}
    df = pd.DataFrame(rows)
    return {
        "clip_count": int(len(df)),
        "stoi_mean": float(df["stoi"].mean()) if "stoi" in df else None,
        "stoi_std": float(df["stoi"].std()) if "stoi" in df and len(df) > 1 else 0.0,
        "pesq_mean": float(df["pesq"].dropna().mean()) if "pesq" in df and df["pesq"].notna().any() else None,
        "pesq_std": float(df["pesq"].dropna().std()) if "pesq" in df and df["pesq"].notna().sum() > 1 else 0.0,
    }


def load_stage1_artifacts(summary_path: Path, predictions_path: Path) -> Stage1Artifacts:
    if not summary_path.exists():
        return Stage1Artifacts(None, None, False, f"Missing Stage 1 summary: {summary_path}")
    if not predictions_path.exists():
        return Stage1Artifacts(None, None, False, f"Missing Stage 1 predictions: {predictions_path}")

    summary = json.loads(summary_path.read_text())
    predictions = pd.read_csv(predictions_path)
    if predictions.empty:
        return Stage1Artifacts(summary, predictions, False, "Stage 1 predictions CSV is empty.")
    if "pred_text" not in predictions.columns or "sequence_confidence" not in predictions.columns:
        return Stage1Artifacts(summary, predictions, False, "Stage 1 predictions CSV is missing required columns.")

    pred_text = predictions["pred_text"].fillna("").astype(str)
    unique_ratio = pred_text.nunique() / max(len(predictions), 1)
    mean_len = pred_text.str.len().mean()
    overall_wer = float(summary.get("overall_wer", 1.0))

    credible = overall_wer <= 0.85 and mean_len >= 8.0 and unique_ratio >= 0.50
    if not credible:
        reason = (
            f"Stage 1 credibility gate failed: WER={overall_wer:.4f}, "
            f"mean_pred_len={mean_len:.1f}, unique_ratio={unique_ratio:.2f}"
        )
        return Stage1Artifacts(summary, predictions, False, reason)
    return Stage1Artifacts(summary, predictions, True, "Stage 1 predictions passed the credibility gate.")


def load_runtime_imports():
    from Pipeline.third_party.LipVoicer.dataloaders.stft import (
        TacotronSTFT,
        denormalise_mel,
        normalise_mel,
    )
    from Pipeline.third_party.LipVoicer.hifi_gan.env import AttrDict
    from Pipeline.third_party.LipVoicer.hifi_gan.generator import Generator
    from Pipeline.third_party.LipVoicer.models.audiovisual_model import AudioVisualModel
    from Pipeline.third_party.LipVoicer.models.model_builder import ModelBuilder
    from Pipeline.third_party.LipVoicer.utils import calc_diffusion_hyperparams, diffwave_fast_inference_schedule

    return {
        "TacotronSTFT": TacotronSTFT,
        "denormalise_mel": denormalise_mel,
        "normalise_mel": normalise_mel,
        "AttrDict": AttrDict,
        "Generator": Generator,
        "AudioVisualModel": AudioVisualModel,
        "ModelBuilder": ModelBuilder,
        "calc_diffusion_hyperparams": calc_diffusion_hyperparams,
        "diffwave_fast_inference_schedule": diffwave_fast_inference_schedule,
    }


class Stage2Runtime:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.args = args
        self.device = device
        self.imports = load_runtime_imports()
        self.output_dir = args.output_dir
        self.audio_dir = self.output_dir / "audio_samples"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self.stft_fn = self.imports["TacotronSTFT"](
            filter_length=CFG["filter_length"],
            hop_length=CFG["hop_length"],
            win_length=CFG["win_length"],
            sampling_rate=CFG["sr"],
            mel_fmin=CFG["mel_fmin"],
            mel_fmax=CFG["mel_fmax"],
        ).to(device)
        self.normalise_mel = self.imports["normalise_mel"]
        self.denormalise_mel = self.imports["denormalise_mel"]
        self._build_model()
        self._load_stage2_weights()
        self._load_vocoder()

    def _build_model(self) -> None:
        builder_cls = self.imports["ModelBuilder"]
        model_cls = self.imports["AudioVisualModel"]

        old_cwd = Path.cwd()
        os.chdir(LV_ROOT)
        try:
            builder = builder_cls()
            net_lip = builder.build_lipreadingnet()
            net_face = builder.build_facial(fc_out=128, with_fc=True)
            model_cfg = dict(
                _name_="melgen",
                in_channels=80,
                out_channels=80,
                diffusion_step_embed_dim_in=128,
                diffusion_step_embed_dim_mid=512,
                diffusion_step_embed_dim_out=512,
                res_channels=512,
                skip_channels=512,
                num_res_layers=12,
                dilation_cycle=1,
                mel_upsample=[2, 2],
            )
            net_dw = builder.build_diffwave_model(model_cfg)
        finally:
            os.chdir(old_cwd)

        self.net = model_cls((net_lip, net_face, net_dw)).to(self.device)

    def _load_stage2_weights(self) -> None:
        if not self.args.melgen_ckpt.exists():
            raise FileNotFoundError(f"Base MelGen checkpoint not found: {self.args.melgen_ckpt}")

        payload = load_torch_payload(self.args.melgen_ckpt)
        state_dict = payload.get("model_state_dict", payload)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        missing, unexpected = self.net.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"Base MelGen checkpoint is missing {len(missing)} keys. Sample: {missing[:5]}"
            )
        if unexpected:
            raise RuntimeError(
                f"Base MelGen checkpoint has {len(unexpected)} unexpected keys. Sample: {unexpected[:5]}"
            )

        self.stage2_meta: dict[str, Any] = {"source": "pretrained"}
        if self.args.stage2_ckpt.exists():
            ft_payload = load_torch_payload(self.args.stage2_ckpt)
            ft_state = ft_payload["model_state_dict"]
            ft_state = {k.replace("module.", "", 1): v for k, v in ft_state.items()}
            self.net.load_state_dict(ft_state)
            self.stage2_meta = {
                "source": str(self.args.stage2_ckpt),
                "step": int(ft_payload.get("step", -1)),
                "val_loss": float(ft_payload.get("val_loss")) if ft_payload.get("val_loss") is not None else None,
            }

        diff_hp = self.imports["calc_diffusion_hyperparams"](
            T=CFG["T"], beta_0=CFG["beta_0"], beta_T=CFG["beta_T"]
        )
        self.diff_hp = {
            key: value.to(self.device) if torch.is_tensor(value) else value for key, value in diff_hp.items()
        }
        if self.args.use_fast_inference:
            fast_beta = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5]
            fast_hp = self.imports["diffwave_fast_inference_schedule"](
                CFG["T"], CFG["beta_0"], CFG["beta_T"], beta=fast_beta
            )
            self.inference_hp = {
                key: value.to(self.device) if torch.is_tensor(value) else value for key, value in fast_hp.items()
            }
        else:
            self.inference_hp = self.diff_hp
        self.net.eval()

    def _load_vocoder(self) -> None:
        if not self.args.hifigan_ckpt.exists():
            raise FileNotFoundError(f"HiFi-GAN checkpoint not found: {self.args.hifigan_ckpt}")
        config = load_hifigan_config(LV_ROOT / "hifi_gan" / "config.json")
        generator_cls = self.imports["Generator"]
        attr_dict_cls = self.imports["AttrDict"]
        self.vocoder = generator_cls(attr_dict_cls(config)).to(self.device)
        payload = load_torch_payload(self.args.hifigan_ckpt)
        gen_sd = payload.get("generator", payload)
        gen_sd = {k.replace("module.", "", 1): v for k, v in gen_sd.items()}
        self.vocoder.load_state_dict(gen_sd)
        self.vocoder.eval()
        self.vocoder.remove_weight_norm()

    def prepare_inputs(self, row: pd.Series) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        clip_id = row["clip_id"]
        speaker_id = row["speaker_id"]
        roi_np = np.load(ROI_DIR / speaker_id / f"{clip_id}.npz")["mouth_rois"]
        roi_tensor = torch.FloatTensor(roi_tfm("val")(roi_np)).unsqueeze(0).unsqueeze(0).to(self.device)
        face_tensor = face_tfm()(Image.open(FACE_DIR / f"{clip_id}_face.jpg").convert("RGB")).unsqueeze(0).to(
            self.device
        )
        return roi_np, roi_tensor, face_tensor

    @torch.inference_mode()
    def sample_mel(self, roi_tensor: torch.Tensor, face_tensor: torch.Tensor, w_video: float) -> torch.Tensor:
        hp = self.inference_hp
        mel_len = roi_tensor.shape[2] * int(CFG["vid_2_aud"])
        x = torch.randn(1, CFG["n_mels"], mel_len, device=self.device)
        num_steps = int(hp["Alpha"].shape[0])
        for t in range(num_steps - 1, -1, -1):
            ts = torch.full((1, 1), t, device=self.device, dtype=torch.float32)
            with maybe_autocast(self.device):
                eps_cond = self.net(x, roi_tensor, face_tensor, ts, cond_drop_prob=0)
                eps_uncond = self.net(x, roi_tensor, face_tensor, ts, cond_drop_prob=1)
            eps = ((1 + w_video) * eps_cond - w_video * eps_uncond).float()
            x = (x - (1 - hp["Alpha"][t]) / (1 - hp["Alpha_bar"][t]).sqrt() * eps) / hp["Alpha"][t].sqrt()
            if t > 0:
                x = x + hp["Sigma"][t] * torch.randn_like(x)
        return x

    @torch.inference_mode()
    def mel_to_wav(self, mel_norm: torch.Tensor) -> torch.Tensor:
        mel_denorm = self.denormalise_mel(mel_norm.unsqueeze(0).to(self.device))
        wav = self.vocoder(mel_denorm).squeeze()
        return wav.cpu()

    def ensure_pair(
        self,
        row: pd.Series,
        generated_path: Path,
        gt_path: Path,
        w_video: float,
        force: bool,
    ) -> tuple[Path, Path]:
        need_gen = force or not generated_path.exists()
        need_gt = force or not gt_path.exists()
        if need_gen or need_gt:
            _, roi_tensor, face_tensor = self.prepare_inputs(row)
            if need_gen:
                mel_gen = self.sample_mel(roi_tensor, face_tensor, w_video=w_video).squeeze(0)
                wav_gen = self.mel_to_wav(mel_gen)
                torchaudio.save(str(generated_path), wav_gen.unsqueeze(0), CFG["sr"])
            if need_gt:
                gt_mel = self.normalise_mel(torch.load(MEL_DIR / f"{row['clip_id']}.wav.spec", map_location="cpu"))
                gt_wav = self.mel_to_wav(gt_mel)
                torchaudio.save(str(gt_path), gt_wav.unsqueeze(0), CFG["sr"])
        return generated_path, gt_path


def append_summary_row(summary_rows: list[dict[str, Any]], scenario: str, metrics: dict[str, Any]) -> None:
    summary_rows.append({"scenario": scenario, **metrics})


def run_demo(runtime: Stage2Runtime, split_df: pd.DataFrame, force: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Stage2 demo"):
        clip_id = row["clip_id"]
        gen_path = runtime.audio_dir / f"{clip_id}_generated.wav"
        gt_path = runtime.audio_dir / f"{clip_id}_gt.wav"
        runtime.ensure_pair(row, gen_path, gt_path, w_video=CFG["w_video"], force=force)
        metrics = compute_metrics(gt_path, gen_path)
        if metrics is None:
            continue
        rows.append({"clip_id": clip_id, **metrics})
    demo_df = pd.DataFrame(rows)
    demo_df.to_csv(runtime.output_dir / "test_metrics.csv", index=False)
    return demo_df, summarize_metric_rows(rows)


def run_full_eval(runtime: Stage2Runtime, split: str, limit: int | None, force: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_df = read_manifest(split, limit=limit)
    rows: list[dict[str, Any]] = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Stage2 full eval ({split})"):
        clip_id = row["clip_id"]
        gen_path = runtime.audio_dir / f"{clip_id}_gen_full.wav"
        gt_path = runtime.audio_dir / f"{clip_id}_gt_full.wav"
        runtime.ensure_pair(row, gen_path, gt_path, w_video=CFG["w_video"], force=force)
        metrics = compute_metrics(gt_path, gen_path)
        record = {"clip_id": clip_id, "gt_text": str(row["text"]).strip().lower()}
        if metrics is None:
            record.update({"stoi": None, "pesq": None})
        else:
            record.update(metrics)
        rows.append(record)
    full_df = pd.DataFrame(rows)
    full_df.to_csv(runtime.output_dir / "test_full_metrics.csv", index=False)
    valid_rows = [row for row in rows if row.get("stoi") is not None]
    return full_df, summarize_metric_rows(valid_rows)


def run_fixed_guidance_ablation(
    runtime: Stage2Runtime,
    weights: list[float],
    ablation_df: pd.DataFrame,
    force: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    for weight in weights:
        per_weight_rows: list[dict[str, Any]] = []
        for _, row in tqdm(ablation_df.iterrows(), total=len(ablation_df), desc=f"guidance w={weight}", leave=False):
            clip_id = row["clip_id"]
            gen_path = runtime.audio_dir / f"abl_{clip_id}_w{weight}.wav"
            gt_path = runtime.audio_dir / f"abl_{clip_id}_gt.wav"
            runtime.ensure_pair(row, gen_path, gt_path, w_video=weight, force=force)
            metrics = compute_metrics(gt_path, gen_path)
            if metrics is None:
                continue
            metric_row = {"clip_id": clip_id, "w_video": weight, **metrics}
            per_weight_rows.append(metric_row)
            clip_rows.append(metric_row)
        metrics_summary = summarize_metric_rows(per_weight_rows)
        summary_rows.append({"w_video": weight, **metrics_summary})
    abl_df = pd.DataFrame(summary_rows)
    abl_df.to_csv(runtime.output_dir / "ablation_guidance.csv", index=False)
    return abl_df, clip_rows


def write_ablation_plot(abl_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if abl_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(abl_df["w_video"].astype(str), abl_df["stoi_mean"], yerr=abl_df["stoi_std"], capsize=5)
    axes[0].set_xlabel("w_video")
    axes[0].set_ylabel("STOI")
    axes[0].set_title("STOI vs guidance")
    axes[1].bar(abl_df["w_video"].astype(str), abl_df["pesq_mean"], yerr=abl_df["pesq_std"], capsize=5)
    axes[1].set_xlabel("w_video")
    axes[1].set_ylabel("PESQ")
    axes[1].set_title("PESQ vs guidance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def run_confidence_weighted_ablation(
    runtime: Stage2Runtime,
    ablation_df: pd.DataFrame,
    stage1_predictions: pd.DataFrame,
    force: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    conf_map = stage1_predictions.set_index("clip_id")["sequence_confidence"].to_dict()
    rows: list[dict[str, Any]] = []
    for _, row in tqdm(ablation_df.iterrows(), total=len(ablation_df), desc="guidance conf", leave=False):
        clip_id = row["clip_id"]
        if clip_id not in conf_map:
            continue
        weight = CFG["w_video"] * float(conf_map[clip_id])
        gen_path = runtime.audio_dir / f"conf_{clip_id}.wav"
        gt_path = runtime.audio_dir / f"conf_{clip_id}_gt.wav"
        runtime.ensure_pair(row, gen_path, gt_path, w_video=weight, force=force)
        metrics = compute_metrics(gt_path, gen_path)
        if metrics is None:
            continue
        rows.append({"clip_id": clip_id, "conf": float(conf_map[clip_id]), "w_eff": weight, **metrics})
    conf_df = pd.DataFrame(rows)
    conf_df.to_csv(runtime.output_dir / "ablation_conf_weighted.csv", index=False)
    return conf_df, summarize_metric_rows(rows)


def write_failure_buckets(conf_df: pd.DataFrame, output_path: Path) -> pd.DataFrame | None:
    if conf_df.empty:
        return None
    bucketed = conf_df.copy()
    bucketed["conf_bucket"] = pd.cut(
        bucketed["conf"],
        bins=[0.0, 0.3, 0.6, 0.8, 1.0],
        labels=["very_low", "low", "mid", "high"],
        include_lowest=True,
    )
    buckets = (
        bucketed.groupby("conf_bucket", observed=False)[["stoi", "pesq"]]
        .mean()
        .round(4)
        .reset_index()
    )
    buckets.to_csv(output_path, index=False)
    return buckets


def write_final_report(
    runtime: Stage2Runtime,
    stage1: Stage1Artifacts,
    demo_metrics: dict[str, Any],
    full_metrics: dict[str, Any],
    abl_df: pd.DataFrame,
    conf_metrics: dict[str, Any] | None,
    summary_rows: list[dict[str, Any]],
    split: str,
    limit: int | None,
) -> None:
    best_guidance = None
    if not abl_df.empty and abl_df["stoi_mean"].notna().any():
        best_idx = abl_df["stoi_mean"].idxmax()
        best_guidance = float(abl_df.loc[best_idx, "w_video"])

    report = {
        "stage1": {
            "summary_path": str(runtime.args.stage1_summary),
            "predictions_path": str(runtime.args.stage1_predictions),
            "credible": stage1.credibility_ok,
            "credibility_reason": stage1.credibility_reason,
            "overall_wer": float(stage1.summary.get("overall_wer")) if stage1.summary else None,
            "overall_cer": float(stage1.summary.get("overall_cer")) if stage1.summary else None,
        },
        "stage2": {
            "checkpoint": runtime.stage2_meta,
            "split": split,
            "full_eval_limit": limit,
            "demo_metrics": demo_metrics,
            "full_metrics": full_metrics,
        },
        "guidance": {
            "fixed_weights": parse_guidance_weights(runtime.args.guidance_weights),
            "best_fixed_weight": best_guidance,
            "confidence_weighted_metrics": conf_metrics,
        },
        "outputs": {
            "demo_metrics_csv": str(runtime.output_dir / "test_metrics.csv"),
            "full_metrics_csv": str(runtime.output_dir / "test_full_metrics.csv"),
            "guidance_csv": str(runtime.output_dir / "ablation_guidance.csv"),
            "confidence_guidance_csv": str(runtime.output_dir / "ablation_conf_weighted.csv"),
            "failure_buckets_csv": str(runtime.output_dir / "failure_buckets.csv"),
            "summary_csv": str(runtime.output_dir / "evaluation_summary.csv"),
        },
        "rows": summary_rows,
    }
    (runtime.output_dir / "final_report.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    weights = parse_guidance_weights(args.guidance_weights)

    stage1 = load_stage1_artifacts(args.stage1_summary, args.stage1_predictions)
    runtime = Stage2Runtime(args, device=device)

    eval_df = read_manifest(args.split)
    demo_df = eval_df.head(min(args.demo_count, len(eval_df))).copy()
    ablation_df = read_manifest("val", limit=args.ablation_count)

    summary_rows: list[dict[str, Any]] = []

    demo_metrics_df, demo_metrics = run_demo(runtime, demo_df, force=args.force)
    append_summary_row(summary_rows, "demo", demo_metrics)

    _, full_metrics = run_full_eval(runtime, args.split, limit=args.full_eval_limit, force=args.force)
    append_summary_row(summary_rows, f"full_{args.split}", full_metrics)

    abl_df, _ = run_fixed_guidance_ablation(runtime, weights, ablation_df, force=args.force)
    write_ablation_plot(abl_df, runtime.output_dir / "ablation_guidance.png")
    for row in abl_df.to_dict(orient="records"):
        append_summary_row(summary_rows, f"guidance_w_{row['w_video']}", row)

    conf_metrics = None
    conf_df = pd.DataFrame()
    if stage1.credibility_ok and stage1.predictions is not None:
        conf_df, conf_metrics = run_confidence_weighted_ablation(
            runtime,
            ablation_df,
            stage1.predictions,
            force=args.force,
        )
        append_summary_row(summary_rows, "guidance_conf_weighted", conf_metrics)
        write_failure_buckets(conf_df, runtime.output_dir / "failure_buckets.csv")
    else:
        print(f"Skipping confidence-weighted guidance: {stage1.credibility_reason}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(runtime.output_dir / "evaluation_summary.csv", index=False)
    write_final_report(
        runtime=runtime,
        stage1=stage1,
        demo_metrics=demo_metrics,
        full_metrics=full_metrics,
        abl_df=abl_df,
        conf_metrics=conf_metrics,
        summary_rows=summary_rows,
        split=args.split,
        limit=args.full_eval_limit,
    )

    print(f"Device: {device}")
    print(f"Stage 2 checkpoint source: {runtime.stage2_meta}")
    print(f"Demo clips: {len(demo_metrics_df)}")
    print(f"Full metrics: {full_metrics}")
    print(f"Stage 1 credible: {stage1.credibility_ok}")
    print(f"Outputs written to: {runtime.output_dir}")


if __name__ == "__main__":
    main()
