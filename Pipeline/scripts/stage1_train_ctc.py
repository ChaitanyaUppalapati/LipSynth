#!/usr/bin/env python3
"""Train and evaluate the repo-native Stage 1 CTC lip-reading model.

The notebook-local quick fine-tune is useful for exploration, but it does not
provide a reproducible retraining track. This script makes the repo-native
CNN/Transformer/CTC model trainable from manifests, validates every epoch, and
exports prediction/summary artifacts that can be compared with the official
LipVoicer pretrained baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


REQUIRED_MANIFEST_COLUMNS = ("clip_id", "text", "speaker_id")


def find_pipe_root(start: Path) -> Path:
    for cand in [start.resolve(), *start.resolve().parents]:
        if (cand / "pyproject.toml").exists() and (cand / "data").exists():
            return cand
    raise FileNotFoundError("Could not locate Pipeline root from the current path.")


PIPE_ROOT = find_pipe_root(Path(__file__).resolve())
PROJECT_ROOT = PIPE_ROOT.parent


def discover_data_root(pipe_root: Path) -> Path:
    candidates = [
        pipe_root / "data" / "custom_data",
        PROJECT_ROOT / "dataset_pipeline" / "data",
        PROJECT_ROOT / "data" / "custom_data",
    ]
    for candidate in candidates:
        if (candidate / "dataset_final" / "train.tsv").exists():
            return candidate
    return candidates[0]


DATA_ROOT = discover_data_root(PIPE_ROOT)
DEFAULT_OUTPUT_DIR = PIPE_ROOT / "outputs" / "stage1_eval" / "retrain"
DEFAULT_PRETRAINED_SUMMARY = (
    PIPE_ROOT / "outputs" / "stage1_eval" / "pretrained" / "val_summary_stage1_pretrained.json"
)
DEFAULT_PRETRAINED_PREDICTIONS = (
    PIPE_ROOT / "outputs" / "stage1_eval" / "pretrained" / "val_predictions_stage1_pretrained.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate Stage 1 CNN+Transformer+CTC.")
    parser.add_argument("--train-tsv", type=Path, default=DATA_ROOT / "dataset_final" / "train.tsv")
    parser.add_argument("--val-tsv", type=Path, default=DATA_ROOT / "dataset_final" / "val.tsv")
    parser.add_argument("--lip-roi-root", type=Path, default=DATA_ROOT / "lip_rois")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true", help="Enable ROI augmentation for training.")
    parser.add_argument("--time-mask-frames", type=int, default=8)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to evaluate in eval-only mode.")
    parser.add_argument("--pretrained-summary", type=Path, default=DEFAULT_PRETRAINED_SUMMARY)
    parser.add_argument("--pretrained-predictions", type=Path, default=DEFAULT_PRETRAINED_PREDICTIONS)
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is not available.")
    return torch.device(requested)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())


def read_manifest(path: Path, max_samples: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path, sep="\t", engine="python")
    df.columns = [str(col).strip() for col in df.columns]
    missing = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {missing}")
    for col in REQUIRED_MANIFEST_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    df["text"] = df["text"].map(normalize_text)
    if max_samples is not None:
        df = df.head(max_samples).copy()
    if df.empty:
        raise ValueError(f"Manifest {path} produced no rows.")
    return df.reset_index(drop=True)


@dataclass(frozen=True)
class Vocabulary:
    idx_to_token: list[str]
    token_to_idx: dict[str, int]
    blank_id: int

    @classmethod
    def from_manifests(cls, *manifests: pd.DataFrame) -> "Vocabulary":
        text_values = pd.concat([df["text"] for df in manifests], axis=0).tolist()
        charset = sorted({ch for text in text_values for ch in normalize_text(text)})
        idx_to_token = ["<blank>", *charset]
        token_to_idx = {tok: idx for idx, tok in enumerate(idx_to_token)}
        return cls(idx_to_token=idx_to_token, token_to_idx=token_to_idx, blank_id=0)

    def encode(self, text: str) -> list[int]:
        encoded = [self.token_to_idx[ch] for ch in normalize_text(text) if ch in self.token_to_idx]
        return encoded or [self.blank_id]

    def decode(self, ids: list[int]) -> str:
        chars = []
        for idx in ids:
            if idx == self.blank_id:
                continue
            if 0 <= idx < len(self.idx_to_token):
                chars.append(self.idx_to_token[idx])
        return "".join(chars).strip()


class Stage1Dataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        lip_roi_root: Path,
        vocab: Vocabulary,
        augment: bool = False,
        time_mask_frames: int = 8,
    ) -> None:
        self.df = manifest.reset_index(drop=True)
        self.lip_roi_root = lip_roi_root
        self.vocab = vocab
        self.augment = augment
        self.time_mask_frames = time_mask_frames

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, rois: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return rois
        if torch.rand(()) < 0.5:
            rois = torch.flip(rois, dims=(-1,))
        if torch.rand(()) < 0.7:
            scale = torch.empty(()).uniform_(0.85, 1.15)
            shift = torch.empty(()).uniform_(-0.05, 0.05)
            rois = torch.clamp(rois * scale + shift, 0.0, 1.0)
        if self.time_mask_frames > 0 and rois.shape[1] > 4 and torch.rand(()) < 0.5:
            width = int(torch.randint(1, min(self.time_mask_frames, rois.shape[1]) + 1, ()).item())
            start = int(torch.randint(0, rois.shape[1] - width + 1, ()).item())
            rois[:, start : start + width] = 0.0
        return rois

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        clip_id = row["clip_id"]
        speaker_id = row["speaker_id"]
        roi_path = self.lip_roi_root / speaker_id / f"{clip_id}.npz"
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI file missing: {roi_path}")

        payload = np.load(roi_path)
        if "mouth_rois" not in payload.files:
            raise KeyError(f"'mouth_rois' key missing in {roi_path}")
        rois_np = payload["mouth_rois"]
        if rois_np.ndim != 3 or rois_np.shape[1:] != (96, 96):
            raise ValueError(f"Expected mouth_rois shape (T,96,96), got {rois_np.shape} in {roi_path}")

        rois = torch.from_numpy(rois_np.astype(np.float32) / 255.0).unsqueeze(0)
        rois = self._augment(rois)
        token_ids = self.vocab.encode(row["text"])
        return {
            "clip_id": clip_id,
            "speaker_id": speaker_id,
            "roi_path": str(roi_path),
            "rois": rois,
            "roi_len": rois.shape[1],
            "text": normalize_text(row["text"]),
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "token_len": len(token_ids),
        }


def collate_stage1(batch: list[dict[str, Any]], blank_id: int) -> dict[str, Any]:
    batch_size = len(batch)
    max_t = max(item["roi_len"] for item in batch)
    max_u = max(item["token_len"] for item in batch)
    rois = torch.zeros((batch_size, 1, max_t, 96, 96), dtype=torch.float32)
    roi_lens = torch.zeros((batch_size,), dtype=torch.long)
    token_ids = torch.full((batch_size, max_u), fill_value=blank_id, dtype=torch.long)
    token_lens = torch.zeros((batch_size,), dtype=torch.long)
    clip_ids, speaker_ids, texts, roi_paths = [], [], [], []

    for idx, item in enumerate(batch):
        t_len = item["roi_len"]
        u_len = item["token_len"]
        rois[idx, :, :t_len] = item["rois"]
        roi_lens[idx] = t_len
        token_ids[idx, :u_len] = item["token_ids"]
        token_lens[idx] = u_len
        clip_ids.append(item["clip_id"])
        speaker_ids.append(item["speaker_id"])
        texts.append(item["text"])
        roi_paths.append(item["roi_path"])

    return {
        "clip_id": clip_ids,
        "speaker_id": speaker_ids,
        "roi_path": roi_paths,
        "rois": rois,
        "roi_lens": roi_lens,
        "text": texts,
        "token_ids": token_ids,
        "token_lens": token_lens,
    }


class Stage1LipVoicerCNNCTC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.frontend3d = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ctc_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        rois: torch.Tensor,
        roi_lens: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.frontend3d(rois)
        batch_size, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * frames, channels, height, width)
        x = self.frame_encoder(x).view(batch_size, frames, -1)
        x = self.proj(x)
        padding_mask = torch.arange(frames, device=x.device).unsqueeze(0) >= roi_lens.unsqueeze(1)
        x = self.temporal_encoder(x, src_key_padding_mask=padding_mask)
        logits = self.ctc_head(x)
        if return_features:
            return logits, x
        return logits


@torch.no_grad()
def greedy_ctc_decode(logits: torch.Tensor, blank_id: int) -> list[list[int]]:
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    decoded = []
    for seq in pred:
        out = []
        prev = None
        for token in seq:
            token = int(token)
            if token != blank_id and token != prev:
                out.append(token)
            prev = token
        decoded.append(out)
    return decoded


@torch.no_grad()
def confidence_from_logits(logits: torch.Tensor, blank_id: int) -> np.ndarray:
    probs = torch.softmax(logits, dim=-1)
    max_probs, max_ids = probs.max(dim=-1)
    non_blank = max_ids.ne(blank_id)
    denom = non_blank.sum(dim=1).clamp_min(1)
    return ((max_probs * non_blank).sum(dim=1) / denom).detach().cpu().numpy()


def ctc_loss_for_batch(
    model: nn.Module,
    batch: dict[str, Any],
    ctc_loss: nn.CTCLoss,
    device: torch.device,
) -> torch.Tensor:
    rois = batch["rois"].to(device)
    roi_lens = batch["roi_lens"].to(device)
    token_ids = batch["token_ids"].to(device)
    token_lens = batch["token_lens"].to(device)
    logits = model(rois, roi_lens)
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    targets = torch.cat([token_ids[i, : token_lens[i]] for i in range(token_ids.size(0))])
    return ctc_loss(log_probs, targets, roi_lens, token_lens)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ctc_loss: nn.CTCLoss,
    device: torch.device,
    grad_clip: float,
    max_steps: int | None,
) -> float:
    model.train()
    losses = []
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        if max_steps is not None and step > max_steps:
            break
        loss = ctc_loss_for_batch(model, batch, ctc_loss, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    ctc_loss: nn.CTCLoss,
    vocab: Vocabulary,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    losses = []
    rows = []
    for batch in tqdm(loader, desc="val", leave=False):
        loss = ctc_loss_for_batch(model, batch, ctc_loss, device)
        losses.append(float(loss.item()))
        rois = batch["rois"].to(device)
        roi_lens = batch["roi_lens"].to(device)
        logits = model(rois, roi_lens)
        decoded = greedy_ctc_decode(logits, vocab.blank_id)
        seq_conf = confidence_from_logits(logits, vocab.blank_id)
        for idx in range(len(batch["clip_id"])):
            gt_text = normalize_text(batch["text"][idx])
            pred_text = normalize_text(vocab.decode(decoded[idx]))
            rows.append(
                {
                    "clip_id": batch["clip_id"][idx],
                    "speaker_id": batch["speaker_id"][idx],
                    "roi_path": batch["roi_path"][idx],
                    "gt_text": gt_text,
                    "pred_text": pred_text,
                    "sequence_confidence": float(seq_conf[idx]),
                    "cer": float(jiwer.cer(gt_text, pred_text)),
                    "wer": float(jiwer.wer(gt_text, pred_text)),
                }
            )

    pred_df = pd.DataFrame(rows)
    overall_wer = float(jiwer.wer(pred_df["gt_text"].tolist(), pred_df["pred_text"].tolist()))
    overall_cer = float(jiwer.cer(pred_df["gt_text"].tolist(), pred_df["pred_text"].tolist()))
    metrics = {
        "val_loss": float(np.mean(losses)) if losses else math.nan,
        "overall_wer": overall_wer,
        "overall_cer": overall_cer,
        "mean_sequence_confidence": float(pred_df["sequence_confidence"].mean()),
        "empty_predictions": int(pred_df["pred_text"].eq("").sum()),
    }
    return pred_df, metrics


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("model_state_dict", payload)


def load_checkpoint_vocab(path: Path) -> Vocabulary | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    vocab_payload = payload.get("vocab") if isinstance(payload, dict) else None
    if not vocab_payload:
        return None
    idx_to_token = [str(token) for token in vocab_payload["idx_to_token"]]
    token_to_idx = {str(token): int(idx) for token, idx in vocab_payload["token_to_idx"].items()}
    return Vocabulary(
        idx_to_token=idx_to_token,
        token_to_idx=token_to_idx,
        blank_id=int(vocab_payload["blank_id"]),
    )


def build_baseline_comparison(
    pred_df: pd.DataFrame,
    pretrained_summary_path: Path,
    pretrained_predictions_path: Path,
) -> dict[str, Any] | None:
    if pretrained_predictions_path.exists():
        baseline_df = pd.read_csv(pretrained_predictions_path, keep_default_na=False)
        baseline_df = baseline_df[baseline_df["clip_id"].isin(pred_df["clip_id"])].copy()
        if len(baseline_df) == len(pred_df):
            baseline_df = baseline_df.set_index("clip_id").loc[pred_df["clip_id"]].reset_index()
            for col in ["gt_text", "pred_text"]:
                baseline_df[col] = baseline_df[col].fillna("").astype(str).map(normalize_text)
            baseline_wer = float(jiwer.wer(baseline_df["gt_text"].tolist(), baseline_df["pred_text"].tolist()))
            baseline_cer = float(jiwer.cer(baseline_df["gt_text"].tolist(), baseline_df["pred_text"].tolist()))
            current_wer = float(jiwer.wer(pred_df["gt_text"].tolist(), pred_df["pred_text"].tolist()))
            current_cer = float(jiwer.cer(pred_df["gt_text"].tolist(), pred_df["pred_text"].tolist()))
            return {
                "source": str(pretrained_predictions_path),
                "scope": "matching_clip_ids",
                "baseline_wer": baseline_wer,
                "baseline_cer": baseline_cer,
                "current_wer": current_wer,
                "current_cer": current_cer,
                "wer_delta": current_wer - baseline_wer,
                "cer_delta": current_cer - baseline_cer,
            }
    if pretrained_summary_path.exists():
        data = json.loads(pretrained_summary_path.read_text())
        return {
            "source": str(pretrained_summary_path),
            "scope": "summary",
            "baseline_wer": data.get("overall_wer"),
            "baseline_cer": data.get("overall_cer"),
        }
    return None


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    vocab: Vocabulary,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "vocab": asdict(vocab),
        "args": vars(args),
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_manifest(args.train_tsv, args.max_train_samples)
    val_df = read_manifest(args.val_tsv, args.max_val_samples)
    checkpoint_for_vocab = (args.checkpoint or args.init_checkpoint) if args.eval_only else None
    vocab = load_checkpoint_vocab(checkpoint_for_vocab) if checkpoint_for_vocab else None
    if vocab is None:
        vocab = Vocabulary.from_manifests(train_df, val_df)

    train_ds = Stage1Dataset(
        train_df,
        args.lip_roi_root,
        vocab,
        augment=args.augment and not args.eval_only,
        time_mask_frames=args.time_mask_frames,
    )
    val_ds = Stage1Dataset(val_df, args.lip_roi_root, vocab, augment=False)
    collate_fn = partial(collate_stage1, blank_id=vocab.blank_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = Stage1LipVoicerCNNCTC(
        vocab_size=len(vocab.idx_to_token),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    ctc_loss = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.init_checkpoint is not None:
        model.load_state_dict(load_state_dict(args.init_checkpoint), strict=True)
        print(f"Loaded initial checkpoint: {args.init_checkpoint}")

    if args.eval_only:
        ckpt = args.checkpoint or args.init_checkpoint
        if ckpt is None:
            raise ValueError("--eval-only requires --checkpoint or --init-checkpoint")
        model.load_state_dict(load_state_dict(ckpt), strict=True)
        pred_df, metrics = evaluate(model, val_loader, ctc_loss, vocab, device)
        pred_path = args.output_dir / "val_predictions_stage1_retrain_eval.csv"
        summary_path = args.output_dir / "val_summary_stage1_retrain_eval.json"
        pred_df.to_csv(pred_path, index=False)
        baseline = build_baseline_comparison(pred_df, args.pretrained_summary, args.pretrained_predictions)
        summary = {
            "mode": "eval_only",
            "checkpoint": str(ckpt),
            "num_train": int(len(train_df)),
            "num_val": int(len(val_df)),
            "metrics": metrics,
            "baseline_comparison": baseline,
            "outputs": {"predictions_csv": str(pred_path)},
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    history: list[dict[str, Any]] = []
    best_wer = math.inf
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    stale_epochs = 0
    best_ckpt = args.output_dir / "stage1_ctc_best.pt"
    last_pred_df: pd.DataFrame | None = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            ctc_loss,
            device,
            grad_clip=args.grad_clip,
            max_steps=args.max_steps_per_epoch,
        )
        pred_df, metrics = evaluate(model, val_loader, ctc_loss, vocab, device)
        metrics = {"train_loss": train_loss, **metrics}
        history.append({"epoch": epoch, **metrics})
        last_pred_df = pred_df

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={metrics['val_loss']:.4f} "
            f"wer={metrics['overall_wer']:.4f} cer={metrics['overall_cer']:.4f}"
        )

        if metrics["overall_wer"] < best_wer:
            best_wer = metrics["overall_wer"]
            best_metrics = metrics
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(best_ckpt, model, optimizer, vocab, args, epoch, metrics)
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} stale epochs.")
            break

    if not best_ckpt.exists():
        raise RuntimeError("Training finished without writing a best checkpoint.")

    model.load_state_dict(load_state_dict(best_ckpt), strict=True)
    best_pred_df, best_eval_metrics = evaluate(model, val_loader, ctc_loss, vocab, device)
    pred_path = args.output_dir / "val_predictions_stage1_retrain.csv"
    summary_path = args.output_dir / "val_summary_stage1_retrain.json"
    history_path = args.output_dir / "train_history_stage1_retrain.csv"
    best_pred_df.to_csv(pred_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)
    baseline = build_baseline_comparison(best_pred_df, args.pretrained_summary, args.pretrained_predictions)

    summary = {
        "mode": "train",
        "num_train": int(len(train_df)),
        "num_val": int(len(val_df)),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "best_eval_metrics": best_eval_metrics,
        "baseline_comparison": baseline,
            "config": to_jsonable(vars(args)),
        "vocab_size": len(vocab.idx_to_token),
        "outputs": {
            "best_checkpoint": str(best_ckpt),
            "predictions_csv": str(pred_path),
            "history_csv": str(history_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
