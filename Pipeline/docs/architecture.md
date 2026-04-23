# Stage 1 + Stage 2 Architecture (Repository Scope)

This document defines the architecture and execution plan for this repository.

Environment and dependency setup is documented separately in [environment.md](./environment.md).

Scope is intentionally limited to:

1. Stage 1: Visual Speech Recognition (lip ROI to text)
2. Stage 2: Conditional Mel generation (lip/text/identity to mel-spectrogram)

Data pipeline creation is out of scope here because it is already implemented elsewhere in the project.

## 1. Architecture In Scope

### 1.1 Inputs expected from existing pipeline

This repository consumes prebuilt artifacts:

- Lip ROI tensors (`.npz`), shape approximately `(T, 96, 96)`
- Text transcripts (clip-level and tokenized forms)
- Audio-derived targets for Stage 2 (mel-spectrograms)
- Split manifests (`train.tsv`, `val.tsv`, `test.tsv`) with speaker-disjoint partitions

No data download, clipping, Whisper run, or ROI extraction logic is part of this repo scope.

### 1.2 Stage 1 model: Lip ROI to text

Goal: predict transcript text from silent mouth motion.

Recommended model stack:

- 3D CNN visual frontend for short-range space-time features
- Transformer encoder for sequence context
- CTC decoder for alignment-free token prediction

Stage 1 outputs used by Stage 2:

- Predicted token sequence `T_hat`
- Token confidence / CTC confidence score `c` in range `[0, 1]`
- Optional intermediate visual embedding `V`

### 1.3 Stage 2 model: conditional diffusion mel generation

Goal: generate mel-spectrogram from visual and text conditions.

Recommended model stack:

- Conditional DDPM (MelGen-style) for mel generation
- Conditioning with visual feature `V`, identity embedding `I_x`, and text signal `T`

Core diffusion behavior:

- Forward process: add noise to mel target for fixed timesteps
- Reverse process: denoise to recover mel estimate

Classifier guidance variant in scope:

$$
g = w_2 \cdot c \cdot \nabla \log p(T \mid M)
$$

where `c` is CTC confidence from Stage 1. Low `c` reduces text guidance influence.

### 1.4 Stage interface contract

Contract between Stage 1 and Stage 2 must be explicit and versioned:

- Input key set: `clip_id`, `speaker_id`, `roi_path`, `token_ids`, `token_confidence`
- Tensor conventions: sequence length, padding mask, dtype, device placement
- Text conventions: vocabulary, blank token id, normalization rules
- Confidence conventions: per-token vs per-sequence score and aggregation rule

## 2. What Models We Train and With What

### 2.1 Stage 1 training

Trainable components:

- 3D CNN frontend
- Transformer encoder
- CTC projection head

Training data:

- Input: lip ROI sequence tensors
- Target: transcript token sequence

Losses and metrics:

- Primary loss: CTC loss
- Validation metrics: WER, CER

Outputs needed for downstream use:

- Best checkpoint for decoding
- Confidence-calibrated decoder output
- Saved inference format consumed by Stage 2

### 2.2 Stage 2 training

Trainable components:

- MelGen diffusion network (full or partial fine-tune)
- Optional LoRA adapters on attention blocks

Training data:

- Input: visual features and/or ROI-derived embeddings, text conditioning, speaker id embedding
- Target: ground-truth mel-spectrograms from paired audio

Losses and metrics:

- Primary diffusion objective (noise prediction / denoising objective)
- Validation metrics: STOI, PESQ, DNSMOS (and WER via downstream ASR on synthesized speech)

Guidance behavior to train/evaluate:

- Baseline fixed guidance weight
- Confidence-weighted guidance using Stage 1 `c`

### 2.3 Not trained in this repository focus

- Data ingestion and preprocessing pipeline
- Raw video download and segmentation tools
- Face alignment and ROI extraction framework
- Standalone vocoder research track (unless needed for Stage 2 output validation)

## 3. Detailed Work Split into Four Sections

### Section A: Data Contract and Stage Boundary

Objective:

- Lock the exact schema between existing dataset artifacts, Stage 1 training, and Stage 2 conditioning.

Detailed tasks:

1. Define manifest schema and required fields.
2. Implement dataset loaders for Stage 1 and Stage 2 using existing TSV/NPZ assets.
3. Add strict validators for missing clips, malformed tokens, and sequence length mismatch.
4. Add deterministic train/val/test loader behavior with seed control.

Deliverables:

- Schema document and parser utilities
- Loader unit tests and shape checks
- Reproducible dataloader configuration

Done criteria:

- Stage 1 and Stage 2 can run one full epoch without data-shape/runtime errors.

### Section B: Stage 1 Model, Training, and Confidence Calibration

Objective:

- Train a reliable lip-to-text model and produce confidence outputs usable by Stage 2.

Detailed tasks:

1. Implement 3D CNN + Transformer + CTC architecture.
2. Train baseline and tune decoding (greedy/beam as needed).
3. Compute confidence (`c`) from CTC outputs and calibrate it on validation.
4. Export per-clip artifacts: predicted tokens, confidence, optional visual embeddings.

Deliverables:

- Best Stage 1 checkpoint
- Validation report (WER/CER + confidence calibration curves)
- Export format consumed by Stage 2

Done criteria:

- Stable validation WER/CER and confidence signal correlated with transcript quality.

### Section C: Stage 2 Diffusion Fine-Tuning and Guidance

Objective:

- Fine-tune mel generation for custom-domain robustness and integrate confidence-weighted guidance.

Detailed tasks:

1. Implement conditional diffusion training with text, visual, and identity conditioning.
2. Add LoRA fine-tuning path for targeted adaptation.
3. Integrate guidance term with both fixed and confidence-weighted variants.
4. Run ablations: no guidance vs fixed guidance vs confidence-weighted guidance.

Deliverables:

- Stage 2 checkpoint(s)
- Ablation table with objective and perceptual metrics
- Inference config presets for each guidance mode

Done criteria:

- Confidence-weighted guidance improves robustness on noisy/ambiguous clips compared to fixed guidance.

### Section D: Joint Evaluation, Error Analysis, and Release Readiness

Objective:

- Validate end-to-end behavior of Stage 1 + Stage 2 and prepare reproducible release artifacts.

Detailed tasks:

1. Build end-to-end evaluation script chaining Stage 1 predictions into Stage 2 generation.
2. Compute WER/CER/STOI/PESQ/DNSMOS on held-out speaker-disjoint split.
3. Perform failure clustering: viseme ambiguity, fast speech, pose, lighting, OOV text.
4. Freeze configs and produce reproducible experiment cards.

Deliverables:

- Final benchmark report
- Error analysis summary with top failure modes
- Reproducible training and inference configs

Done criteria:

- Team can reproduce reported metrics from pinned configs/checkpoints.

## 4. Recommended Execution Order

1. Complete Section A first (no stable stage boundary means downstream churn).
2. Build Section B baseline to produce reliable text and confidence.
3. Execute Section C with ablations and select best guidance strategy.
4. Close with Section D and freeze release artifacts.

## 5. Scope Summary

This repository focuses only on Stage 1 and Stage 2 model development and evaluation.

- Keep data-pipeline implementation details out of this doc.
- Treat existing data artifacts as upstream dependencies.
- Optimize for robust stage coupling, confidence-aware guidance, and reproducible evaluation.
