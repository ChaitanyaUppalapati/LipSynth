# CHANGES / Setup Notes (LipVoicer)

This document describes the changes made to get LipVoicer running in this workspace, and the steps to run inference on **GPU** and **CPU-only** machines.

---

## Key changes made during setup (LipVoicer repo)

### 1. Dependency & environment fixes

- Ensured the venv has `pip` installed and works (`python -m ensurepip`).
- Installed necessary Python packages in the venv (torch, torchvision, torchaudio, mediapipe, face-alignment, librosa, hydra-core, tensorboard, etc.).
- Downloaded all LipVoicer pretrained checkpoints using `download_checkpoints.py`.
- Added a **greedy fallback decoder** so LipVoicer can run even when `ctcdecode` cannot be built.

### 2. CPU-friendly patches

- Updated `inference_real_video.py` to select `cuda` only if it’s available (otherwise `cpu`).
- Updated diffusion hyperparameter utility in `utils.py` to avoid forcing `.cuda()` when no GPU exists.
- Updated `crop_and_infer.py` and the Mediapipe detector to avoid forcing CUDA.

### 3. Face detection fallback

- LipVoicer’s original detector depended on `ibug`/`retinaface`, which is missing.
- Switched the default detector to **Mediapipe** (but note: current Mediapipe version uses the `tasks` API, not `solutions`).

---

## Running inference in GPU mode

### Verify GPU is available

```bash
cd /home/shravan/Workspace/LipVoicer
. ../LipSynth/.venv/bin/activate
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

### Run inference (GPU)

```bash
cd /home/shravan/Workspace/LipVoicer
. ../LipSynth/.venv/bin/activate
python inference_real_video.py \
  generate.video_path=/home/shravan/Workspace/LipSynth/dataset_pipeline/data/dataset_final/videos/spk_001_0008.mp4 \
  generate.save_dir=/home/shravan/Workspace/LipSynth/lipvoicer_output
```

Results will be stored under `lipvoicer_output/<clip_name>/...`

---

## Running inference in CPU-only mode

### (Already patched for CPU use)

The repository now supports CPU inference, though it may run slower.

### Run inference (CPU)

```bash
cd /home/shravan/Workspace/LipVoicer
. ../LipSynth/.venv/bin/activate
python inference_real_video.py \
  generate.video_path=/home/shravan/Workspace/LipSynth/dataset_pipeline/data/dataset_final/videos/spk_001_0008.mp4 \
  generate.save_dir=/home/shravan/Workspace/LipSynth/lipvoicer_output
```

Notes:

- Lip ROI extraction uses Mediapipe (CPU compute can be slow).
- `ctcdecode` is not available and is replaced by greedy CTC decoding.

---

## Notes on what works now

1. LipVoicer can run with the **existing pretrained checkpoints**.
2. The `crop_and_infer` pipeline is configured to run without CUDA.
3. A fallback decoder ensures the model still runs when `ctcdecode` cannot be installed.

---

If you want, I can also add a small helper tool that allows passing precomputed `.npz` lip ROIs and transcripts directly into LipVoicer so you can benchmark using your dataset without any face detection step.
