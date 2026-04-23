# Environment Setup

This repository uses `uv` for dependency management and local virtualenv workflows.

## Python Version

- Python `3.12`
- Version pin is stored in [.python-version](../.python-version)

## Base Setup

Create or reuse the repository virtualenv:

```bash
uv venv --python 3.12 .venv
uv sync
```

The base dependency set covers the current notebook and general repo analysis workflow:

- `ipykernel`
- `jiwer`
- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `torchaudio`
- `torchvision`
- `tqdm`

## Pretrained Stage 1 Setup

The official LipVoicer pretrained Stage 1 path needs additional dependencies. Install them with:

```bash
uv sync --extra stage1-pretrained
```

This extra includes:

- `configargparse`
- `editdistance`
- `gdown`
- `librosa`
- `opencv-python-headless`
- `requests`
- `sentencepiece`
- `torch-complex`

These packages support the benchmark lip-reading model, beam-search decoding stack, and benchmark asset download flow used by `scripts/stage1_pretrained_eval.py`.

## Benchmark Assets

The pretrained evaluation script requires the official LipVoicer benchmark files:

- `mouthroi_processing/benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth`
- `mouthroi_processing/benchmarks/LRS3/models/LRS3_V_WER19.1/model.json`
- `mouthroi_processing/benchmarks/LRS3/language_models/lm_en_subword/model.pth`
- `mouthroi_processing/benchmarks/LRS3/language_models/lm_en_subword/model.json`

If they are missing, `scripts/stage1_pretrained_eval.py` fails loudly with the exact missing paths.

## Commands

Base notebook environment:

```bash
uv sync
```

Pretrained Stage 1 environment:

```bash
uv sync --extra stage1-pretrained
```

Run the pretrained Stage 1 evaluator:

```bash
uv run python scripts/stage1_pretrained_eval.py --split val
```
