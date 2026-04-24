# LipSynth

LipSynth is a lip-reading and lip-to-speech research repo with two main parts:

- `dataset_pipeline/` for building a dataset from videos
- `Pipeline/` for running experiments, notebooks, and pretrained evaluation

## Folder Guide

### Top Level

- `dataset_pipeline/`: scripts for downloading videos, segmenting clips, extracting visual features, and building dataset manifests
- `Pipeline/`: the main project workspace for Stage 1 and Stage 2 experiments
- `LipVoicer/`: benchmark outputs and reference artifacts already tracked in this repo
- `benchmarks.ipynb`: benchmark notebook from the original LipSynth workspace
- `pyproject.toml`: root Python project config for the dataset pipeline

### Inside `Pipeline/`

- `Pipeline/docs/`: architecture notes and environment setup docs
- `Pipeline/notebooks/`: research and experiment notebooks
- `Pipeline/scripts/`: runnable scripts such as pretrained Stage 1 evaluation
- `Pipeline/data/`: local datasets and processed data files
- `Pipeline/outputs/`: generated evaluation results and experiment outputs
- `Pipeline/third_party/`: external dependencies and reference code/assets that are not stored in Git

## What Must Be Downloaded Separately

Some large assets are not stored in Git and need to be downloaded manually.

Use this shared Drive folder:

- <https://drive.google.com/drive/folders/1MX69faHA2zc9UqwHApSrm_HH1d9Xz3C0?usp=drive_link>

Place the downloaded contents into the matching folders under `Pipeline/`.

Most importantly:

- `Pipeline/data/` should come from the shared Drive
- `Pipeline/third_party/` should come from LipVoicer [GitHub Repo](https://github.com/yochaiye/LipVoicer)

These folders are required because they contain large files such as:

- prepared datasets
- checkpoints
- benchmark assets
- external code or model resources used by pretrained evaluation

## Simple Setup

### What You Need

1. `git`
2. Python `3.12`
3. `uv`
4. The shared Drive files for `Pipeline/data/` and `Pipeline/third_party/`

Install `uv` if needed:

```bash
pip install uv
```

### Setup Steps

1. Clone the repo.
2. Download the shared assets from Drive.
3. Put the Drive contents into:
   - `Pipeline/data/`
   - `Pipeline/third_party/`
4. Create the environment:

```bash
cd Pipeline
uv venv --python 3.12 .venv
uv sync
```

5. If you want pretrained Stage 1 evaluation, install the extra dependencies:

```bash
uv sync --extra stage1-pretrained
```

## Quick Check

After setup, you can test the environment with:

```bash
cd Pipeline
uv run python scripts/stage1_pretrained_eval.py --split val --sample-clip-id spk_001_0008
```

If this script fails because files are missing, first check:

- `Pipeline/data/`
- `Pipeline/third_party/LipVoicer/`
- `Pipeline/third_party/LipVoicer/mouthroi_processing/benchmarks/`

## Which Part To Use

- Use `dataset_pipeline/` if you want to create or refresh datasets from raw videos
- Use `Pipeline/` if you want to run the main project experiments

## Useful Docs

- [Pipeline/README.md](Pipeline/README.md)
- [Pipeline/docs/architecture.md](Pipeline/docs/architecture.md)
- [Pipeline/docs/environment.md](Pipeline/docs/environment.md)
- [dataset_pipeline/README.md](dataset_pipeline/README.md)
