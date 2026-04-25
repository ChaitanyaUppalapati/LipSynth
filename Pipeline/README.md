# Data255 Project

This repository focuses on the model-development scope described in [docs/architecture.md](docs/architecture.md):

- Stage 1: visual speech recognition from mouth ROIs to text
- Stage 2: conditional mel generation from visual/text/speaker conditioning

The environment is managed with `uv` so the local `.venv` can be recreated from repository metadata instead of manual package installs.

## Environment

Base environment for notebooks and core repo work:

```bash
uv venv --python 3.12 .venv
uv sync
```

Add the optional pretrained Stage 1 dependencies:

```bash
uv sync --extra stage1-pretrained
```

This extra is required for the official LipVoicer pretrained inference path in `scripts/stage1_pretrained_eval.py`.

## Stage 1 Pretrained Evaluation

The pretrained Stage 1 path expects:

- `third_party/LipVoicer`
- benchmark assets under `third_party/LipVoicer/mouthroi_processing/benchmarks/...`
- the `stage1-pretrained` dependency extra

Run it with:

```bash
uv run python scripts/stage1_pretrained_eval.py --split val
```

For a small smoke test:

```bash
uv run python scripts/stage1_pretrained_eval.py --split val --sample-clip-id spk_001_0008
```

Outputs are written under `outputs/stage1_eval/pretrained/`.

The script now resolves repo-local defaults automatically, so it can be run from this checkout without editing hardcoded paths. For a smoke test:

```bash
uv run python scripts/stage1_pretrained_eval.py --split val --max-samples 1
```

## Stage 2 Evaluation

Full Stage 2 evaluation, full-test metrics export, and guidance ablation now live in a reusable script instead of only inside the notebook:

```bash
uv run python scripts/stage2_evaluate.py
```

Useful smoke-test flags:

```bash
uv run python scripts/stage2_evaluate.py --demo-count 1 --full-eval-limit 1 --ablation-count 1 --use-fast-inference
```

The script writes:

- `outputs/stage2_finetune/test_metrics.csv`
- `outputs/stage2_finetune/test_full_metrics.csv`
- `outputs/stage2_finetune/ablation_guidance.csv`
- `outputs/stage2_finetune/ablation_conf_weighted.csv` when Stage 1 passes the credibility gate
- `outputs/stage2_finetune/final_report.json`
- `outputs/stage2_finetune/evaluation_summary.csv`

## Third-party Dependencies

The notebook and scripts depend on LipVoicer, which must be cloned into `third_party/`:

```bash
git clone https://github.com/yochaiye/LipVoicer Pipeline/third_party/LipVoicer
```

This directory is not committed to the repo. The notebook imports from `Pipeline.third_party.LipVoicer.*` and requires pretrained checkpoints.

Download all pretrained checkpoints (MelGen LRS2/LRS3, HiFi-GAN, ASR, lip-reading, tokenizer, LM):

```bash
cd Pipeline/third_party/LipVoicer
python download_checkpoints.py
```

This downloads ~2GB into subdirs: `exp/`, `hifi_gan/`, `ASR/`, `mouthroi_processing/`. Must be run from inside `third_party/LipVoicer/` so relative paths resolve correctly.

## Docs

- Architecture and work split: [docs/architecture.md](docs/architecture.md)
- Environment and dependency workflow: [docs/environment.md](docs/environment.md)
