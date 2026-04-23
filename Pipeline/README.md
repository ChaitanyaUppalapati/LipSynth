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

## Docs

- Architecture and work split: [docs/architecture.md](docs/architecture.md)
- Environment and dependency workflow: [docs/environment.md](docs/environment.md)
