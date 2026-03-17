# LipVoicer Benchmarks (LipSynth)

## What Was Implemented

- Benchmarks run from precomputed LipSynth NPZ inputs (no live ROI extraction during benchmark).
- Inference invocation standardized to `inference_precomputed_npz.py` with explicit config and deterministic sampling.
- Benchmark artifacts consolidated under:
  - `/home/shravan/Workspace/LipSynth/LipVoicer/benchmark_outputs/runs`
  - `/home/shravan/Workspace/LipSynth/LipVoicer/benchmark_outputs/reports`
- Metrics pipeline includes:
  - Required: `status`, `elapsed_s`, `rtf`, `wer`, `cer`
  - Optional: `stoi`, `pesq` (enabled when dependencies are present)
- Notebook includes:
  - Per-run artifact export (CSV/JSON summary/meta)
  - CSV explorer (table preview + status/latency/WER/CER/RTF plots)
  - Cross-run trend summary
  - Diagnostics for NaN text metrics and smoke-test debugging

## Environment/Runtime Notes Captured in Notebook

- Device: `cuda`
- Seed: `1337`
- Sample size: `24`
- ASR guidance: enabled (`disable_asr = false`)
- STOI/PESQ flags in setup output: enabled and available

## Latest Full Run (from notebook output)

- Run ID: `bench_20260316_181304`
- Total clips: `24`
- OK: `24`
- Failed: `0`
- Skipped: `0`

### Aggregate Metrics

- `mean_elapsed_s`: `77.45050709266677`
- `mean_rtf`: `19.62787999775628`
- `mean_wer`: `0.6452494264994265`
- `mean_cer`: `0.43840012367318626`
- `mean_stoi`: `0.17814584394022911`
- `mean_pesq`: `1.0595375746488571`

### Exported Artifacts (reported by notebook)

- `/home/shravan/Workspace/LipSynth/LipVoicer/benchmark_outputs/reports/bench_20260316_181304_per_clip.csv`
- `/home/shravan/Workspace/LipSynth/LipVoicer/benchmark_outputs/reports/bench_20260316_181304_per_clip.json`
- `/home/shravan/Workspace/LipSynth/LipVoicer/benchmark_outputs/reports/bench_20260316_181304_summary.json`
- `/home/shravan/Workspace/LipSynth/LipVoicer/benchmark_outputs/reports/bench_20260316_181304_meta.json`

## Smoke Check Result (notebook)

- Clip: `spk_003_0106`
- Status: `ok`
- `elapsed_s`: `72.12417673899836`
- `wer`: `0.6666666666666666`
- `cer`: `0.48333333333333334`
- `ref_text_len`: `60`
- `hyp_text_len`: `65`

## Cross-Run Snapshot (from CSV explorer output)

- `bench_20260316_172621`: `mean_elapsed_s=62.373156461291615`, `mean_wer=NaN`, `mean_cer=NaN`
- `bench_20260316_181304`: `mean_elapsed_s=77.45050709266677`, `mean_wer=0.6452494264994265`, `mean_cer=0.43840012367318626`

## Interpretation

- The initial run had complete inference failures / missing text metrics.
- The latest run is stable (`24/24 ok`) and produces valid WER/CER and optional STOI/PESQ aggregates.
- Cross-run output shows the transition from non-reportable text metrics to fully populated metrics in the corrected run.
