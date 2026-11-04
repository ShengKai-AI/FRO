# FRO: paired audio-video compression experiments

Experiment code lives in `exp/baseline/`; the proposed FRO scorer lives in `src/fro/`.
The plan compares Qwen2.5-Omni **3B / 7B**, **45% / 35% final AV-token retention**, and six datasets: AVUT Human, WorldSense, Video-MME, MVBench, LongVideoBench, DailyOmni. Decoding is greedy, once per question pair, with at most 32 new tokens.

[Experiment plan (PDF)](docs/experiment_plan.pdf) · [HTML source](docs/experiment_plan.html) · [Baseline papers and implementation lineage](docs/baseline_sources.md) · [Reproduction](docs/reproduction.md) · [Validation evidence](results/validation_summary.json)

## Current validation and scope

A single real Video-MME question pair, with 8 frames per video for this short validation only, exercised all 11 methods on the 7B model. Independent jobs ran on four physical A100 GPUs. DyCoke-TTM and OmniZip received a targeted 45%/35% budget repair check. This is a repository integration check, not a benchmark result. See the saved evidence for exact counts, failures before repair, measured ratios and GPU activity.

Full six-dataset evaluation has **not** started. The following work remains before claiming full plan reproduction:

- Validate long-audio segment timestamp remapping and joint resampling when audio alone exceeds the paired input budget. These paths currently fail explicitly instead of truncating input.
- Validate missing-audio variants for every baseline (OmniSelect currently rejects absent audio), all six dataset media resolvers, 3B execution and the full budget matrix.
- Measure selection/merging and auxiliary-model FLOPs. Decoder estimates are available; total FLOPs are left unmeasured. Bootstrap confidence intervals and the final six-dataset main table await complete results.

## Install

Linux, Python 3.11, CUDA 12.4-compatible NVIDIA driver, FFmpeg, Git, and `uv` are expected.

```bash
uv sync --frozen
uv pip install --no-build-isolation 'flash-attn==2.7.4.post1'
uv run python -m scripts.fetch_baselines --execute
uv run python -m scripts.download_models 7B --execute
uv run python -m scripts.download_models 3B --execute
uv run python -m scripts.download_auxiliary --execute
```

The lock file pins Python dependencies. FlashAttention is a CUDA-specific optional dependency installed after Torch; the real validation used Torch 2.6.0+cu124 and FlashAttention 2.7.4.post1. Upstream checkout commits are pinned separately. A complete fresh-environment installation has not been tested.

## Data and execution

Download commands print a plan unless `--execute` is supplied. Fetch annotations first, then one dataset's media at a time. Access-controlled sources use existing Hugging Face credentials.

```bash
uv run python -m scripts.download_datasets Video-MME --execute
# Download media archives separately when ready; archives are not extracted automatically.
uv run python -m scripts.download_datasets Video-MME --patterns 'videos_chunked_*.zip' --execute
uv run python -m exp.baseline.datasets.normalize --dataset Video-MME \
  --annotations data/raw/Video-MME/videomme --output data/pairs
```

Create a JSON media map from each `video_id` to its extracted local file, or to `{"path": "...", "audio_path": "...", "start": 0, "end": 10}` for separate audio/temporal bounds. Do not include answers in this map. MVBench frame-directory and restricted NTU sources require a verified media resolver before use.

```bash
uv run python -m exp.baseline.prepare --dataset Video-MME \
  --pairs data/pairs/Video-MME.json --media-map data/raw/Video-MME/media_map.json
uv run python -m scripts.preflight
# Creates inspectable jobs; no GPU workers start without --execute.
uv run python -m exp.baseline.run --dataset Video-MME --run videomme-v4
# When the required preprocessing adaptations have been validated:
uv run python -m exp.baseline.run --dataset Video-MME --run videomme-v4 --execute
```

The queue respects inherited `CUDA_VISIBLE_DEVICES`, checks existing compute processes and binds one worker to each permitted free GPU. Job outputs, logs and monitoring are isolated under `runs/<name>/`. For partial runs use `--models`, `--methods`, `--ratios` or `--pair-ids`. Never use `--validation-frame-cap` for main benchmark results.

## Layout

```text
src/fro/                 FRO scoring, selection and temporal features
exp/baseline/            preparation, methods, runtime, scoring, scheduler
scripts/                 pinned asset downloads, verification and preflight
manifests/               model/dataset revisions, source lineage and checksums
docs/                    approved plan, reproduction and source references
results/                 compact validation evidence and analytical cost examples
weights/ data/ third_party/ runs/   local assets and outputs (ignored)
```

Weights, datasets, local font files, credentials and detailed runtime caches are excluded from Git. Read [third-party notices](THIRD_PARTY_NOTICES.md) before redistributing upstream components. The original plan PDF is retained as `experiment_plan.pdf`; FRO remains in the plan.
