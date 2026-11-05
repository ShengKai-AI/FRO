# FRO: paired audio-video compression experiments

Experiment code lives in `exp/baseline/`; the proposed FRO scorer lives in `src/fro/`.
The active data protocol selects **native, exactly-two-video multiple-choice questions** from CrossVid, X-Stream, CVBench, MVU-Eval, MVPBench and SYNCR. Dataset revisions and selection rules are pinned in [the manifest](manifests/native_two_video.json). X-Stream additionally requires an audio track in at least one original stream; the report separately counts questions with audio in both streams.

The current pilot uses 3B / 7B, eight baseline methods (FRO excluded), and at most 200 questions per dataset. Previous results, weights and cached inputs remain available. These new samples contain one question over two videos; they are not the earlier artificial pairs of two independent questions. The active runner accepts one native question over exactly two videos. Legacy pairing entry points have been removed.

[Selection/download report](results/native_two_video/selection_report.json) · [Baseline sources](docs/baseline_sources.md) · [Archived configuration](docs/archive/pre_native_two_video_20261105/)

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

## Data selection and downloads

```bash
# Fetch small pinned annotations and write selected question manifests.
uv run python -m scripts.download_datasets all
# Fetch only media referenced by those questions; reruns reuse completed files.
uv run python -m scripts.download_datasets CrossVid --execute --workers 32
uv run python -m scripts.download_datasets CVBench --execute --workers 8
uv run python -m scripts.download_datasets MVU-Eval --execute --workers 16
uv run python -m scripts.download_datasets MVPBench --execute --workers 8
uv run python -m scripts.download_datasets SYNCR --execute --workers 16
uv run python -m scripts.download_datasets X-Stream --execute --workers 5
uv run python -m scripts.native_dataset_report
```

- Outputs: `data/native_two_video/<dataset>/questions.json`, `selection_summary.json`, `media/`; original metadata stays under `sources/`.
- X-Stream uses `questions.candidates.json` until every referenced source has been probed; `questions.json` contains the final audio-filtered selection. Question timestamps and proactive timestamps are preserved.
- CrossVid includes 1,160 distinct-view questions and 251 questions about two clips of one movie. Clip bounds, object IDs and frame-aligned bounding boxes are retained.
- Set `FRO_USE_LFS_BATCH=1` for CrossVid to resolve many Git LFS download URLs per request; `FRO_LFS_PROXY` optionally configures the batch API proxy.
- ZIP and uncompressed TAR downloads use HTTP ranges. X-Stream's compressed TAR shards require streaming traversal; only selected candidate videos are saved, and sources excluded by the final audio filter are removed.
- Google Drive and CLEVRER use their official hosts. Standard `HTTPS_PROXY` / `HTTP_PROXY` environment variables and `HF_ENDPOINT` are supported. `FFPROBE` can override the ffprobe executable.
- Training splits and media needed only by excluded questions are omitted. Downloads reserve 5 GiB of free space. Completed-file reuse, atomic writes, per-archive receipts and retries support continuation.
- Legacy experiment results and frozen source snapshots remain under local `runs/`; their old launch and question-pairing scripts are removed from the active repository.

## Baseline pilot

```bash
python -m scripts.build_native_pilot MVU-Eval SYNCR CVBench CrossVid --limit 200
python -m scripts.run_native_pilot --run native-baseline-pilot-20261105
```

`exp/baseline/configs/native_pilot.yaml` excludes FRO. It runs Full Tokens, Random, FastV, FastVID, DyCoke-TTM, OmniZip, OmniSelect and OmniPack on both models. Two analytical compute budgets retain the existing FRO-based 45% / 35% reference; this does not execute FRO. Generation and FLOPs accounting use 64 decode steps. Each dataset has a fixed task-proportional subset shared across methods. Gold answers are stored separately and read only by the scorer. Datasets run sequentially; independent configurations and samples use all permitted idle GPUs with resident models, CPU preprocessing and asynchronous input prefetch.

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

Weights, datasets, local font files, credentials and detailed runtime caches are excluded from Git. Read [third-party notices](THIRD_PARTY_NOTICES.md) before redistributing upstream components. The existing plan PDF is retained as `experiment_plan.pdf`; the active data revision is recorded in the new manifest.
