# Native dual-video pilot reproduction

## Data

```bash
python -m scripts.download_datasets all
python -m scripts.download_datasets MVU-Eval --execute --workers 16
python -m scripts.native_dataset_report
python -m scripts.build_native_pilot MVU-Eval SYNCR CVBench CrossVid --limit 200
```

Pinned releases and annotation hashes are in `manifests/native_two_video.json`. Selected questions have exactly two native video inputs and a single-choice answer. Original metadata, bounds, audio probes and selective-download receipts are saved below `data/native_two_video/`. CrossVid frame-directory tasks retain object grounding and scaled bounding boxes; SYNCR FPS placeholders are filled from the source videos. Training splits are excluded.

## Run

```bash
python -m scripts.preflight
python -m scripts.run_native_pilot --run native-baseline-pilot-20261105
python -m scripts.native_pilot_results --run runs/native-baseline-pilot-20261105-MVU-Eval --dataset-dir data/native_pilot/MVU-Eval
```

The pilot selects up to 200 questions per dataset by task-proportional seeded hashing. Video duration affects scheduling, not sample selection. Gold labels are separate from inference manifests. All eight methods share prompts, sampled frames, source order and temporal bounds; FRO is excluded. Both models and two compute budgets are configured in `exp/baseline/configs/native_pilot.yaml`. Dataset sampling follows `docs/native_dual_video_parameters.json`.

Each GPU runs one resident model/configuration. Independent samples share a ready queue; 16 CPU processes prepare inputs and workers prefetch the next input asynchronously. Shared frame processing is cached across questions and models. Consumed derived inputs can be retired; original videos and results remain available. Output directories, leases and per-sample atomic writes prevent concurrent result overwrites.

## Visual-only adapters and scoring

The four pilot datasets use visual inputs. OmniSelect keeps its AudioCLIP image/text scoring, POS keyword filter and visual ranking; absent audio contributes no synthetic tokens. OmniZip applies its released unguided visual branch. OmniPack applies visual stage-I merging and joint stage-II selection. Other visual branches and shared text-token retention are unchanged.

Generation is greedy with at most 64 tokens. Scoring accepts a single option label or an explicit answer label/tag; ambiguous or unparsed responses count as wrong. FLOPs use measured per-layer prefill/KV lengths, a standardized 64-step decode, and measured auxiliary matrix operations. The two targets retain the 45% / 35% FRO reference calculation without running FRO. Budget mismatch and execution failures remain explicit.

## Historical experiments

Old results and frozen source snapshots remain in local `runs/`. The removed legacy files also have a recovery copy at `runs/legacy-paired-code-20261105/`. They are not part of the active launch workflow or published dataset downloader.
