# Reproduction notes

## Assets and paths

Run module commands from the repository root, after installation. `FRO_WEIGHTS`, `FRO_DATA`, `FRO_THIRD_PARTY`, `FRO_RUNS` override asset roots; `FRO_MODEL_3B` and `FRO_MODEL_7B` override individual model paths. `FRO_FFMPEG` and `FRO_FFPROBE` select media executables. No algorithm requires a historical workspace path.

`download_models` and `download_datasets` use immutable Hugging Face revisions and support resumable transfers through the hub client. The manifests preserve upstream LFS SHA256 hashes where published. Downloads save measured checksums and byte counts in `download_receipt.json`; verify these using `python -m scripts.verify_assets PATH/TO/download_receipt.json`. Disk space is checked before downloading. Media extraction is separate and no cache is deleted automatically.

`fetch_baselines` verifies pinned commits and applies the explicit OmniSelect token-pooling patch. It refuses to move a modified checkout to another revision. AudioCLIP weights and BPE resources have fixed SHA256 hashes in `manifests/auxiliary.json`. NLTK resources are downloaded by resource name and are not yet revision-locked; archive them for exact cross-machine replay.

## Dataset preparation

Use only `AV_Human_filtered_data.json` for AVUT. WorldSense and Video-MME use released parquet annotations; MVBench uses the 20 annotation JSONs and the source namespaces in configuration; LongVideoBench uses `lvb_val.json`; DailyOmni uses `qa.json`. Dataset full counts are checked unless `--allow-subset` is explicitly supplied. Annotation adapters beyond the Video-MME path have not yet received end-to-end validation.

Pairing reads questions and video IDs, never labels. Pair files contain no gold; the normalizer writes labels separately as `<dataset>.gold.json`. Each source question has one scored slot. Unscored companions allow distinct-video pairing for odd totals. Preserve the generated pair manifest across every model/method/budget. A newly generated manifest is not assumed byte-identical to historical manifests.

The cache stores source hashes, frame times, bounds and preprocessing settings. Preparation uses full valid timelines with 2-fps candidate sampling and per-dataset frame caps, FFmpeg mono 16-kHz audio, and the inherited gain 0.5. For long audio or unresolved joint audio budgets the preparer raises a descriptive error. The planned 60×5-second sampling and original-timestamp remapping remain a full-evaluation prerequisite.

Missing audio is never synthesized. Video-only inputs are supported in the shared feature path and FRO scorer, but each adapted baseline still requires validation on that case. Unresolved source answers remain null and are rejected by scoring.

## Evaluation and costs

```bash
python -m exp.baseline.evaluate --run runs/videomme-v4 \
  --gold data/pairs/Video-MME.gold.json --output results/videomme-v4.json
python -m exp.baseline.report --run runs/videomme-v4 --output results/videomme-v4.csv
```

Evaluation requires every selected job result to be present and passing, rejects duplicate scored IDs and parses Q1/Q2 separately. Its current summaries apply to the selected job scope. Pass `--require-full-coverage` with the complete dataset gold manifest before calling it a complete benchmark. MVBench's main-table metric must use the macro-average over its 20 task accuracies. The cross-dataset relative main table and video-pair bootstrap have not yet been implemented.

v5 decoder accounting uses actual per-layer prefill and KV lengths, GQA projection dimensions, dense attention matrices, and 100 decode steps, exactly as defined in the current PDF. It counts 1 MAC as 2 FLOPs. `results/archive/v4_decoder_flops_example.csv` is a superseded v4 illustration and must not be used for v5 comparisons. Current reference budgets are in `docs/matched_budgets.csv`. Norm/softmax/RoPE are excluded from the analytical decoder component. Independent FRO scoring, AudioCLIP, selection and merging costs are not measured and total FLOPs remain null; they must not be replaced with zero. Encoder costs are included in runtime timing but excluded from this analytical component.

## Method adaptations

See `manifests/methods.json`. FastV-AV executes five full layers and prunes in the actual joint decoder; OmniPack uses stage-I 2r and stage-II .5 at layer 18 for 7B or 26 for 3B. FRO replays original features after its separate scorer. Video-only baseline adapters use uniform audio retention. DyCoke-TTM uses explicit four-frame anchor groups with integer remainder allocation and rejects budgets below its anchor floor; it is not full DyCoke. OmniZip adjusts its two retention parameters together to preserve the audio-priority difference while accounting for anchor tokens.

## PDF rebuilding

The checked-in PDF was exported from the HTML using Chromium, A4 portrait, Times New Roman and SimSun, with the references last. For a rebuild, install the document extra and Playwright Chromium and supply licensed fonts under `docs/assets/` (`times.ttf`, `timesbd.ttf`, `timesi.ttf`, `simsun.ttf`); then run `python docs/export_pdf.py`. Font binaries are intentionally excluded from version control.

## Start/continue the v5 baseline stage

```bash
python -m exp.baseline.campaign --pairs data/pairs/v5/Video-MME.json \
  --media-map data/pairs/v5/Video-MME.media.json --run v5-videomme-fulltokens --prep-workers 6
```

The controller keeps a complete source manifest, runs ready Full Tokens jobs on available GPUs, and retains unsupported preparation records with reasons. Downloads run separately. Do not start a second controller for the same run; a process lock prevents that. Model-specific compressed matching is gated on complete auxiliary-cost calibration. Main run resumption must use the same manifest, media map and plan hash.
