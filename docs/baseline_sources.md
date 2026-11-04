# Baseline sources and implementation lineage

The experiment uses AV adaptations, not official single-video leaderboard scores. Immutable implementation commits are in `manifests/baselines.yaml`.

## full_tokens

Local control / proposed method; no separate paper is claimed.

Implementation: unmodified Qwen thinker.



## random

Local control / proposed method; no separate paper is claimed.

Implementation: per-modality stable seeded sampling.



## fastv

[Paper](https://arxiv.org/abs/2403.06764) · [PDF](https://arxiv.org/pdf/2403.06764) · [Original code](https://github.com/pkunlp-icler/FastV)

Implementation: OmniPack/baselines/fastv + fastv_omni.

joint layer-5 AV selection, final budget (no layer-average conversion); native KV pruning

## visionzip

[Paper](https://arxiv.org/abs/2412.04467) · [PDF](https://arxiv.org/pdf/2412.04467) · [Original code](https://github.com/JIA-Lab-research/VisionZip)

Implementation: OmniPack/baselines/visionzip.

90% salience / 10% contextual retained budget; audio uniform

## fastvid

[Paper](https://arxiv.org/abs/2503.11187) · [PDF](https://arxiv.org/pdf/2503.11187) · [Original code](https://github.com/LunarShen/FastVID)

Implementation: official Qwen2.5-VL core.

bounded short-frame segmentation; DySeg_ignore=1.0; retain merged features and representative positions

## vidcom2

[Paper](https://arxiv.org/abs/2505.14454) · [PDF](https://arxiv.org/pdf/2505.14454) · [Original code](https://github.com/xuyang-liu16/VidCom2)

Implementation: official omni branch/token_compressor/vidcom2.

per-stream video selection; audio uniform; label-blind count calibration

## dycoke_ttm

[Paper](https://arxiv.org/abs/2411.15024) · [PDF](https://arxiv.org/pdf/2411.15024) · [Original code](https://github.com/KD-TAO/DyCoke)

Implementation: four-frame temporal ranking adapted from VidCom2 omni TTM port.

one anchor frame per group; exact remaining integer counts allocated across non-anchor frames in stable order; no posthoc truncation; no dynamic decode KV compression; infeasible anchor floors fail

## omnizip

[Paper](https://arxiv.org/abs/2511.14582) · [PDF](https://arxiv.org/pdf/2511.14582) · [Original code](https://github.com/KD-TAO/OmniZip)

Implementation: OmniPack/baselines/omnizip + local partial-group core.

audio-priority starting rho pairs (.30,.60) / (.40,.70); common offset count calibration including anchors

## omniselect

[Paper](https://arxiv.org/abs/2605.18041) · [PDF](https://arxiv.org/pdf/2605.18041) · [Original code](https://github.com/Yangmrl-nlp/OmniSelect)

Implementation: official OmniSelect + tracked pooling patch.

AudioCLIP scoring, NLTK keyword filtering; pool by actual contiguous token groups; feature extraction microbatch=8

## omnipack

[Paper](https://arxiv.org/abs/2608.03812) · [PDF](https://arxiv.org/pdf/2608.03812) · [Original code](https://github.com/RowanSu/OmniPack)

Implementation: official two-stage compressor.

stage-I 2*r, stage-II .5; joint stage-II at 18 (7B) / 26 (3B)

## fro

Local control / proposed method; no separate paper is claimed.

Implementation: src/fro/scoring.py.

independent prefix score_layer=2; same-modality cross-stream support; original embedding replay; missing modalities receive no token allocation

## Supporting models and second-level sources

- [Qwen2.5-Omni paper](https://arxiv.org/abs/2503.20215) · [code and model links](https://github.com/QwenLM/Qwen2.5-Omni). Used by all methods.
- [AudioCLIP paper](https://arxiv.org/abs/2106.13043) · [code and weights](https://github.com/AndreyGuzhov/AudioCLIP). Used by OmniSelect.
- [CLIP paper](https://arxiv.org/abs/2103.00020) · [code](https://github.com/openai/CLIP). AudioCLIP visual/text parent.
- [ESResNe(X)t-fbsp code](https://github.com/AndreyGuzhov/ESResNeXt-fbsp) and [ESResNet code](https://github.com/AndreyGuzhov/ESResNet). AudioCLIP audio-model lineage, as linked in its official README.
- [DyCoke](https://github.com/KD-TAO/DyCoke) → [VidCom2 omni port](https://github.com/xuyang-liu16/VidCom2) → this repository’s explicit integer-budget TTM adaptation.
- [VisionZip](https://github.com/JIA-Lab-research/VisionZip), [FastV](https://github.com/pkunlp-icler/FastV), [OmniZip](https://github.com/KD-TAO/OmniZip) → [OmniPack Qwen port](https://github.com/RowanSu/OmniPack) → this paired-input adapter.

Benchmark references are grouped first, followed by methods, in chronological order on the final page of `experiment_plan.pdf`; structured entries are in `references.json`.
