# Third-party sources

Official checkouts are downloaded into ignored `third_party/`; immutable commits are in `manifests/baselines.yaml`. Their respective licenses apply. This repository does not grant a new license to upstream code or model/data assets.

- `src/fro/rotary.py`: Qwen/Hugging Face mRoPE helpers, derived from the Qwen2.5-Omni implementation in pinned OmniPack. Copyright 2025 The Qwen team, Alibaba Group and HuggingFace Inc.; Apache-2.0.
- `exp/baseline/methods/fastvid_core.py`: extracted official FastVID Qwen2.5-VL core; its MIT license is retained in `docs/FastVID-LICENSE.txt`. Short-frame boundaries and duplicate-frame threshold adapted as recorded in the method manifest.
- `exp/baseline/methods/omnizip_core.py`: adapted OmniPack port of OmniZip, including partial-group handling and chunked similarity; see pinned upstream repositories and their source notices.
- `exp/baseline/methods/omniselect.patch`: explicit patch against the pinned OmniSelect commit. AudioCLIP retains its own MIT license in the fetched repository.
- Times New Roman and SimSun font files are local document-build inputs and are excluded from Git. The requested existing PDF is retained. Supply licensed fonts locally when rebuilding it.

Method papers, original repositories, ports, and second-level dependencies are listed in `docs/baseline_sources.md` and `manifests/methods.json`.
