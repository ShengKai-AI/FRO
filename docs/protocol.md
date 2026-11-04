# Protocol v5: matched FLOPs

The authoritative plan is the eight-page `experiment_plan.pdf`, copied from `20261104_v5_matched_flops`. Its SHA256 is recorded in `plan_metadata.json` and every main-run job. A/B targets are FRO's analytical FLOPs† at 45%/35% final retention; other methods have distinct pre-LLM and final retentions in `matched_budgets.json`.

The reference convention is dense matrix FLOPs, GQA dimensions, 1 MAC=2 FLOPs, 100 decode steps, and explicitly counted FRO prefix/QK/cross-stream matrices. It excludes the shared encoders, LM head and listed elementwise operations. AudioCLIP and other selection/merging delta must be measured before complete-cost compressed main runs. Those runs are gated; only Full Tokens is currently authorized by code to execute as a main stage.

Full Tokens actual cost uses observed per-layer prefill and KV lengths, replacing reference KV assumptions. Generation remains greedy, once per pair, with at most 32 actual output tokens. Shared input cap is 32,256; all text survives. Main preparation does not silently drop audio, truncate long clips or change frame-only budgets when joint resampling is required.

The full Video-MME manifest preserves all 2,700 unique scored question IDs and 1,350 fixed pairs. CPU preparation, pending weight downloads and GPU inference have separate queues. Blocked pairs and failed jobs remain in progress reports and denominators. Outputs are atomic, individually fingerprinted and recoverable. Old runs and the v4 integration evidence are retained.
