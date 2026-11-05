# Protocol v5.1: matched FLOPs

The authoritative plan is the eight-page `experiment_plan.pdf`, copied from `20261104_v5_matched_flops`. VisionZip and VidCom² were removed on user instruction; the unchanged Full Tokens results from v5 remain compatible. LongVideoBench is deferred on user instruction. Its SHA256 is recorded in `plan_metadata.json` and every main-run job. A/B targets are FRO's analytical FLOPs† at 45%/35% final retention; other methods have distinct pre-LLM and final retentions in `matched_budgets.json`.

The reference convention is dense matrix FLOPs, GQA dimensions, 1 MAC=2 FLOPs, 100 decode steps, and explicitly counted FRO prefix/QK/cross-stream matrices. It excludes the shared encoders, LM head and listed elementwise operations. Per-input calibration counts AudioCLIP, additional encoder attention statistics, and selection/merging dense products. Compressed outputs pass only when complete counted FLOPs differ from the input-specific FRO target by at most 0.5%. Discarded calibration trials remain in the trace and their wall time remains included in measured latency.

Full Tokens actual cost uses observed per-layer prefill and KV lengths, replacing reference KV assumptions. Generation remains greedy, once per pair, with at most 32 actual output tokens. Shared input cap is 32,256; all text survives. Main preparation does not silently drop audio, truncate long clips or change frame-only budgets when joint resampling is required.

The full Video-MME manifest preserves all 2,700 unique scored question IDs and 1,350 fixed pairs. CPU preparation, pending weight downloads and GPU inference have separate queues. Blocked pairs and failed jobs remain in progress reports and denominators. Outputs are atomic, individually fingerprinted and recoverable. Old runs and the v4 integration evidence are retained.

Answer parsing uses a deterministic parser shared by every method: explicit Q1/Q2 labels on one or two lines, or an exact ordered pair. Multiple choices and ambiguous text are invalid. This requires no LLM judge and never consults gold labels while parsing.

`python -m exp.baseline.complete_suite` runs Video-MME, DailyOmni, AVUT, WorldSense, and MVBench in sequence. Each GPU has a private resident process and output shard. CPU preparation has 16 slots, each shared input has 34 required consumers, and cached inputs retire only after all consumers pass. A stalled worker is restarted after 20 minutes without an output; persistent failures remain explicit. Original media are retained for failed datasets.
