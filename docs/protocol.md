# Protocol v4

The authoritative planned settings are `experiment_plan.pdf` and `exp/baseline/configs/experiment.yaml`. Six datasets, two model sizes, two final AV-token budgets; all text survives, one greedy answer per question pair, 32 output tokens maximum. Shared input cap is 32,256 tokens. Budget tolerance is absolute 0.005; unmatched samples are excluded from equal-budget claims and remain explicit failures.

Each GPU worker writes separate, atomic pair results. A job lock prevents concurrent writers to one shard. A ready-input queue controls GPU allocation, retries failed shards at most once per invocation and saves utilization/process mappings and per-GPU completed shard counts. Existing passed outputs are reused only when their job configuration, source-code hash, model configuration and pair-manifest hash match. Detailed provenance and full-input validation remain prerequisites for publication-quality full evaluation.

The repository validation intentionally uses one short, real pair and eight frames per stream. It does not change the main dataset caps and is not an accuracy experiment. Earlier failed budget records are preserved in their original run; corrected records are in a separate run. No old experiment results were deleted.
