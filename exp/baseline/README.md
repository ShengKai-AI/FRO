# Native two-video baseline experiments

- Build fixed subsets: `python -m scripts.build_native_pilot MVU-Eval SYNCR CVBench CrossVid --limit 200`.
- Run: `python -m scripts.run_native_pilot`.
- Plan: `configs/native_pilot.yaml`; eight baseline methods, no FRO jobs.
- Preparation: `native_prepare.py`; one native question with two ordered video inputs.
- GPU queue: `suite.py`; resident consumers: `resident.py` and `worker.py`.
- Score: `python -m scripts.native_pilot_results --run RUN --dataset-dir DATASET_DIR`.
- Measured compute tolerance: 0.5%; per-sample token counts and costs remain in every output.
- Original results and frozen historical code are retained locally; single-video question-pairing code has been removed from the active tree.
