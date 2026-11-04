# Baseline experiments

Run with `python -m exp.baseline.run`; preparation, evaluation and reports have corresponding module entry points. Use the top-level README and `docs/reproduction.md` for download and execution commands. All experiment-specific code and configs are contained here; FRO's reusable scorer is in `src/fro/`.

Official sources are fetched separately with `scripts.fetch_baselines`. Never substitute a method with an approximation without recording the adaptation in `manifests/methods.json`. Runtime outputs record measured token counts; budget mismatch is a failure, not a passing equal-budget result.
