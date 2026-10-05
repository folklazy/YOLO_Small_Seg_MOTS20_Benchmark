# Small execution and saved-result documentation

Use the existing workspace `.venv/bin/python` directly. The owning paths are resolved from the scripts; shared datasets/checkpoints/environments are not copied into this experiment.

Completed run: `benchmark-20261005T051531Z`. `run_small.py --all --run-id <fresh-id>` performs singleton Small-only execution: shared validation, all-model preflight/maxDet, sequential accuracy, clean timing, canonical conversion, reports and final validation. This command starts a new benchmark and must not be rerun for documentation or after Small completion. No Nano runner is invoked.

`benchmark_adapter.py` and `frozen_pilot/` preserve the Largest reference preprocessing/evaluator. Runtime source hashes and frozen input archives identify the exact implementation used. `timing_small.py` reuses the reference measurement loop and retains contaminated attempts separately; only three clean rounds per model enter primary metrics.

`build_small_results.py` validates saved artifacts and creates canonical CSVs without inference. `report_small.py <run-id>` renders CSV-derived reports using the user-requested Small layout snapshots under `configs/report_templates/`. `finalize_small_documentation.py` applies final editorial interpretation and checks that measured artifacts remain unchanged; its claim/hash provenance is in `manifests/FINAL_DOCUMENT_REVIEW.json`.

Lossless predictions and verbose logs remain local/ignored. Canonical metrics, reports, plots, source, configuration and provenance are versioned. Pipeline excludes RLE preparation and disk I/O. Small is complete: STOP and wait for user approval; do not begin Nano.
