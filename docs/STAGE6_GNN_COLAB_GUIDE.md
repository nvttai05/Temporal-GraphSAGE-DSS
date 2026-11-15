# Stage 6 Static vs Temporal GraphSAGE — Colab guide

This guide runs the fast-track comparison without changing `medajob-temporal-v0.3` or `configs/stage6_protocol.yaml`. Smoke and validation are separate evidence levels. Test remains closed until a six-cell validation config lock exists.

## Required preparation

1. Commit and push the GNN implementation on `feat/stage6-temporal-graphsage`.
2. Copy the frozen dataset to Drive. Its `dataset_manifest.json` SHA-256 must be `a6d983e754f49a9e2076022d48109516a931def7c17b1a725394245872aad287`.
3. Open `notebooks/STAGE6_GNN_COLAB_RUNNER.ipynb`, set the exact commit, repository URL, dataset root and output root.
4. Select a GPU runtime. The notebook still fails unless both `torch.cuda.is_available()` and `nvidia-smi` succeed; selecting a GPU in the UI is not evidence by itself.

## Exact runner commands

Smoke, same `53021/S1` cell for both M0 and M1:

```bash
python scripts/run_stage6_gnn.py \
  --mode smoke \
  --device cuda \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_ROOT/smoke" \
  --resume
```

Six-cell validation with model seed 42 and Temporal half-lives 7/14/30:

```bash
python scripts/run_stage6_gnn.py \
  --mode validation \
  --device cuda \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_ROOT/validation" \
  --model-seed 42 \
  --half-life-days 7 14 30 \
  --resume
```

`--models` defaults to `static temporal`. To run/resume the families separately
while retaining model-specific checkpoint directories:

```bash
python scripts/run_stage6_gnn.py --mode validation --models static \
  --device cuda --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_ROOT/validation_static" --model-seed 42 --resume

python scripts/run_stage6_gnn.py --mode validation --models temporal \
  --half-life-days 7 14 30 --device cuda --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_ROOT/validation_temporal" --model-seed 42 --resume
```

A single-family validation run does not create `gnn_config_lock.json`. The lock
gate remains reserved for a combined six-cell M0/M1 comparison.

Only after `validation/gnn_config_lock.json` reports `LOCKED_FROM_SIX_CELL_VALIDATION`, run final test once:

```bash
python scripts/run_stage6_gnn.py \
  --mode final \
  --device cuda \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_ROOT/final_seed42" \
  --model-seed 42 \
  --config-lock "$OUTPUT_ROOT/validation/gnn_config_lock.json" \
  --resume
```

Seeds 43/44 are optional follow-ups only if resources remain. Never retune after test.

## Evidence rules

Every completed model/cell directory contains a checkpoint, train curve CSV/PNG, session metrics, predictions, graph leakage audit and run manifest with parameter/epoch/runtime/GPU memory. `model_comparison_validation.csv` is preliminary validation evidence only. Do not claim M1 is better unless the same six cells completed for M0 and the selected M1 configuration.
