# Stage 6.3 XGBoost Colab guide

GĐ6.3A only establishes local implementation readiness. Final model selection is valid only after the full Colab run completes 20 locked trials for both `xgboost_static` and `xgboost_time` across all six validation data cells.

## Before opening Colab

1. Commit the complete GĐ6.3A implementation and record the exact commit SHA.
2. Push that commit to the remote used by Colab.
3. Copy the frozen `medajob-temporal-v0.3` package to Google Drive without changing any bytes.
4. Confirm the SHA-256 of `data/candidate55/v0_3/dataset_manifest.json` is `a6d983e754f49a9e2076022d48109516a931def7c17b1a725394245872aad287`.
5. In the notebook, replace `REPLACE_WITH_STAGE6_3A_COMMIT` and `REPLACE_WITH_REPOSITORY_URL` before running any training cell.

## Execution

Use `notebooks/STAGE6_3_COLAB_RUNNER.ipynb`. It mounts Drive, checks out and verifies the exact commit, verifies the dataset manifest, installs `requirements-colab.txt`, runs pytest, and launches:

```bash
python scripts/run_stage6_3.py \
  --mode colab-full \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --resume
```

`--resume` treats a trial/cell receipt as the completion marker. An interrupted run skips completed receipts and resumes the remaining work. Do not edit `configs/stage6_3_xgboost.yaml` after viewing any result. The config contains all 20 deterministic trials generated with seed 42.

## Required Drive outputs

- Per model/trial/data-cell `model.json`, `predictions.csv`, and `receipt.json`.
- `all_trial_cell_metrics.csv` and `tuning_summary.csv`.
- Test log, environment receipt, config/code/dataset hashes, and artifact hashes.
- Six-cell validation metrics for both model families.

Do not run test evaluation, merge validation into train, or claim GĐ6.3 completion until all 240 trial-cell fits (2 models × 20 trials × 6 cells) have valid receipts.
