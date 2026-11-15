# DSS Stage 6.3A — XGBoost Static/Time-Aware Local Smoke

Run UTC: `2026-11-15T13:56:22.986798+00:00`  
Status: **READY_FOR_COLAB**  
Full GĐ6.3 complete: **NO**

This laptop run is implementation validation only. It uses data-cell `53021/S0`, at most two pre-locked trials per model, and a bounded validation-session subset. It does not select the final hyperparameter configuration.

## Local smoke metrics (informational only)

| Model | Trial | Sessions | NDCG@10 | Recall@10 |
|---|---|---:|---:|---:|
| static | trial_01 | 40 | 0.201000 | 0.387500 |
| static | trial_02 | 40 | 0.152564 | 0.312500 |
| time | trial_01 | 40 | 0.245873 | 0.508333 |
| time | trial_02 | 40 | 0.246267 | 0.454167 |

## Acceptance

| Check | Status | Result |
|---|---:|---|
| prerequisites | **PASS** | GĐ6.0/GĐ6.1/GĐ6.2 PASS and CBF config locked |
| locked_trials | **PASS** | 20 deterministic trials were persisted before training and locked hashes stayed unchanged |
| feature_schema | **PASS** | Static contains F0+F1 only; time contains F0+F1+F2 |
| leakage | **PASS** | Feature future/current-session violations=0 |
| train_only_fit | **PASS** | Encoder/scaler/CBF vocabulary fit on train and unchanged by validation |
| coverage | **PASS** | Every smoke prediction covers exactly the exposed validation pairs |
| shared_evaluator | **PASS** | NDCG@10 and Recall@10 produced by shared RankingEvaluator |
| determinism | **PASS** | Repeated seed-42 training produced byte-identical score arrays |
| checkpoint_roundtrip | **PASS** | Every smoke checkpoint reload produced identical scores |
| tests | **PASS** | Full pytest exit=0 |
| colab_package | **PASS** | Colab runner, pinned requirements and guide are present |

## Boundary

No test predictions were opened. No final configuration was selected. Full 20×2 trial tuning and six-cell validation must run on Colab before GĐ6.3 can be completed.

## Colab hand-off

Use `notebooks/STAGE6_3_COLAB_RUNNER.ipynb` and `docs/STAGE6_3_COLAB_GUIDE.md`. The runner verifies Git commit and dataset manifest, reruns tests, supports per-trial resume, and writes outputs to Google Drive.
