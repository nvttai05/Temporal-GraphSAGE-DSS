# DSS Stage 6.1 — Shared Data Pipeline & Evaluation Engine

Run UTC: `2026-11-15T10:34:57.417374+00:00`  
Overall: **PASS** — **8/8 PASS**  
Eligible for model implementation: **YES**

This acceptance run trained no recommendation model and opened no test predictions. The frozen dataset and locked protocol/config files were not modified.

## Acceptance results

| Check | Status | Result |
|---|---:|---|
| 1. DatasetIndex protocol lattice | **PASS** | Indexed 6 data cells and 36 seed/scenario/split/window slices Evidence: `artifacts/stage6_1/six_cell_smoke.csv`. |
| 2. SessionLoader exposed labels/cohorts | **PASS** | Loaded binary labels and cohort-bearing exposed sets for one scoreable validation session in every data cell Evidence: `artifacts/stage6_1/six_cell_smoke.csv`. |
| 3. HistoryStore rolling as-of-t | **PASS** | Rolling history used strict action_at<t and observed_at<t with zero future-edge violations Evidence: `artifacts/stage6_1/six_cell_smoke.csv`, `artifacts/stage6_1/full_test_suite.log`. |
| 4. FeatureBuilder train-only fit | **PASS** | FeatureBuilder fit on train and retained identical state across validation transforms Evidence: `artifacts/stage6_1/six_cell_smoke.csv`, `artifacts/stage6_1/full_test_suite.log`. |
| 5. StaticGraphBuilder leakage exclusion | **PASS** | Static graphs contain only observable prior actions and exclude current-session edges Evidence: `artifacts/stage6_1/six_cell_smoke.csv`. |
| 6. Shared RankingEvaluator | **PASS** | One shared exposed-set evaluator produced finite NDCG@10 and Recall@10 for all six cells with job_id ASC tie-breaking Evidence: `artifacts/stage6_1/six_cell_smoke.csv`, `artifacts/stage6_1/full_test_suite.log`. |
| 7. RunRegistry provenance | **PASS** | Run receipt stores config hash, code version, seeds, environment, runtime and artifact hashes Evidence: `artifacts/stage6_1/run_registry/stage6_1_20261115T103511297422Z.json`. |
| 8. Tests and deterministic six-cell smoke | **PASS** | Full pytest exit=0; repeated six-cell smoke digest=6aac6ab88189e9f3f5e95830312051cb17de0d272b360322fae27ee1636c99fb Evidence: `artifacts/stage6_1/full_test_suite.log`, `artifacts/stage6_1/six_cell_smoke.csv`. |

## Deterministic smoke scope

The smoke uses one scoreable validation Δ14 session in each of the six seed×scenario cells. Scores are a stable hash fixture, not a trained model. The complete smoke is executed twice and the tables must be byte-equivalent under canonical CSV serialization.

## Decision

GĐ6.1 PASS. The shared pipeline is deterministic, leakage-safe, and ready for all GĐ6 models to consume the common SessionLoader and RankingEvaluator.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\run_stage6_1_acceptance.py
.\.venv\Scripts\python.exe -m pytest -q
```
