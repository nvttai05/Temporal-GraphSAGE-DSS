# DSS Stage 6 Data & Evaluation Preflight

Run UTC: `2026-11-15T10:23:34.983757+00:00`  
Dataset: `medajob-temporal-v0.3` (frozen, read-only)  
Overall: **PASS** — **6/6 PASS**  
Eligible for GĐ6.1: **YES**

No model was trained and no prediction file was opened. Δ14 is the primary window; Δ7 was used only to verify handoff/ranking consistency.

## Gate results

| Check | Status | Evidence-backed result |
|---|---:|---|
| 1. Manifest and handoff SHA-256 | **PASS** | 143/143 checksum records matched; checksum invariants PASS Evidence: `artifacts/preflight/checksum_verification.csv`, `artifacts/preflight/checksum_invariants.json`. |
| 2. Loader coverage and future-free graph snapshots | **PASS** | Loaded 36 expected seed/scenario/delta/split cells; 36 deterministic snapshots audited with 0 violations Evidence: `artifacts/preflight/loader_coverage.csv`, `artifacts/preflight/leakage_deterministic_samples.csv`. |
| 3. Label and Top-K eligibility statistics | **PASS** | Recorded positive/negative/censored and Top-10 eligibility for 72 primary-window seed/scenario/split/cohort groups Evidence: `artifacts/preflight/label_and_eligibility_statistics.csv`. |
| 4. Hand-calculated Recall@10/NDCG@10 fixture | **PASS** | Hand fixture Recall@10=0.500000000000, NDCG@10=0.386852807235; full pytest exit=0 Evidence: `artifacts/preflight/metric_fixture.json`, `artifacts/preflight/full_test_suite.log`. |
| 5. Exposed-set ranking and rolling as-of-t compliance | **PASS** | Audited exposed-set membership, eligible full candidate sets, and rolling policy across 12 seed/scenario/window cells Evidence: `artifacts/preflight/protocol_compliance.csv`, `artifacts/preflight/leakage_deterministic_samples.csv`, `artifacts/preflight/full_test_suite.log`. |
| 6. Experiment contract completeness | **PASS** | Model list, applicable feature/loss definitions, seeds, and tuning budgets are explicit Evidence: `artifacts/preflight/experiment_contract_completeness.csv`. |

## Decision

All six gates passed; the repository is eligible to proceed to GĐ6.1.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\run_stage6_preflight.py
.\.venv\Scripts\python.exe -m pytest -q
```

Machine-readable gate output: `artifacts/preflight/preflight_results.json`. Run metadata and hashes are stored beside it.
