# DSS Stage 6.2 — Random, Popularity and Content-Based Filtering

Run UTC: `2026-11-15T13:32:19.644593+00:00`  
Overall: **PASS** — **10/10 PASS**  
Selected common CBF configuration: `cbf_05_skills3_occupation2`

Validation uses Δ14 exposed-set ranking over all six seed×scenario data cells. No test predictions were opened and no dataset or locked protocol file was modified.

## Acceptance

| Check | Status | Result |
|---|---:|---|
| prerequisites | **PASS** | GĐ6.0 and GĐ6.1 acceptance are PASS |
| random | **PASS** | Seed 42 sanity scorer is deterministic and scores exposed jobs only |
| popularity_asof_t | **PASS** | Scores count apply events from shared strict as-of-t HistoryStore with no tuning; violations=0 |
| cbf_train_only | **PASS** | Six F0 configurations share train-fitted vocabularies that remain unchanged on validation |
| six_cell_validation | **PASS** | Random, Popularity and selected CBF evaluated on validation Δ14 in all six cells |
| prediction_coverage | **PASS** | Every prediction file exactly equals its evaluated exposed candidate pairs |
| shared_evaluator | **PASS** | All metrics were produced through the shared RankingEvaluator |
| determinism | **PASS** | Repeated scores match for every model and data cell |
| tests | **PASS** | Full pytest exit=0 |
| locked_inputs | **PASS** | Dataset was read-only and locked protocol/experiment hashes are unchanged |

## Validation macro metrics

| Model | Config | Cohort | NDCG@10 | Recall@10 | Sessions |
|---|---|---|---:|---:|---:|
| cbf | cbf_05_skills3_occupation2 | all | 0.237420 | 0.447249 | 1537 |
| cbf | cbf_05_skills3_occupation2 | changed_profile | 0.227044 | 0.429801 | 172 |
| cbf | cbf_05_skills3_occupation2 | low_history | 0.310367 | 0.536180 | 79 |
| cbf | cbf_05_skills3_occupation2 | new_job | 0.232038 | 0.466590 | 989 |
| popularity_asof_t | no_tuning | all | 0.218315 | 0.423033 | 1537 |
| popularity_asof_t | no_tuning | changed_profile | 0.201046 | 0.390648 | 172 |
| popularity_asof_t | no_tuning | low_history | 0.238359 | 0.451990 | 79 |
| popularity_asof_t | no_tuning | new_job | 0.014334 | 0.045282 | 989 |
| random | seed_42 | all | 0.226339 | 0.433953 | 1537 |
| random | seed_42 | changed_profile | 0.246166 | 0.492120 | 172 |
| random | seed_42 | low_history | 0.216303 | 0.424194 | 79 |
| random | seed_42 | new_job | 0.214534 | 0.446829 | 989 |

## CBF schema limitation

The frozen Stage 5 contract contains no job title or description and forbids reconstructing free text. CBF therefore uses every permitted F0 field actually present: skills, occupation/target, location and experience. Salary and all behavioral/time features are excluded. Synthetic candidates without source demographics receive the explicit contract value `unknown`; their as-of-t profile skills remain versioned.

## Selection rule

Six configurations were evaluated. One common configuration was selected by macro NDCG@10 over the six validation cells, then Recall@10, then lower declared complexity, then config ID for deterministic resolution.

## Decision

GĐ6.2 PASS. Random, Popularity as-of-t and one common selected CBF configuration are deterministic, leakage-safe and evaluated through the shared engine.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\run_stage6_2.py
.\.venv\Scripts\python.exe -m pytest -q
```
