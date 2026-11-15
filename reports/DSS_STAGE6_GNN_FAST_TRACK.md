# DSS Stage 6 — Static vs Temporal GraphSAGE fast-track

Date: 2026-11-15  
Branch: `feat/stage6-temporal-graphsage`  
Inspected HEAD: `47121aad34e8dcc6632573bb1aee34d156747717`

## Status

**READY_FOR_COLAB_VALIDATION**, not a completed Stage 6 GNN benchmark.

The local CPU smoke passed for M0 Static GraphSAGE and M1 Temporal GraphSAGE with 14-day exponential decay on the required `53021/S1` cell. Six-cell validation, CUDA verification, configuration lock, and final test remain pending. Test predictions were not opened.

The repository was not rebasing or merging and had no conflicts before implementation. The only initial working-tree change was the untracked user-provided plan `docs/DSS_STAGE6_TEMPORAL_GRAPHSAGE_FAST_TRACK_PLAN.md`. Existing Stage 6.3 results were not modified or regenerated.

## Protocol and implementation evidence

- `medajob-temporal-v0.3` remained read-only; manifest SHA-256 recorded by the run is `a6d983e754f49a9e2076022d48109516a931def7c17b1a725394245872aad287`.
- `configs/stage6_protocol.yaml` was not changed. Its recorded SHA-256 is `298d157d13ced167cd2e85e85730149f4906776b25eb5a49d62c3bc8adddbf2a`.
- `configs/experiments.yaml` was bumped traceably from `stage6-v1` to `stage6-v2`; the data/ranking protocol was not changed.
- M0 and M1 use the same two-layer relation-aware encoder, scorer, weighted BCE loss, optimizer, seed, and evaluator. M1 differs only by exponential weights on event-interaction messages. Candidate/job skill-membership edges always have weight 1.
- Aggregation uses PyTorch `index_add_`; it does not pass `edge_weight` to PyG `SAGEConv`.
- Every training session is loaded with `eligible_only=True`; censored sessions cannot reach the loss. Evaluation ranks only the exposed candidate set with the shared `dss_stage6.evaluation.RankingEvaluator`.
- Each graph snapshot applies strict `action_at < t`, `observed_at < t`, and excludes current-session actions and impressions. The local smoke graph audits recorded zero violations.

## Local smoke results

These are prototype smoke measurements on four validation sessions after eight train sessions and three epochs. They are not six-cell validation evidence and must not be used to select or lock a configuration.

| Evidence scope | Model | Cell | Eligible sessions | NDCG@10 | Recall@10 | Parameters | Best epoch | Train runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Smoke only | M0 Static GraphSAGE | 53021/S1 | 4 | 0.269869 | 0.750 | 125,441 | 1 | 2.953 s |
| Smoke only | M1 Temporal GraphSAGE, half-life 14 | 53021/S1 | 4 | 0.314179 | 0.875 | 125,441 | 1 | 1.063 s |

Both runs covered exactly 95 exposed `(session_id, job_id)` pairs, had zero duplicate predictions, restored checkpoint scores exactly, used model seed 42, and opened no test prediction. The numerical difference above does **not** establish that M1 is better.

Primary evidence:

- `artifacts/gnn_smoke/local_cpu/per_cell_results.csv`
- `artifacts/gnn_smoke/local_cpu/model_comparison_validation.csv`
- `artifacts/gnn_smoke/local_cpu/M0_static_graphsage/seed_53021_S1/acceptance_audit.json`
- `artifacts/gnn_smoke/local_cpu/M1_temporal_graphsage_hl14/seed_53021_S1/acceptance_audit.json`
- `artifacts/gnn_smoke/local_cpu/run_registry/`
- `artifacts/gnn_smoke/local_cpu/stage6_gnn_smoke_acceptance.json`

## Tests

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result: **39 passed, 1 skipped, 0 failed** in 23.13 seconds. The skipped test is the CUDA device-placement test because the local PyTorch build is CPU-only. The full log is `artifacts/gnn_smoke/local_cpu/pytest.log` with SHA-256 `9c3e967aa624044bc31034dc92d61c1c648229f8a2e4ae18e6eaf44640991b31`.

Covered checks include future/current-session injection, strict boundary handling, decay monotonicity and half-life, all-ones equivalence, finite loss/non-zero gradient, deterministic seed 42 initialization/scores, checkpoint restore, shared metric correctness, and exposed-pair coverage. CUDA remains pending until Colab.

## Six-cell validation table

| Model/config | 53021/S0 | 53021/S1 | 53021/S2 | 53022/S0 | 53022/S1 | 53022/S2 | Macro NDCG@10 | Status |
|---|---|---|---|---|---|---|---|---|
| M0 Static GraphSAGE | — | — | — | — | — | — | — | Pending Colab CUDA |
| M1, half-life 7 | — | — | — | — | — | — | — | Pending Colab CUDA |
| M1, half-life 14 | — | — | — | — | — | — | — | Pending Colab CUDA |
| M1, half-life 30 | — | — | — | — | — | — | — | Pending Colab CUDA |

No configuration lock was written locally. No final/test run is authorized until this table is complete and `gnn_config_lock.json` reports `LOCKED_FROM_SIX_CELL_VALIDATION`.

## Exact commands

Local smoke:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage6_gnn.py --mode smoke --device cpu --output-dir artifacts\gnn_smoke\local_cpu
```

Colab validation after committing and pushing the implementation:

```bash
python scripts/run_stage6_gnn.py --mode validation --device cuda --dataset-root "$DATASET_ROOT" --output-dir "$OUTPUT_ROOT/validation" --model-seed 42 --half-life-days 7 14 30 --resume
```

The notebook `notebooks/STAGE6_GNN_COLAB_RUNNER.ipynb` mounts Drive, checks out an exact commit, verifies the dataset manifest, installs pinned dependencies, runs all tests, requires both PyTorch CUDA and `nvidia-smi`, runs smoke, and then runs six-cell validation. The gated final command is documented in `docs/STAGE6_GNN_COLAB_GUIDE.md` and must not be used before validation lock review.

## Baseline reuse

Existing CBF and XGBoost outputs were detected but are marked unverified in `artifacts/gnn_smoke/local_cpu/baseline_evidence.csv`: the available evidence does not establish the same commit/manifest six-cell protocol. No baseline numbers were invented or copied into the comparison.
