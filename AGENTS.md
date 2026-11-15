# DSS Stage 6 — Codex Instructions

This repository implements Stage 6 of the DSS temporal job-recommendation project.

## Non-negotiable protocol

- Treat `medajob-temporal-v0.3` as frozen and read-only.
- Do not regenerate, relabel, move split boundaries, or change exposure policy.
- Primary label window is 14 days. Seven days is sensitivity analysis only.
- Primary ranking protocol is exposed-set ranking.
- Fit vocabularies, scalers, encoders and model parameters on train only.
- Validation may select configuration. Test must remain unopened until the configuration is locked.
- At prediction time `t`, use only records satisfying the as-of-t rules in `configs/stage6_protocol.yaml`.
- Never turn censored or unexposed jobs into negative labels.
- Never report PASS without a machine-readable log or table that supports it.

## Engineering rules

- Use the shared evaluator in `src/dss_stage6/evaluation`; models must not implement private metric variants.
- Store every run's config, code commit, seeds, environment and output hashes.
- Keep laptop work to development, preflight and smoke tests. Official results are produced on Colab.
- Ask before changing any locked item in `configs/stage6_protocol.yaml` or `configs/experiments.yaml`.
- GĐ6 contains Random, Popularity, CBF, XGBoost-static, XGBoost-time and Static GraphSAGE only.
- Temporal GraphSAGE, TGAT and TGN belong to GĐ7.

## Required GĐ6.0 outputs

- `reports/DSS_STAGE6_DATA_PREFLIGHT.md`
- `artifacts/preflight/preflight_results.json`

The initial status is 0/6. A failed or unavailable check must be recorded as `FAIL` or `BLOCKED`, not silently skipped.

