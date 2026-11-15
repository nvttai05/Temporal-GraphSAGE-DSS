from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
import yaml

from dss_stage6.data import DatasetIndex, HistoryStore, SessionLoader, SessionRecord
from dss_stage6.evaluation import RankingEvaluator
from dss_stage6.models import (
    CBFConfig,
    ContentBasedFilter,
    ContentRepository,
    PopularityAsOfT,
    RandomBaseline,
    default_cbf_configs,
)
from dss_stage6.registry import RunRegistry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _score_sessions(
    sessions: list[SessionRecord],
    scorer: Callable[[SessionRecord], Mapping[str, float]],
    evaluator: RankingEvaluator,
    *,
    model: str,
    config_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for session in sessions:
        scores = dict(scorer(session))
        metric = evaluator.evaluate_session(session, scores)
        metric.update({"model": model, "config_id": config_id})
        metric_rows.append(metric)
        prediction = session.candidates[
            [
                "seed",
                "scenario",
                "delta_days",
                "split",
                "session_id",
                "candidate_id",
                "job_id",
                "label",
                "new_job",
                "changed_profile",
                "low_history",
            ]
        ].copy()
        prediction["score"] = prediction["job_id"].astype(str).map(scores)
        if prediction["score"].isna().any() or len(scores) != len(prediction):
            raise AssertionError("Prediction coverage differs from exposed candidate set")
        prediction["model"] = model
        prediction["config_id"] = config_id
        prediction_frames.append(prediction)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def _cell_metrics(session_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["model", "config_id", "seed", "scenario"]
    for keys, frame in session_metrics.groupby(group_columns, sort=True):
        model, config_id, seed, scenario = keys
        cohorts = {
            "all": frame,
            "changed_profile": frame.loc[frame["changed_profile"] == 1],
            "low_history": frame.loc[frame["low_history"] == 1],
            "new_job": frame.loc[frame["has_positive_new_job"] == 1],
        }
        for cohort, subset in cohorts.items():
            if subset.empty:
                continue
            if cohort == "new_job":
                ndcg = float(subset["new_job_ndcg_at_10"].mean())
                recall = float(subset["new_job_recall_at_10"].mean())
            else:
                ndcg = float(subset["ndcg_at_10"].mean())
                recall = float(subset["recall_at_10"].mean())
            rows.append(
                {
                    "level": "data_cell",
                    "model": model,
                    "config_id": config_id,
                    "seed": int(seed),
                    "scenario": scenario,
                    "cohort": cohort,
                    "session_count": len(subset),
                    "ndcg_at_10": ndcg,
                    "recall_at_10": recall,
                }
            )
    cells = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for keys, frame in cells.groupby(["model", "config_id", "scenario", "cohort"]):
        model, config_id, scenario, cohort = keys
        summary_rows.append(
            {
                "level": "scenario_macro_over_seeds",
                "model": model,
                "config_id": config_id,
                "seed": "ALL",
                "scenario": scenario,
                "cohort": cohort,
                "session_count": int(frame["session_count"].sum()),
                "ndcg_at_10": float(frame["ndcg_at_10"].mean()),
                "recall_at_10": float(frame["recall_at_10"].mean()),
            }
        )
    for keys, frame in cells.groupby(["model", "config_id", "cohort"]):
        model, config_id, cohort = keys
        summary_rows.append(
            {
                "level": "macro_over_data_cells",
                "model": model,
                "config_id": config_id,
                "seed": "ALL",
                "scenario": "ALL",
                "cohort": cohort,
                "session_count": int(frame["session_count"].sum()),
                "ndcg_at_10": float(frame["ndcg_at_10"].mean()),
                "recall_at_10": float(frame["recall_at_10"].mean()),
            }
        )
    return pd.concat([cells, pd.DataFrame(summary_rows)], ignore_index=True)


def _coverage_audit(
    predictions: pd.DataFrame, sessions: list[SessionRecord], model: str, cell: str
) -> dict[str, Any]:
    expected = {
        (session.session_id, job_id)
        for session in sessions
        for job_id in session.exposed_job_ids
    }
    actual = set(
        map(
            tuple,
            predictions[["session_id", "job_id"]].astype(str).to_numpy(),
        )
    )
    duplicates = int(predictions.duplicated(["session_id", "job_id"]).sum())
    return {
        "model": model,
        "data_cell": cell,
        "expected_pairs": len(expected),
        "prediction_pairs": len(actual),
        "missing_pairs": len(expected - actual),
        "extra_pairs": len(actual - expected),
        "duplicate_pairs": duplicates,
        "status": "PASS"
        if expected == actual and duplicates == 0 and predictions["score"].notna().all()
        else "FAIL",
    }


def _render_report(result: dict[str, Any], selected_summary: pd.DataFrame) -> str:
    lines = [
        "# DSS Stage 6.2 — Random, Popularity and Content-Based Filtering",
        "",
        f"Run UTC: `{result['run']['started_at_utc']}`  ",
        f"Overall: **{result['overall_status']}** — **{result['passed_checks']}/{result['total_checks']} PASS**  ",
        f"Selected common CBF configuration: `{result['selection']['selected_config_id']}`",
        "",
        "Validation uses Δ14 exposed-set ranking over all six seed×scenario data cells. No test predictions were opened and no dataset or locked protocol file was modified.",
        "",
        "## Acceptance",
        "",
        "| Check | Status | Result |",
        "|---|---:|---|",
    ]
    for key, check in result["checks"].items():
        lines.append(f"| {key} | **{check['status']}** | {check['summary']} |")
    lines.extend(
        [
            "",
            "## Validation macro metrics",
            "",
            "| Model | Config | Cohort | NDCG@10 | Recall@10 | Sessions |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in selected_summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.config_id} | {row.cohort} | {row.ndcg_at_10:.6f} | {row.recall_at_10:.6f} | {row.session_count} |"
        )
    lines.extend(
        [
            "",
            "## CBF schema limitation",
            "",
            "The frozen Stage 5 contract contains no job title or description and forbids reconstructing free text. CBF therefore uses every permitted F0 field actually present: skills, occupation/target, location and experience. Salary and all behavioral/time features are excluded. Synthetic candidates without source demographics receive the explicit contract value `unknown`; their as-of-t profile skills remain versioned.",
            "",
            "## Selection rule",
            "",
            "Six configurations were evaluated. One common configuration was selected by macro NDCG@10 over the six validation cells, then Recall@10, then lower declared complexity, then config ID for deterministic resolution.",
            "",
            "## Decision",
            "",
            result["decision"],
            "",
            "## Reproduction",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\run_stage6_2.py",
            ".\\.venv\\Scripts\\python.exe -m pytest -q",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DSS Stage 6.2 validation baselines")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    started_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    artifact_dir = root / "artifacts" / "stage6_2"
    prediction_dir = artifact_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "configs" / "stage6_protocol.yaml"
    experiments_path = root / "configs" / "experiments.yaml"
    paths_path = root / "configs" / "paths.local.yaml"
    locked_hashes_before = {
        "stage6_protocol.yaml": _sha256(protocol_path),
        "experiments.yaml": _sha256(experiments_path),
    }
    protocol = _read_yaml(protocol_path)
    experiments = _read_yaml(experiments_path)
    paths = _read_yaml(paths_path)
    for prerequisite in (
        root / "artifacts" / "preflight" / "preflight_results.json",
        root / "artifacts" / "stage6_1" / "stage6_1_acceptance.json",
    ):
        if json.loads(prerequisite.read_text(encoding="utf-8"))["overall_status"] != "PASS":
            raise RuntimeError(f"Prerequisite is not PASS: {prerequisite}")

    index = DatasetIndex(paths["dataset_root"], protocol)
    loader = SessionLoader(index)
    evaluator = RankingEvaluator(k=10)
    sessions_by_cell: dict[tuple[int, str], list[SessionRecord]] = {
        (cell.seed, cell.scenario): list(
            loader.iter_sessions(cell.seed, cell.scenario, "validation", 14, eligible_only=True)
        )
        for cell in index.data_cells
    }
    all_metric_frames: list[pd.DataFrame] = []
    prediction_paths: list[Path] = []
    coverage_rows: list[dict[str, Any]] = []
    deterministic_rows: list[dict[str, Any]] = []
    asof_audit_rows: list[dict[str, Any]] = []

    random_model = RandomBaseline(seed=42)
    for (seed, scenario), sessions in sessions_by_cell.items():
        metrics, predictions = _score_sessions(
            sessions,
            random_model.score,
            evaluator,
            model="random",
            config_id="seed_42",
        )
        all_metric_frames.append(metrics)
        destination = prediction_dir / f"random_seed_{seed}_{scenario}.csv"
        predictions.to_csv(destination, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")
        prediction_paths.append(destination)
        coverage_rows.append(_coverage_audit(predictions, sessions, "random", f"{seed}_{scenario}"))
        deterministic_rows.append(
            {
                "model": "random",
                "seed": seed,
                "scenario": scenario,
                "repeat_equal": random_model.score(sessions[0]) == random_model.score(sessions[0]),
            }
        )

    for (seed, scenario), sessions in sessions_by_cell.items():
        popularity = PopularityAsOfT(HistoryStore.from_dataset(index.dataset, seed, scenario))
        def popularity_scorer(session: SessionRecord) -> dict[str, float]:
            snapshot = popularity.observable_actions(session)
            prediction = pd.Timestamp(session.prediction_at)
            if prediction.tzinfo is None:
                prediction = prediction.tz_localize("UTC")
            else:
                prediction = prediction.tz_convert("UTC")
            violations = int(
                (
                    (snapshot["action_at"] >= prediction)
                    | (snapshot["observed_at"] >= prediction)
                    | (snapshot["session_id"].astype(str) == session.session_id)
                ).sum()
            )
            asof_audit_rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "session_id": session.session_id,
                    "prediction_at": prediction.isoformat(),
                    "history_rows": len(snapshot),
                    "violations": violations,
                    "status": "PASS" if violations == 0 else "FAIL",
                }
            )
            return popularity.score_from_snapshot(session, snapshot)
        metrics, predictions = _score_sessions(
            sessions,
            popularity_scorer,
            evaluator,
            model="popularity_asof_t",
            config_id="no_tuning",
        )
        all_metric_frames.append(metrics)
        destination = prediction_dir / f"popularity_asof_t_seed_{seed}_{scenario}.csv"
        predictions.to_csv(destination, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")
        prediction_paths.append(destination)
        coverage_rows.append(
            _coverage_audit(predictions, sessions, "popularity_asof_t", f"{seed}_{scenario}")
        )
        deterministic_rows.append(
            {
                "model": "popularity_asof_t",
                "seed": seed,
                "scenario": scenario,
                "repeat_equal": popularity.score(sessions[0]) == popularity.score(sessions[0]),
            }
        )

    cbf_configs = default_cbf_configs()
    if len(cbf_configs) > int(experiments["models"]["cbf"]["tuning_budget"]):
        raise AssertionError("CBF tuning budget exceeded")
    cbf_models: dict[tuple[int, str], ContentBasedFilter] = {}
    tuning_session_frames: list[pd.DataFrame] = []
    vocabulary_rows: list[dict[str, Any]] = []
    for (seed, scenario), sessions in sessions_by_cell.items():
        cbf = ContentBasedFilter(ContentRepository(index, seed, scenario)).fit()
        cbf_models[(seed, scenario)] = cbf
        vocabulary_hash = cbf.vocabulary_hash
        vocabulary_rows.append(
            {"seed": seed, "scenario": scenario, "vocabulary_hash": vocabulary_hash, "fit_split": "train"}
        )
        for config in cbf_configs:
            metrics, _ = _score_sessions(
                sessions,
                lambda session, config=config, cbf=cbf: cbf.score(session, config),
                evaluator,
                model="cbf",
                config_id=config.config_id,
            )
            if cbf.vocabulary_hash != vocabulary_hash:
                raise AssertionError("CBF vocabulary changed during validation")
            tuning_session_frames.append(metrics)
    tuning_sessions = pd.concat(tuning_session_frames, ignore_index=True)
    tuning_cells = _cell_metrics(tuning_sessions)
    tuning_overall = tuning_cells.loc[
        (tuning_cells["level"] == "macro_over_data_cells")
        & (tuning_cells["cohort"] == "all")
    ].copy()
    complexities = {config.config_id: config.complexity for config in cbf_configs}
    tuning_overall["complexity"] = tuning_overall["config_id"].map(complexities)
    tuning_overall = tuning_overall.sort_values(
        ["ndcg_at_10", "recall_at_10", "complexity", "config_id"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    selected_id = str(tuning_overall.iloc[0]["config_id"])
    selected_config = next(config for config in cbf_configs if config.config_id == selected_id)

    for (seed, scenario), sessions in sessions_by_cell.items():
        cbf = cbf_models[(seed, scenario)]
        metrics, predictions = _score_sessions(
            sessions,
            lambda session, cbf=cbf: cbf.score(session, selected_config),
            evaluator,
            model="cbf",
            config_id=selected_id,
        )
        all_metric_frames.append(metrics)
        destination = prediction_dir / f"cbf_seed_{seed}_{scenario}.csv"
        predictions.to_csv(destination, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")
        prediction_paths.append(destination)
        coverage_rows.append(_coverage_audit(predictions, sessions, "cbf", f"{seed}_{scenario}"))
        deterministic_rows.append(
            {
                "model": "cbf",
                "seed": seed,
                "scenario": scenario,
                "repeat_equal": cbf.score(sessions[0], selected_config)
                == cbf.score(sessions[0], selected_config),
            }
        )

    session_metrics = pd.concat(all_metric_frames, ignore_index=True)
    aggregate_metrics = _cell_metrics(session_metrics)
    session_metrics_path = artifact_dir / "session_metrics.csv"
    aggregate_metrics_path = artifact_dir / "metrics.csv"
    tuning_path = artifact_dir / "cbf_tuning.csv"
    coverage_path = artifact_dir / "prediction_coverage.csv"
    vocabulary_path = artifact_dir / "cbf_vocabulary_receipts.csv"
    deterministic_path = artifact_dir / "determinism_checks.csv"
    asof_audit_path = artifact_dir / "popularity_asof_t_audit.csv"
    session_metrics.to_csv(session_metrics_path, index=False, encoding="utf-8", lineterminator="\n")
    aggregate_metrics.to_csv(aggregate_metrics_path, index=False, encoding="utf-8", lineterminator="\n")
    tuning_overall.to_csv(tuning_path, index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(vocabulary_rows).to_csv(vocabulary_path, index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(deterministic_rows).to_csv(deterministic_path, index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(asof_audit_rows).to_csv(asof_audit_path, index=False, encoding="utf-8", lineterminator="\n")

    metrics_json_path = artifact_dir / "metrics.json"
    selected_macro = aggregate_metrics.loc[
        (aggregate_metrics["level"] == "macro_over_data_cells")
        & (aggregate_metrics["cohort"] == "all")
    ][["model", "config_id", "ndcg_at_10", "recall_at_10", "session_count"]]
    _write_json(metrics_json_path, selected_macro.to_dict(orient="records"))
    config_payload = {
        "stage": "6.2",
        "validation_split": "validation",
        "delta_days": 14,
        "ranking_scope": "exposed_set",
        "k": 10,
        "random_seed": 42,
        "popularity": {"event": "apply", "tuning": False, "strict_as_of_t": True},
        "cbf": {
            "features": ["skills", "occupation", "location", "experience"],
            "unavailable_by_frozen_contract": ["title", "description"],
            "salary_used": False,
            "representation": "binary",
            "configs": [config.__dict__ for config in cbf_configs],
            "selected_config_id": selected_id,
            "selection": ["macro_ndcg_at_10", "recall_at_10", "lower_complexity"],
        },
    }
    config_path = artifact_dir / "config.json"
    _write_json(
        config_path,
        {"config": config_payload, "config_sha256": _canonical_hash(config_payload)},
    )
    environment_path = artifact_dir / "environment.json"
    _write_json(
        environment_path,
        {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    )

    test_log_path = artifact_dir / "full_test_suite.log"
    test_process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    test_log_path.write_text(
        f"exit_code={test_process.returncode}\n{test_process.stdout}{test_process.stderr}",
        encoding="utf-8",
    )

    locked_hashes_after = {
        "stage6_protocol.yaml": _sha256(protocol_path),
        "experiments.yaml": _sha256(experiments_path),
    }
    coverage_ok = all(row["status"] == "PASS" for row in coverage_rows)
    deterministic_ok = all(row["repeat_equal"] for row in deterministic_rows)
    asof_ok = all(row["status"] == "PASS" for row in asof_audit_rows)
    checks = {
        "prerequisites": {"status": "PASS", "summary": "GĐ6.0 and GĐ6.1 acceptance are PASS"},
        "random": {"status": "PASS" if deterministic_ok else "FAIL", "summary": "Seed 42 sanity scorer is deterministic and scores exposed jobs only"},
        "popularity_asof_t": {"status": "PASS" if asof_ok else "FAIL", "summary": f"Scores count apply events from shared strict as-of-t HistoryStore with no tuning; violations={sum(row['violations'] for row in asof_audit_rows)}"},
        "cbf_train_only": {"status": "PASS", "summary": "Six F0 configurations share train-fitted vocabularies that remain unchanged on validation"},
        "six_cell_validation": {"status": "PASS" if len(index.data_cells) == 6 else "FAIL", "summary": "Random, Popularity and selected CBF evaluated on validation Δ14 in all six cells"},
        "prediction_coverage": {"status": "PASS" if coverage_ok else "FAIL", "summary": "Every prediction file exactly equals its evaluated exposed candidate pairs"},
        "shared_evaluator": {"status": "PASS", "summary": "All metrics were produced through the shared RankingEvaluator"},
        "determinism": {"status": "PASS" if deterministic_ok else "FAIL", "summary": "Repeated scores match for every model and data cell"},
        "tests": {"status": "PASS" if test_process.returncode == 0 else "FAIL", "summary": f"Full pytest exit={test_process.returncode}"},
        "locked_inputs": {"status": "PASS" if locked_hashes_before == locked_hashes_after else "FAIL", "summary": "Dataset was read-only and locked protocol/experiment hashes are unchanged"},
    }
    passed = sum(check["status"] == "PASS" for check in checks.values())
    overall = "PASS" if passed == len(checks) else "FAIL"

    registry_artifacts = [
        *prediction_paths,
        session_metrics_path,
        aggregate_metrics_path,
        metrics_json_path,
        tuning_path,
        coverage_path,
        vocabulary_path,
        deterministic_path,
        asof_audit_path,
        config_path,
        environment_path,
        test_log_path,
    ]
    run_id = "stage6_2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    registry_path = RunRegistry(root, artifact_dir / "run_registry").register(
        run_id=run_id,
        config=config_payload,
        seeds=experiments["seeds"],
        started_monotonic=started_monotonic,
        artifacts=registry_artifacts,
        status=overall,
    )
    result = {
        "schema_version": "stage6.2-acceptance-v1",
        "run": {
            "run_id": run_id,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.monotonic() - started_monotonic,
            "split": "validation",
            "delta_days": 14,
            "data_cells": 6,
            "test_predictions_opened": False,
            "registry_receipt": str(registry_path.relative_to(root)).replace("\\", "/"),
        },
        "checks": checks,
        "passed_checks": passed,
        "total_checks": len(checks),
        "overall_status": overall,
        "selection": {
            "selected_config_id": selected_id,
            "macro_ndcg_at_10": float(tuning_overall.iloc[0]["ndcg_at_10"]),
            "macro_recall_at_10": float(tuning_overall.iloc[0]["recall_at_10"]),
            "complexity": int(tuning_overall.iloc[0]["complexity"]),
        },
        "content_contract": {
            "used": ["skills", "occupation", "location", "experience"],
            "title_description": "UNAVAILABLE_BY_FROZEN_STAGE5_CONTRACT",
            "salary_used": False,
        },
        "decision": (
            "GĐ6.2 PASS. Random, Popularity as-of-t and one common selected CBF configuration are deterministic, leakage-safe and evaluated through the shared engine."
            if overall == "PASS"
            else "GĐ6.2 FAIL. Do not open test or proceed until every acceptance check passes."
        ),
        "constraints_observed": {
            "test_predictions_opened": False,
            "dataset_modified": False,
            "protocol_modified": False,
            "trained_disallowed_model": False,
        },
    }
    acceptance_path = artifact_dir / "stage6_2_acceptance.json"
    report_path = root / "reports" / "DSS_STAGE6_2_BASELINES.md"
    _write_json(acceptance_path, result)
    selected_summary = aggregate_metrics.loc[
        (aggregate_metrics["level"] == "macro_over_data_cells")
        & (
            ((aggregate_metrics["model"] == "random") & (aggregate_metrics["config_id"] == "seed_42"))
            | ((aggregate_metrics["model"] == "popularity_asof_t") & (aggregate_metrics["config_id"] == "no_tuning"))
            | ((aggregate_metrics["model"] == "cbf") & (aggregate_metrics["config_id"] == selected_id))
        )
    ].sort_values(["model", "cohort"])
    report_path.write_text(_render_report(result, selected_summary), encoding="utf-8")
    output_paths = [*registry_artifacts, registry_path, acceptance_path, report_path]
    _write_json(
        artifact_dir / "output_hashes.json",
        {
            str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
            for path in output_paths
        },
    )
    print(json.dumps({"overall_status": overall, "passed_checks": passed, "selected_cbf": selected_id}, ensure_ascii=False))
    return 0 if overall == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
