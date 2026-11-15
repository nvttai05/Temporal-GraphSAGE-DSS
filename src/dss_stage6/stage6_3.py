from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from dss_stage6.data import DatasetIndex, SessionRecord
from dss_stage6.evaluation import RankingEvaluator
from dss_stage6.models import Stage6XGBoost, scale_pos_weight
from dss_stage6.registry import RunRegistry
from dss_stage6.xgboost_features import BuiltFeatures, XGBoostFeaturePipeline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _evaluate(
    sessions: list[SessionRecord],
    built: BuiltFeatures,
    scores: np.ndarray,
    evaluator: RankingEvaluator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = built.identifiers.copy()
    predictions["label"] = built.labels
    predictions["score"] = scores
    metric_rows: list[dict[str, Any]] = []
    expected_pairs: set[tuple[str, str]] = set()
    for session in sessions:
        session_scores = predictions.loc[
            predictions["session_id"] == session.session_id
        ].set_index("job_id")["score"].to_dict()
        metric_rows.append(evaluator.evaluate_session(session, session_scores))
        expected_pairs.update((session.session_id, job_id) for job_id in session.exposed_job_ids)
    actual_pairs = set(
        map(tuple, predictions[["session_id", "job_id"]].astype(str).to_numpy())
    )
    if expected_pairs != actual_pairs or predictions.duplicated(["session_id", "job_id"]).any():
        raise AssertionError("Smoke predictions do not equal exposed candidate pairs")
    return pd.DataFrame(metric_rows), predictions


def _git_state(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return {"commit": commit, "status_lines": status}


def prepare_local_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    prediction_dir = output_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, prediction_dir


def _render_local_report(result: dict[str, Any], metrics: pd.DataFrame) -> str:
    lines = [
        "# DSS Stage 6.3A — XGBoost Static/Time-Aware Local Smoke",
        "",
        f"Run UTC: `{result['run']['started_at_utc']}`  ",
        f"Status: **{result['overall_status']}**  ",
        "Full GĐ6.3 complete: **NO**",
        "",
        "This laptop run is implementation validation only. It uses data-cell `53021/S0`, at most two pre-locked trials per model, and a bounded validation-session subset. It does not select the final hyperparameter configuration.",
        "",
        "## Local smoke metrics (informational only)",
        "",
        "| Model | Trial | Sessions | NDCG@10 | Recall@10 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.model_kind} | {row.trial_id} | {row.session_count} | {row.ndcg_at_10:.6f} | {row.recall_at_10:.6f} |"
        )
    lines.extend(["", "## Acceptance", "", "| Check | Status | Result |", "|---|---:|---|"])
    for name, check in result["checks"].items():
        lines.append(f"| {name} | **{check['status']}** | {check['summary']} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "No test predictions were opened. No final configuration was selected. Full 20×2 trial tuning and six-cell validation must run on Colab before GĐ6.3 can be completed.",
            "",
            "## Colab hand-off",
            "",
            "Use `notebooks/STAGE6_3_COLAB_RUNNER.ipynb` and `docs/STAGE6_3_COLAB_GUIDE.md`. The runner verifies Git commit and dataset manifest, reruns tests, supports per-trial resume, and writes outputs to Google Drive.",
            "",
        ]
    )
    return "\n".join(lines)


def run_local_smoke(root: Path, dataset_root: Path, output_dir: Path) -> int:
    started_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    config_path = root / "configs" / "stage6_3_xgboost.yaml"
    protocol_path = root / "configs" / "stage6_protocol.yaml"
    experiments_path = root / "configs" / "experiments.yaml"
    locked_before = {
        "stage6_3_xgboost.yaml": _sha256(config_path),
        "stage6_protocol.yaml": _sha256(protocol_path),
        "experiments.yaml": _sha256(experiments_path),
    }
    config = _read_yaml(config_path)
    protocol = _read_yaml(protocol_path)
    trials = config["trials"]
    if len(trials) != 20 or config["scope"]["local_smoke_max_trials_per_model"] != 2:
        raise AssertionError("Locked trial contract is invalid")
    for prerequisite in (
        root / "artifacts" / "preflight" / "preflight_results.json",
        root / "artifacts" / "stage6_1" / "stage6_1_acceptance.json",
        root / "artifacts" / "stage6_2" / "stage6_2_acceptance.json",
    ):
        if json.loads(prerequisite.read_text(encoding="utf-8"))["overall_status"] != "PASS":
            raise RuntimeError(f"Prerequisite not PASS: {prerequisite}")
    stage62 = json.loads(
        (root / "artifacts" / "stage6_2" / "stage6_2_acceptance.json").read_text(
            encoding="utf-8"
        )
    )
    if stage62["selection"]["selected_config_id"] != "cbf_05_skills3_occupation2":
        raise RuntimeError("Locked CBF config differs from GĐ6.2")

    index = DatasetIndex(dataset_root, protocol)
    pipeline = XGBoostFeaturePipeline(
        index,
        53021,
        "S0",
        recency_sentinel_days=config["feature_contract"]["f2"][
            "recency_missing_sentinel_days"
        ],
        decay_half_life_days=config["feature_contract"]["f2"][
            "exponential_decay_half_life_days"
        ],
    )
    smoke_config = config["local_smoke"]
    train_sessions = pipeline.sessions(
        "train", eligible_only=False, max_sessions=smoke_config["max_train_sessions"]
    )
    validation_sessions = pipeline.sessions(
        "validation",
        eligible_only=True,
        max_sessions=smoke_config["max_validation_sessions"],
    )
    train_raw = pipeline.raw_frame(train_sessions)
    validation_raw = pipeline.raw_frame(validation_sessions)
    pipeline.fit(train_raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_schema_path = output_dir / "feature_schema.json"
    schema_receipt = pipeline.schema_receipt()
    _write_json(feature_schema_path, schema_receipt)
    leakage_path = output_dir / "feature_leakage_audit.csv"
    pd.DataFrame(pipeline.leakage_audit).to_csv(
        leakage_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    fit_receipts = {
        "fit_split": "train",
        "train_sessions": len(train_sessions),
        "train_rows": len(train_raw),
        "validation_sessions": len(validation_sessions),
        "validation_rows": len(validation_raw),
        "static_state_before_validation": pipeline.static_builder.state_hash,
        "time_state_before_validation": pipeline.time_builder.state_hash,
        "config_sha256": locked_before["stage6_3_xgboost.yaml"],
    }

    evaluator = RankingEvaluator(10)
    metric_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    roundtrip_rows: list[dict[str, Any]] = []
    determinism_rows: list[dict[str, Any]] = []
    artifact_paths: list[Path] = [feature_schema_path, leakage_path]
    checkpoint_dir, prediction_dir = prepare_local_output_dirs(output_dir)
    for model_kind in ("static", "time"):
        train_built = pipeline.transform(train_raw, model_kind)
        state_before = (
            pipeline.static_builder.state_hash
            if model_kind == "static"
            else pipeline.time_builder.state_hash
        )
        validation_built = pipeline.transform(validation_raw, model_kind)
        state_after = (
            pipeline.static_builder.state_hash
            if model_kind == "static"
            else pipeline.time_builder.state_hash
        )
        if state_before != state_after:
            raise AssertionError("Validation transform refit feature state")
        class_weight = scale_pos_weight(train_built.labels)
        for trial in trials[:2]:
            model = Stage6XGBoost(trial, model_seed=42, class_weight=class_weight).fit(
                train_built.transformed.to_numpy(), train_built.labels
            )
            scores = model.predict_scores(validation_built.transformed.to_numpy())
            session_metrics, predictions = _evaluate(
                validation_sessions, validation_built, scores, evaluator
            )
            checkpoint = checkpoint_dir / f"xgboost_{model_kind}_{trial['trial_id']}.json"
            model.save(checkpoint)
            loaded_scores = Stage6XGBoost.load(checkpoint).predict_scores(
                validation_built.transformed.to_numpy()
            )
            roundtrip_equal = bool(np.array_equal(scores, loaded_scores))
            if not roundtrip_equal:
                raise AssertionError("Checkpoint round trip changed scores")
            prediction_path = (
                prediction_dir / f"xgboost_{model_kind}_{trial['trial_id']}_53021_S0.csv"
            )
            predictions.insert(0, "model_kind", model_kind)
            predictions.insert(1, "trial_id", trial["trial_id"])
            predictions.to_csv(
                prediction_path,
                index=False,
                encoding="utf-8",
                lineterminator="\n",
                float_format="%.17g",
            )
            repeat = Stage6XGBoost(
                trial, model_seed=42, class_weight=class_weight
            ).fit(train_built.transformed.to_numpy(), train_built.labels)
            repeat_scores = repeat.predict_scores(validation_built.transformed.to_numpy())
            deterministic = bool(np.array_equal(scores, repeat_scores))
            metric_rows.append(
                {
                    "model_kind": model_kind,
                    "trial_id": trial["trial_id"],
                    "session_count": len(session_metrics),
                    "candidate_rows": len(predictions),
                    "scale_pos_weight": class_weight,
                    "ndcg_at_10": float(session_metrics["ndcg_at_10"].mean()),
                    "recall_at_10": float(session_metrics["recall_at_10"].mean()),
                    "informational_only": True,
                }
            )
            coverage_rows.append(
                {
                    "model_kind": model_kind,
                    "trial_id": trial["trial_id"],
                    "expected_pairs": len(validation_raw),
                    "prediction_pairs": len(predictions),
                    "duplicates": int(
                        predictions.duplicated(["session_id", "job_id"]).sum()
                    ),
                    "status": "PASS"
                    if len(validation_raw) == len(predictions)
                    and not predictions.duplicated(["session_id", "job_id"]).any()
                    else "FAIL",
                }
            )
            roundtrip_rows.append(
                {
                    "model_kind": model_kind,
                    "trial_id": trial["trial_id"],
                    "checkpoint": str(checkpoint.relative_to(root)).replace("\\", "/"),
                    "score_exact_match": roundtrip_equal,
                    "status": "PASS" if roundtrip_equal else "FAIL",
                }
            )
            determinism_rows.append(
                {
                    "model_kind": model_kind,
                    "trial_id": trial["trial_id"],
                    "seed": 42,
                    "score_exact_match": deterministic,
                    "status": "PASS" if deterministic else "FAIL",
                }
            )
            artifact_paths.extend([checkpoint, prediction_path])

    fit_receipts["static_state_after_validation"] = pipeline.static_builder.state_hash
    fit_receipts["time_state_after_validation"] = pipeline.time_builder.state_hash
    fit_receipts["states_unchanged"] = (
        fit_receipts["static_state_before_validation"]
        == fit_receipts["static_state_after_validation"]
        and fit_receipts["time_state_before_validation"]
        == fit_receipts["time_state_after_validation"]
    )
    fit_receipt_path = output_dir / "train_only_fit_receipts.json"
    _write_json(fit_receipt_path, fit_receipts)
    artifact_paths.append(fit_receipt_path)
    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "smoke_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8", lineterminator="\n")
    _write_json(output_dir / "smoke_metrics.json", metrics.to_dict(orient="records"))
    coverage_path = output_dir / "prediction_coverage.csv"
    roundtrip_path = output_dir / "checkpoint_roundtrip.csv"
    determinism_path = output_dir / "determinism.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(roundtrip_rows).to_csv(roundtrip_path, index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(determinism_rows).to_csv(determinism_path, index=False, encoding="utf-8", lineterminator="\n")
    artifact_paths.extend(
        [metrics_path, output_dir / "smoke_metrics.json", coverage_path, roundtrip_path, determinism_path]
    )

    test_log_path = output_dir / "full_test_suite.log"
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
    artifact_paths.append(test_log_path)
    locked_after = {
        "stage6_3_xgboost.yaml": _sha256(config_path),
        "stage6_protocol.yaml": _sha256(protocol_path),
        "experiments.yaml": _sha256(experiments_path),
    }
    leakage_ok = not pd.DataFrame(pipeline.leakage_audit)[
        "future_or_current_violations"
    ].any()
    schema_ok = (
        not schema_receipt["static"]["contains_f2"]
        and schema_receipt["time"]["contains_f2"]
        and not set(schema_receipt["static"]["raw_columns"])
        & set(schema_receipt["time"]["raw_columns"])
        - set(schema_receipt["static"]["raw_columns"])
    )
    # Explicit subset assertion is easier to audit than relying on names.
    schema_ok = schema_ok and set(schema_receipt["static"]["raw_columns"]).issubset(
        schema_receipt["time"]["raw_columns"]
    )
    colab_assets = [
        root / "notebooks" / "STAGE6_3_COLAB_RUNNER.ipynb",
        root / "requirements-colab.txt",
        root / "docs" / "STAGE6_3_COLAB_GUIDE.md",
    ]
    checks = {
        "prerequisites": {
            "status": "PASS",
            "summary": "GĐ6.0/GĐ6.1/GĐ6.2 PASS and CBF config locked",
        },
        "locked_trials": {
            "status": "PASS" if len(trials) == 20 and locked_before == locked_after else "FAIL",
            "summary": "20 deterministic trials were persisted before training and locked hashes stayed unchanged",
        },
        "feature_schema": {
            "status": "PASS" if schema_ok else "FAIL",
            "summary": "Static contains F0+F1 only; time contains F0+F1+F2",
        },
        "leakage": {
            "status": "PASS" if leakage_ok else "FAIL",
            "summary": f"Feature future/current-session violations={sum(row['future_or_current_violations'] for row in pipeline.leakage_audit)}",
        },
        "train_only_fit": {
            "status": "PASS" if fit_receipts["states_unchanged"] else "FAIL",
            "summary": "Encoder/scaler/CBF vocabulary fit on train and unchanged by validation",
        },
        "coverage": {
            "status": "PASS" if all(row["status"] == "PASS" for row in coverage_rows) else "FAIL",
            "summary": "Every smoke prediction covers exactly the exposed validation pairs",
        },
        "shared_evaluator": {
            "status": "PASS",
            "summary": "NDCG@10 and Recall@10 produced by shared RankingEvaluator",
        },
        "determinism": {
            "status": "PASS" if all(row["status"] == "PASS" for row in determinism_rows) else "FAIL",
            "summary": "Repeated seed-42 training produced byte-identical score arrays",
        },
        "checkpoint_roundtrip": {
            "status": "PASS" if all(row["status"] == "PASS" for row in roundtrip_rows) else "FAIL",
            "summary": "Every smoke checkpoint reload produced identical scores",
        },
        "tests": {
            "status": "PASS" if test_process.returncode == 0 else "FAIL",
            "summary": f"Full pytest exit={test_process.returncode}",
        },
        "colab_package": {
            "status": "PASS" if all(path.is_file() for path in colab_assets) else "FAIL",
            "summary": "Colab runner, pinned requirements and guide are present",
        },
    }
    passed = sum(check["status"] == "PASS" for check in checks.values())
    ready = passed == len(checks)
    git_state = _git_state(root)
    run_id = "stage6_3a_local_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    registry_path = RunRegistry(root, output_dir / "run_registry").register(
        run_id=run_id,
        config=config,
        seeds={"tuning_model_seed": 42, "data_seed": 53021},
        started_monotonic=started_monotonic,
        artifacts=artifact_paths,
        status="READY_FOR_COLAB" if ready else "FAIL",
    )
    result = {
        "schema_version": "stage6.3a-local-smoke-v1",
        "run": {
            "run_id": run_id,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.monotonic() - started_monotonic,
            "git_commit_before_work": "a8a90fd4f0bfd63c2c22b288d153f6d8d95a2f1c",
            "git_state_at_receipt": git_state,
            "cell": {"seed": 53021, "scenario": "S0"},
            "trials_per_model": 2,
            "test_predictions_opened": False,
            "registry_receipt": str(registry_path.relative_to(root)).replace("\\", "/"),
        },
        "checks": checks,
        "passed_checks": passed,
        "total_checks": len(checks),
        "overall_status": "READY_FOR_COLAB" if ready else "FAIL",
        "full_stage6_3_complete": False,
        "configuration_selected_from_smoke": False,
        "decision": (
            "GĐ6.3A local implementation READY_FOR_COLAB. Full GĐ6.3 remains incomplete until 20-trial tuning for both models across all six validation cells finishes on Colab."
            if ready
            else "GĐ6.3A local implementation is not ready for Colab."
        ),
        "constraints_observed": {
            "dataset_modified": False,
            "protocol_modified": False,
            "stage6_2_modified": False,
            "test_predictions_opened": False,
            "final_config_selected": False,
        },
    }
    acceptance_path = output_dir / "stage6_3_local_smoke.json"
    report_path = root / "reports" / "DSS_STAGE6_3A_LOCAL_SMOKE.md"
    _write_json(acceptance_path, result)
    report_path.write_text(_render_local_report(result, metrics), encoding="utf-8")
    output_paths = [*artifact_paths, registry_path, acceptance_path, report_path]
    _write_json(
        output_dir / "output_hashes.json",
        {
            str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
            for path in output_paths
        },
    )
    print(
        json.dumps(
            {
                "overall_status": result["overall_status"],
                "passed_checks": passed,
                "full_stage6_3_complete": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if ready else 2


def run_colab_full(
    root: Path, dataset_root: Path, output_dir: Path, *, resume: bool
) -> int:
    """Full six-cell/20-trial execution; intended only for Colab."""

    started_monotonic = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = root / "configs" / "stage6_3_xgboost.yaml"
    config = _read_yaml(config_path)
    protocol = _read_yaml(root / "configs" / "stage6_protocol.yaml")
    manifest_path = dataset_root / "data" / "candidate55" / "v0_3" / "dataset_manifest.json"
    expected_manifest_sha = "a6d983e754f49a9e2076022d48109516a931def7c17b1a725394245872aad287"
    actual_manifest_sha = _sha256(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise RuntimeError("Frozen dataset manifest SHA-256 mismatch")
    git_state = _git_state(root)
    initial_receipt_path = output_dir / "colab_run_environment_receipt.json"
    _write_json(
        initial_receipt_path,
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_state,
            "dataset_manifest_path": str(manifest_path),
            "dataset_manifest_sha256": actual_manifest_sha,
            "config_sha256": _sha256(config_path),
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "xgboost", "PyYAML")
            },
            "resume": bool(resume),
            "test_predictions_opened": False,
        },
    )
    index = DatasetIndex(dataset_root, protocol)
    evaluator = RankingEvaluator(10)
    result_rows: list[dict[str, Any]] = []
    for cell in index.data_cells:
        pipeline = XGBoostFeaturePipeline(index, cell.seed, cell.scenario)
        train_sessions = pipeline.sessions("train", eligible_only=False)
        validation_sessions = pipeline.sessions("validation", eligible_only=True)
        train_raw = pipeline.raw_frame(train_sessions)
        validation_raw = pipeline.raw_frame(validation_sessions)
        pipeline.fit(train_raw)
        for model_kind in ("static", "time"):
            train_built = pipeline.transform(train_raw, model_kind)
            validation_built = pipeline.transform(validation_raw, model_kind)
            weight = scale_pos_weight(train_built.labels)
            for trial in config["trials"]:
                trial_dir = (
                    output_dir
                    / model_kind
                    / trial["trial_id"]
                    / f"seed_{cell.seed}_{cell.scenario}"
                )
                receipt_path = trial_dir / "receipt.json"
                if resume and receipt_path.is_file():
                    result_rows.append(json.loads(receipt_path.read_text(encoding="utf-8")))
                    continue
                trial_dir.mkdir(parents=True, exist_ok=True)
                model = Stage6XGBoost(trial, model_seed=42, class_weight=weight).fit(
                    train_built.transformed.to_numpy(), train_built.labels
                )
                scores = model.predict_scores(validation_built.transformed.to_numpy())
                session_metrics, predictions = _evaluate(
                    validation_sessions, validation_built, scores, evaluator
                )
                checkpoint = trial_dir / "model.json"
                prediction_path = trial_dir / "predictions.csv"
                model.save(checkpoint)
                predictions.to_csv(prediction_path, index=False, encoding="utf-8", lineterminator="\n")
                receipt = {
                    "model_kind": model_kind,
                    "trial_id": trial["trial_id"],
                    "seed": cell.seed,
                    "scenario": cell.scenario,
                    "session_count": len(session_metrics),
                    "ndcg_at_10": float(session_metrics["ndcg_at_10"].mean()),
                    "recall_at_10": float(session_metrics["recall_at_10"].mean()),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "predictions_sha256": _sha256(prediction_path),
                }
                _write_json(receipt_path, receipt)
                result_rows.append(receipt)
    results = pd.DataFrame(result_rows)
    results.to_csv(output_dir / "all_trial_cell_metrics.csv", index=False)
    summary = (
        results.groupby(["model_kind", "trial_id"], as_index=False)[
            ["ndcg_at_10", "recall_at_10"]
        ]
        .mean()
        .sort_values(
            ["model_kind", "ndcg_at_10", "recall_at_10", "trial_id"],
            ascending=[True, False, False, True],
        )
    )
    summary.to_csv(output_dir / "tuning_summary.csv", index=False)
    selected = (
        summary.sort_values(
            ["model_kind", "ndcg_at_10", "recall_at_10", "trial_id"],
            ascending=[True, False, False, True],
        )
        .groupby("model_kind", as_index=False)
        .first()
    )
    selected.to_csv(output_dir / "selected_configs.csv", index=False)
    expected_receipts = 2 * 20 * 6
    if len(results) != expected_receipts:
        raise RuntimeError(
            f"Full run incomplete: {len(results)}/{expected_receipts} trial-cell receipts"
        )
    completion_path = output_dir / "colab_completion_receipt.json"
    _write_json(
        completion_path,
        {
            "status": "COMPLETE",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "trial_cell_receipts": len(results),
            "expected_trial_cell_receipts": expected_receipts,
            "selected_configs": selected.to_dict(orient="records"),
            "test_predictions_opened": False,
        },
    )
    registry_path = RunRegistry(root, output_dir / "run_registry").register(
        run_id="stage6_3_colab_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        config=config,
        seeds={"tuning_model_seed": 42, "data_seeds": [53021, 53022]},
        started_monotonic=started_monotonic,
        artifacts=[
            initial_receipt_path,
            output_dir / "all_trial_cell_metrics.csv",
            output_dir / "tuning_summary.csv",
            output_dir / "selected_configs.csv",
            completion_path,
        ],
        status="COMPLETE",
    )
    hash_targets = [
        initial_receipt_path,
        output_dir / "all_trial_cell_metrics.csv",
        output_dir / "tuning_summary.csv",
        output_dir / "selected_configs.csv",
        completion_path,
        registry_path,
        *sorted(output_dir.glob("static/trial_*/seed_*/receipt.json")),
        *sorted(output_dir.glob("time/trial_*/seed_*/receipt.json")),
    ]
    _write_json(
        output_dir / "colab_output_hashes.json",
        {
            str(path.relative_to(output_dir)).replace("\\", "/"): _sha256(path)
            for path in hash_targets
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DSS Stage 6.3 XGBoost runner")
    parser.add_argument("--mode", choices=["local-smoke", "colab-full"], default="local-smoke")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    paths = _read_yaml(root / "configs" / "paths.local.yaml")
    dataset_root = (args.dataset_root or Path(paths["dataset_root"])).resolve()
    if args.mode == "local-smoke":
        output_dir = (args.output_dir or root / "artifacts" / "stage6_3_local").resolve()
        return run_local_smoke(root, dataset_root, output_dir)
    if args.output_dir is None:
        raise ValueError("--output-dir is required for colab-full")
    return run_colab_full(root, dataset_root, args.output_dir.resolve(), resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
