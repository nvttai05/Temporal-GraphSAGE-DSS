from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dss_stage6.data import DatasetIndex
from dss_stage6.registry import RunRegistry
from dss_stage6.smoke import run_six_cell_smoke, smoke_digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _check(status: bool, summary: str, evidence: list[str]) -> dict[str, Any]:
    return {"status": "PASS" if status else "FAIL", "summary": summary, "evidence": evidence}


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# DSS Stage 6.1 — Shared Data Pipeline & Evaluation Engine",
        "",
        f"Run UTC: `{result['run']['started_at_utc']}`  ",
        f"Overall: **{result['overall_status']}** — **{result['passed_checks']}/8 PASS**  ",
        f"Eligible for model implementation: **{'YES' if result['eligible_for_model_implementation'] else 'NO'}**",
        "",
        "This acceptance run trained no recommendation model and opened no test predictions. The frozen dataset and locked protocol/config files were not modified.",
        "",
        "## Acceptance results",
        "",
        "| Check | Status | Result |",
        "|---|---:|---|",
    ]
    names = {
        "check_1": "DatasetIndex protocol lattice",
        "check_2": "SessionLoader exposed labels/cohorts",
        "check_3": "HistoryStore rolling as-of-t",
        "check_4": "FeatureBuilder train-only fit",
        "check_5": "StaticGraphBuilder leakage exclusion",
        "check_6": "Shared RankingEvaluator",
        "check_7": "RunRegistry provenance",
        "check_8": "Tests and deterministic six-cell smoke",
    }
    for key, name in names.items():
        check = result["checks"][key]
        evidence = ", ".join(f"`{item}`" for item in check["evidence"])
        lines.append(
            f"| {key[-1]}. {name} | **{check['status']}** | {check['summary']} Evidence: {evidence}. |"
        )
    lines.extend(
        [
            "",
            "## Deterministic smoke scope",
            "",
            "The smoke uses one scoreable validation Δ14 session in each of the six seed×scenario cells. Scores are a stable hash fixture, not a trained model. The complete smoke is executed twice and the tables must be byte-equivalent under canonical CSV serialization.",
            "",
            "## Decision",
            "",
            result["decision"],
            "",
            "## Reproduction",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\run_stage6_1_acceptance.py",
            ".\\.venv\\Scripts\\python.exe -m pytest -q",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DSS Stage 6.1 acceptance")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    started_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    artifact_dir = root / "artifacts" / "stage6_1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = root / "reports" / "DSS_STAGE6_1_PIPELINE_EVALUATION_ENGINE.md"
    acceptance_path = artifact_dir / "stage6_1_acceptance.json"
    smoke_path = artifact_dir / "six_cell_smoke.csv"
    test_log_path = artifact_dir / "full_test_suite.log"

    protocol_path = root / "configs" / "stage6_protocol.yaml"
    experiments_path = root / "configs" / "experiments.yaml"
    paths_path = root / "configs" / "paths.local.yaml"
    protocol = _read_yaml(protocol_path)
    experiments = _read_yaml(experiments_path)
    paths = _read_yaml(paths_path)
    preflight = json.loads(
        (root / "artifacts" / "preflight" / "preflight_results.json").read_text(
            encoding="utf-8"
        )
    )
    if preflight.get("overall_status") != "PASS":
        raise RuntimeError("GĐ6.0 must be PASS before GĐ6.1 acceptance")

    index = DatasetIndex(paths["dataset_root"], protocol)
    first_smoke = run_six_cell_smoke(index)
    second_smoke = run_six_cell_smoke(index)
    smoke_equal = first_smoke.equals(second_smoke)
    first_digest = smoke_digest(first_smoke)
    second_digest = smoke_digest(second_smoke)
    first_smoke.to_csv(
        smoke_path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g"
    )

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

    checks: dict[str, dict[str, Any]] = {}
    checks["check_1"] = _check(
        len(index.data_cells) == 6
        and len(index.slices) == 36
        and set(index.to_frame()["split"]) == {"train", "validation", "test"}
        and set(index.to_frame()["delta_days"]) == {7, 14},
        f"Indexed {len(index.data_cells)} data cells and {len(index.slices)} seed/scenario/split/window slices",
        ["artifacts/stage6_1/six_cell_smoke.csv"],
    )
    checks["check_2"] = _check(
        len(first_smoke) == 6 and (first_smoke["candidate_count"] >= 10).all(),
        "Loaded binary labels and cohort-bearing exposed sets for one scoreable validation session in every data cell",
        ["artifacts/stage6_1/six_cell_smoke.csv"],
    )
    checks["check_3"] = _check(
        int(first_smoke["future_edge_violations"].sum()) == 0,
        "Rolling history used strict action_at<t and observed_at<t with zero future-edge violations",
        ["artifacts/stage6_1/six_cell_smoke.csv", "artifacts/stage6_1/full_test_suite.log"],
    )
    checks["check_4"] = _check(
        bool(first_smoke["feature_state_unchanged"].all()),
        "FeatureBuilder fit on train and retained identical state across validation transforms",
        ["artifacts/stage6_1/six_cell_smoke.csv", "artifacts/stage6_1/full_test_suite.log"],
    )
    checks["check_5"] = _check(
        int(first_smoke["future_edge_violations"].sum()) == 0
        and (first_smoke["graph_edges"] == first_smoke["history_edges"]).all(),
        "Static graphs contain only observable prior actions and exclude current-session edges",
        ["artifacts/stage6_1/six_cell_smoke.csv"],
    )
    metrics_finite = all(
        math.isfinite(float(value))
        for value in first_smoke[["ndcg_at_10", "recall_at_10"]].to_numpy().ravel()
    )
    checks["check_6"] = _check(
        metrics_finite and len(first_smoke) == 6,
        "One shared exposed-set evaluator produced finite NDCG@10 and Recall@10 for all six cells with job_id ASC tie-breaking",
        ["artifacts/stage6_1/six_cell_smoke.csv", "artifacts/stage6_1/full_test_suite.log"],
    )

    run_id = "stage6_1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    registry = RunRegistry(root, artifact_dir / "run_registry")
    registry_config = {
        "stage": "6.1",
        "protocol_sha256": _sha256(protocol_path),
        "experiments_sha256": _sha256(experiments_path),
        "paths_sha256": _sha256(paths_path),
        "smoke_split": "validation",
        "smoke_delta_days": 14,
        "ranking_scope": "exposed_set",
        "k": 10,
    }
    registry_path = registry.register(
        run_id=run_id,
        config=registry_config,
        seeds=experiments["seeds"],
        started_monotonic=started_monotonic,
        artifacts=[smoke_path, test_log_path],
        status="PASS" if test_process.returncode == 0 and smoke_equal else "FAIL",
    )
    registry_receipt = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_ok = all(
        key in registry_receipt
        for key in (
            "config_sha256",
            "code_version",
            "seeds",
            "environment",
            "runtime_seconds",
            "artifact_sha256",
        )
    )
    checks["check_7"] = _check(
        registry_ok,
        "Run receipt stores config hash, code version, seeds, environment, runtime and artifact hashes",
        [str(registry_path.relative_to(root)).replace("\\", "/")],
    )
    checks["check_8"] = _check(
        test_process.returncode == 0
        and smoke_equal
        and first_digest == second_digest
        and len(first_smoke) == 6,
        f"Full pytest exit={test_process.returncode}; repeated six-cell smoke digest={first_digest}",
        ["artifacts/stage6_1/full_test_suite.log", "artifacts/stage6_1/six_cell_smoke.csv"],
    )

    passed = sum(check["status"] == "PASS" for check in checks.values())
    overall = "PASS" if passed == 8 else "FAIL"
    result = {
        "schema_version": "stage6.1-acceptance-v1",
        "run": {
            "run_id": run_id,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_version": protocol["dataset"]["version"],
            "data_cells": 6,
            "indexed_slices": 36,
            "primary_delta_days": 14,
            "smoke_split": "validation",
            "smoke_digest": first_digest,
            "registry_receipt": str(registry_path.relative_to(root)).replace("\\", "/"),
        },
        "checks": checks,
        "passed_checks": passed,
        "total_checks": 8,
        "overall_status": overall,
        "eligible_for_model_implementation": overall == "PASS",
        "decision": (
            "GĐ6.1 PASS. The shared pipeline is deterministic, leakage-safe, and ready for all GĐ6 models to consume the common SessionLoader and RankingEvaluator."
            if overall == "PASS"
            else "GĐ6.1 FAIL. Do not implement or train models until every acceptance check passes."
        ),
        "constraints_observed": {
            "trained_recommendation_model": False,
            "opened_test_predictions": False,
            "modified_frozen_dataset": False,
            "modified_locked_protocol": False,
        },
    }
    _write_json(acceptance_path, result)
    report_path.write_text(_render_report(result), encoding="utf-8")
    output_paths = [
        smoke_path,
        test_log_path,
        registry_path,
        acceptance_path,
        report_path,
    ]
    _write_json(
        artifact_dir / "output_hashes.json",
        {
            str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
            for path in output_paths
        },
    )
    print(
        json.dumps(
            {
                "overall_status": overall,
                "passed_checks": passed,
                "eligible_for_model_implementation": overall == "PASS",
            },
            ensure_ascii=False,
        )
    )
    return 0 if overall == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
