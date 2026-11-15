from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dss_stage6.data import FrozenTemporalDataset, filter_actions_as_of
from dss_stage6.evaluation import ndcg_at_k, recall_at_k


CHECK_NAMES = {
    "check_1": "Manifest and handoff SHA-256",
    "check_2": "Loader coverage and future-free graph snapshots",
    "check_3": "Label and Top-K eligibility statistics",
    "check_4": "Hand-calculated Recall@10/NDCG@10 fixture",
    "check_5": "Exposed-set ranking and rolling as-of-t compliance",
    "check_6": "Experiment contract completeness",
}
VALID_STATUSES = {"PASS", "FAIL", "BLOCKED"}
HASH_METHODS = {
    "raw_file_sha256",
    "canonical_json_sha256",
    "recorded_provenance_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_hash_invariants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return conflicts where one path/method was assigned multiple actual hashes."""

    grouped: dict[tuple[str, str], set[str]] = {}
    violations: list[dict[str, Any]] = []
    for row in rows:
        method = str(row.get("hash_method", ""))
        if method not in HASH_METHODS:
            violations.append(
                {
                    "path": str(row.get("path", "")),
                    "hash_method": method,
                    "actual_sha256_values": [],
                    "reason": "unknown_hash_method",
                }
            )
            continue
        actual = str(row.get("actual_sha256", ""))
        if actual:
            grouped.setdefault((str(row.get("path", "")), method), set()).add(actual)
    for (path, method), values in sorted(grouped.items()):
        if len(values) > 1:
            violations.append(
                {
                    "path": path,
                    "hash_method": method,
                    "actual_sha256_values": sorted(values),
                    "reason": "multiple_actual_hashes_for_same_path_and_method",
                }
            )
    return violations


class HashVerifier:
    """Checksum recorder with one raw-byte hash computation per resolved path."""

    def __init__(self, raw_hasher=sha256_file):
        self.rows: list[dict[str, Any]] = []
        self._raw_hasher = raw_hasher
        self._raw_cache: dict[str, str] = {}
        self.raw_hash_computations: dict[str, int] = {}
        self.raw_requested_paths: set[str] = set()

    @staticmethod
    def _resolved(path: Path) -> str:
        return str(path.resolve())

    def _append(
        self,
        source: str,
        path: Path,
        expected: str,
        actual: str,
        method: str,
        *,
        missing: bool = False,
    ) -> None:
        status = "MISSING" if missing else ("MATCH" if actual and actual == expected else "MISMATCH")
        self.rows.append(
            {
                "source": source,
                "path": self._resolved(path),
                "hash_method": method,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": status,
            }
        )

    def verify_raw(self, source: str, path: Path, expected: str) -> None:
        resolved = self._resolved(path)
        self.raw_requested_paths.add(resolved)
        if not path.is_file():
            self._append(source, path, expected, "", "raw_file_sha256", missing=True)
            return
        if resolved not in self._raw_cache:
            self._raw_cache[resolved] = self._raw_hasher(path)
            self.raw_hash_computations[resolved] = self.raw_hash_computations.get(resolved, 0) + 1
        self._append(source, path, expected, self._raw_cache[resolved], "raw_file_sha256")

    def verify_canonical_json(self, source: str, path: Path, expected: str) -> None:
        if not path.is_file():
            self._append(source, path, expected, "", "canonical_json_sha256", missing=True)
            return
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        self._append(source, path, expected, canonical_json_sha256(value), "canonical_json_sha256")

    def verify_recorded_provenance(
        self,
        source: str,
        path: Path,
        expected: str,
        field: str,
    ) -> None:
        if not path.is_file():
            self._append(source, path, expected, "", "recorded_provenance_sha256", missing=True)
            return
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        actual = value.get(field, "") if isinstance(value, dict) else ""
        self._append(
            source,
            path,
            expected,
            str(actual),
            "recorded_provenance_sha256",
            missing=not bool(actual),
        )

    def invariant_report(self) -> dict[str, Any]:
        conflicts = validate_hash_invariants(self.rows)
        existing_raw_paths = {
            path for path in self.raw_requested_paths if Path(path).is_file()
        }
        computed_once = (
            set(self.raw_hash_computations) == existing_raw_paths
            and all(count == 1 for count in self.raw_hash_computations.values())
        )
        return {
            "supported_hash_methods": sorted(HASH_METHODS),
            "raw_hash_requests": sum(
                row["hash_method"] == "raw_file_sha256" for row in self.rows
            ),
            "unique_existing_raw_paths": len(existing_raw_paths),
            "raw_hash_computations": sum(self.raw_hash_computations.values()),
            "raw_sha256_computed_once_per_resolved_path": computed_once,
            "same_path_and_method_has_unique_actual_sha256": not conflicts,
            "conflicts": conflicts,
            "status": "PASS" if computed_once and not conflicts else "FAIL",
        }


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected mapping in {path}")
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def record_check(status: str, summary: str, evidence: list[str]) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(status)
    return {"status": status, "summary": summary, "evidence": evidence}


def check_manifest_hashes(
    root: Path,
    dataset_root: Path,
    handoff_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    candidate_root = dataset_root / "data" / "candidate55" / "v0_3"
    manifest_path = candidate_root / "dataset_manifest.json"
    verifier = HashVerifier()
    if not manifest_path.is_file():
        verifier.verify_raw("manifest", manifest_path, "")
        write_csv(artifact_dir / "checksum_verification.csv", verifier.rows)
        invariant_report = verifier.invariant_report()
        write_json(artifact_dir / "checksum_invariants.json", invariant_report)
        return record_check(
            "BLOCKED",
            "dataset_manifest.json is unavailable",
            ["checksum_verification.csv", "checksum_invariants.json"],
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for source in manifest.get("source_logs", []):
        base = dataset_root / source["path"]
        for relative, expected in source.get("files_sha256", {}).items():
            verifier.verify_raw("manifest.source_logs", base / relative, expected)
    for relative, metadata in manifest.get("files", {}).items():
        verifier.verify_raw("manifest.files", candidate_root / relative, metadata["sha256"])
    for section in ("qc", "eda"):
        metadata = manifest.get(section, {})
        for key, expected_key in (("path", "sha256"), ("report_path", "report_sha256"), ("validation_path", "validation_sha256")):
            if key in metadata:
                verifier.verify_raw(f"manifest.{section}", dataset_root / metadata[key], metadata[expected_key])

    # The input receipts connect each seed to the frozen processed source files.
    for seed in manifest.get("seeds", []):
        receipt_path = dataset_root / "data" / "candidate55" / "v0_3_logs" / f"seed_{seed}" / "candidate_v03_inputs.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            verifier.verify_raw("input_receipt.jobs", dataset_root / "data" / "processed" / "jobs_clean.csv", receipt["jobs_sha256"])
            verifier.verify_raw("input_receipt.profiles", dataset_root / "data" / "processed" / "candidate_profiles_clean.csv", receipt["profiles_sha256"])
            expected_config = manifest.get("configs_sha256", {}).get(str(seed), "")
            verifier.verify_recorded_provenance(
                "input_receipt.config",
                receipt_path,
                expected_config,
                "config_sha256",
            )

    # Stage 5 evidence uses fixed package locations documented by its contract.
    evidence_paths = {
        "source_zip_sha256": dataset_root / "data" / "raw" / "MEDAJob.zip",
        "generator_sha256": dataset_root / "scripts" / "stage5_evidence" / "generate_candidate_v03.py",
        "builder_sha256": dataset_root / "scripts" / "stage5_evidence" / "build_candidate_v03.py",
        "loader_sha256": dataset_root / "scripts" / "stage5_evidence" / "candidate_v03_loader.py",
    }
    for key, path in evidence_paths.items():
        if key in manifest:
            verifier.verify_raw(f"manifest.{key}", path, manifest[key])

    if handoff_path.is_file():
        # utf-8-sig accepts both BOM and non-BOM checksum manifests.
        for line_number, line in enumerate(handoff_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                verifier.rows.append(
                    {
                        "source": "handoff_file",
                        "path": f"line:{line_number}",
                        "hash_method": "raw_file_sha256",
                        "expected_sha256": "",
                        "actual_sha256": "",
                        "status": "MALFORMED",
                    }
                )
                continue
            expected, relative = parts
            relative = relative.lstrip("* ")
            verifier.verify_raw("handoff_file", handoff_path.parent / relative, expected)
    else:
        verifier.rows.append(
            {
                "source": "configured_handoff_file",
                "path": str(handoff_path.resolve()),
                "hash_method": "raw_file_sha256",
                "expected_sha256": "",
                "actual_sha256": "",
                "status": "MISSING",
            }
        )

    rows = verifier.rows
    invariant_report = verifier.invariant_report()
    write_csv(artifact_dir / "checksum_verification.csv", rows)
    write_json(artifact_dir / "checksum_invariants.json", invariant_report)
    statuses = {row["status"] for row in rows}
    matched = sum(row["status"] == "MATCH" for row in rows)
    missing = sum(row["status"] == "MISSING" for row in rows)
    unavailable = sum(row["status"] == "UNAVAILABLE" for row in rows)
    if invariant_report["status"] != "PASS" or "MISMATCH" in statuses or "MALFORMED" in statuses:
        status = "FAIL"
    elif statuses & {"MISSING", "UNAVAILABLE"}:
        status = "BLOCKED"
    else:
        status = "PASS"
    return record_check(
        status,
        (
            f"{matched}/{len(rows)} checksum records matched; {missing} referenced files missing and {unavailable} hashes have no resolvable artifact path"
            if status == "BLOCKED"
            else f"{matched}/{len(rows)} checksum records matched; checksum invariants {invariant_report['status']}"
        ),
        ["checksum_verification.csv", "checksum_invariants.json"],
    )


def check_loader(
    dataset: FrozenTemporalDataset,
    protocol: dict[str, Any],
    artifact_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    sessions = dataset.session_splits()
    pool = dataset.benchmark_pool()
    expected_cells = [
        (seed, scenario, delta, split)
        for seed in protocol["dataset"]["data_seeds"]
        for scenario in protocol["dataset"]["scenarios"]
        for delta in (protocol["dataset"]["primary_label_window_days"], protocol["dataset"]["sensitivity_label_window_days"])
        for split in protocol["dataset"]["splits"]
    ]
    session_counts = sessions.groupby(["seed", "scenario", "delta_days", "split"]).size()
    pool_counts = pool.groupby(["seed", "scenario", "delta_days", "split"]).size()
    coverage: list[dict[str, Any]] = []
    failures: list[str] = []
    action_cache: dict[tuple[int, str], pd.DataFrame] = {}
    leakage_rows: list[dict[str, Any]] = []

    for seed, scenario, delta, split in expected_cells:
        key = (seed, scenario, delta, split)
        session_count = int(session_counts.get(key, 0))
        pool_count = int(pool_counts.get(key, 0))
        coverage.append(
            {
                "seed": seed,
                "scenario": scenario,
                "delta_days": delta,
                "split": split,
                "session_rows": session_count,
                "benchmark_rows": pool_count,
                "status": "PASS" if session_count > 0 and pool_count > 0 else "FAIL",
            }
        )
        if session_count == 0 or pool_count == 0:
            failures.append(f"empty cell {key}")
            continue
        sample = sessions.loc[
            (sessions["seed"] == seed)
            & (sessions["scenario"] == scenario)
            & (sessions["delta_days"] == delta)
            & (sessions["split"] == split)
        ].sort_values("session_id").iloc[0]
        action_key = (seed, scenario)
        if action_key not in action_cache:
            action_cache[action_key] = dataset.actions(seed, scenario)
        snapshot = filter_actions_as_of(
            action_cache[action_key],
            sample["prediction_at"],
            current_session_id=str(sample["session_id"]),
        )
        prediction = pd.Timestamp(sample["prediction_at"])
        violation_count = int(
            (
                (pd.to_datetime(snapshot["action_at"], utc=True) >= prediction)
                | (pd.to_datetime(snapshot["observed_at"], utc=True) >= prediction)
                | (snapshot["session_id"].astype(str) == str(sample["session_id"]))
            ).sum()
        )
        leakage_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "delta_days": delta,
                "split": split,
                "sample_session_id": sample["session_id"],
                "prediction_at": sample["prediction_at"],
                "snapshot_edges": len(snapshot),
                "violations": violation_count,
                "status": "PASS" if violation_count == 0 else "FAIL",
            }
        )
        if violation_count:
            failures.append(f"leakage in {key}")

    actual_cells = set(map(tuple, sessions[["seed", "scenario", "delta_days", "split"]].drop_duplicates().to_numpy()))
    extras = actual_cells - set(expected_cells)
    if extras:
        failures.append(f"unexpected cells: {sorted(extras)}")
    write_csv(artifact_dir / "loader_coverage.csv", coverage)
    write_csv(artifact_dir / "leakage_deterministic_samples.csv", leakage_rows)
    status = "FAIL" if failures else "PASS"
    return (
        record_check(
            status,
            f"Loaded {len(expected_cells)} expected seed/scenario/delta/split cells; {len(leakage_rows)} deterministic snapshots audited with {sum(r['violations'] for r in leakage_rows)} violations",
            ["loader_coverage.csv", "leakage_deterministic_samples.csv"],
        ),
        sessions,
        pool,
    )


def load_label_frames(
    dataset: FrozenTemporalDataset,
    sessions: pd.DataFrame,
    seeds: list[int],
    scenarios: list[str],
    delta_days: int,
) -> dict[tuple[int, str, int], pd.DataFrame]:
    result: dict[tuple[int, str, int], pd.DataFrame] = {}
    split_map = sessions.loc[sessions["delta_days"] == delta_days, ["seed", "scenario", "session_id", "split"]]
    for seed in seeds:
        for scenario in scenarios:
            path = dataset.cell_path(seed, scenario, f"labels_d{delta_days}.csv")
            labels = pd.read_csv(path)
            required = {"exposure_id", "session_id", "candidate_id", "job_id", "shown_at", "label", "label_status", "label_available_at", "new_job", "changed_profile", "low_history"}
            missing = sorted(required - set(labels.columns))
            if missing:
                raise ValueError(f"Schema mismatch in {path}: missing columns {missing}")
            mapping = split_map.loc[(split_map["seed"] == seed) & (split_map["scenario"] == scenario), ["session_id", "split"]]
            labels = labels.merge(mapping, on="session_id", how="left", validate="many_to_one")
            labels["seed"] = seed
            labels["scenario"] = scenario
            labels["delta_days"] = delta_days
            result[(seed, scenario, delta_days)] = labels
    return result


def check_statistics(
    dataset: FrozenTemporalDataset,
    protocol: dict[str, Any],
    sessions: pd.DataFrame,
    pool: pd.DataFrame,
    artifact_dir: Path,
) -> tuple[dict[str, Any], dict[tuple[int, str, int], pd.DataFrame]]:
    delta = protocol["dataset"]["primary_label_window_days"]
    labels_by_cell = load_label_frames(
        dataset,
        sessions,
        protocol["dataset"]["data_seeds"],
        protocol["dataset"]["scenarios"],
        delta,
    )
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    allowed_statuses = {"positive", "negative", "censored_close", "repeat_suppressed"}

    for (seed, scenario, _), labels in labels_by_cell.items():
        if labels["split"].isna().any():
            failures.append(f"unmapped sessions for {seed}/{scenario}")
        unknown = set(labels["label_status"].dropna().unique()) - allowed_statuses
        if unknown:
            failures.append(f"unknown label statuses {unknown} for {seed}/{scenario}")
        valid = labels["label_status"].isin(["positive", "negative"])
        if not labels.loc[valid, "label"].isin([0, 1]).all():
            failures.append(f"non-binary valid labels for {seed}/{scenario}")
        if labels.loc[~valid, "label"].notna().any():
            failures.append(f"excluded/censored rows carry labels for {seed}/{scenario}")

        expected_counts = sessions.loc[
            (sessions["seed"] == seed)
            & (sessions["scenario"] == scenario)
            & (sessions["delta_days"] == delta),
            ["session_id", "valid_label_count", "positive_count", "top10_eligible"],
        ]
        evaluation_cell = pool.loc[
            (pool["seed"] == seed) & (pool["scenario"] == scenario) & (pool["delta_days"] == delta)
        ]
        observed_counts = evaluation_cell.groupby("session_id").agg(
            observed_valid=("label", "size"),
            observed_positive=("label", "sum"),
        )
        count_audit = expected_counts.merge(observed_counts, on="session_id", how="outer", validate="one_to_one")
        count_audit[["observed_valid", "observed_positive"]] = count_audit[["observed_valid", "observed_positive"]].fillna(0)
        count_mismatch = (
            count_audit[["valid_label_count", "positive_count", "observed_valid", "observed_positive"]].isna().any(axis=1)
            | (count_audit["valid_label_count"] != count_audit["observed_valid"])
            | (count_audit["positive_count"] != count_audit["observed_positive"])
        )
        computed_top10 = (
            (count_audit["observed_valid"] >= protocol["ranking"]["minimum_valid_candidates"])
            & (count_audit["observed_positive"] >= protocol["ranking"]["minimum_positive_labels"])
        )
        eligibility_mismatch = count_audit["top10_eligible"].fillna(-1).astype(int) != computed_top10.astype(int)
        if count_mismatch.any():
            failures.append(f"session label counts disagree for {seed}/{scenario}: {int(count_mismatch.sum())}")
        if eligibility_mismatch.any():
            failures.append(f"Top-10 flags disagree for {seed}/{scenario}: {int(eligibility_mismatch.sum())}")

        for split in protocol["dataset"]["splits"]:
            split_labels = labels.loc[labels["split"] == split].copy()
            split_evaluation = evaluation_cell.loc[evaluation_cell["split"] == split].copy()
            all_sessions = set(split_labels["session_id"])
            group_sessions = {
                "all": all_sessions,
                "changed_profile": set(split_labels.loc[split_labels["changed_profile"] == 1, "session_id"]),
                "low_history": set(split_labels.loc[split_labels["low_history"] == 1, "session_id"]),
                "new_job": set(split_evaluation.loc[(split_evaluation["new_job"] == 1) & (split_evaluation["label"] == 1), "session_id"]),
            }
            for group, session_ids in group_sessions.items():
                subset = split_labels.loc[split_labels["session_id"].isin(session_ids)]
                evaluation_subset = split_evaluation.loc[split_evaluation["session_id"].isin(session_ids)]
                eligible = expected_counts.loc[
                    expected_counts["session_id"].isin(session_ids), "top10_eligible"
                ]
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "delta_days": delta,
                        "split": split,
                        "group": group,
                        "qualifying_sessions": len(session_ids),
                        "candidate_rows_full_set": len(subset),
                        "positive": int((evaluation_subset["label"] == 1).sum()),
                        "negative": int((evaluation_subset["label"] == 0).sum()),
                        "censored_or_excluded": int(len(subset) - len(evaluation_subset)),
                        "censored_close": int((subset["label_status"] == "censored_close").sum()),
                        "repeat_suppressed": int((subset["label_status"] == "repeat_suppressed").sum()),
                        "split_boundary_or_pool_excluded_valid": int(
                            subset["label_status"].isin(["positive", "negative"]).sum() - len(evaluation_subset)
                        ),
                        "top10_eligible_sessions": int(eligible.sum()),
                    }
                )

    write_csv(artifact_dir / "label_and_eligibility_statistics.csv", rows)
    status = "FAIL" if failures else "PASS"
    return (
        record_check(
            status,
            f"Recorded positive/negative/censored and Top-10 eligibility for {len(rows)} primary-window seed/scenario/split/cohort groups",
            ["label_and_eligibility_statistics.csv"],
        ),
        labels_by_cell,
    )


def check_metric_fixture(root: Path, artifact_dir: Path) -> dict[str, Any]:
    labels = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    scores = list(reversed(range(12)))
    job_ids = [f"job-{index:02d}" for index in range(12)]
    expected_recall = 0.5
    expected_ndcg = (1.0 / __import__("math").log2(3)) / (1.0 + 1.0 / __import__("math").log2(3))
    actual_recall = recall_at_k(labels, scores, 10, job_ids)
    actual_ndcg = ndcg_at_k(labels, scores, 10, job_ids)
    fixture_ok = actual_recall == expected_recall and abs(actual_ndcg - expected_ndcg) <= 1e-12
    fixture = {
        "description": "two positives at score ranks 2 and 11; only rank 2 is a Top-10 hit",
        "expected": {"recall_at_10": expected_recall, "ndcg_at_10": expected_ndcg},
        "actual": {"recall_at_10": actual_recall, "ndcg_at_10": actual_ndcg},
        "tolerance": 1e-12,
        "status": "PASS" if fixture_ok else "FAIL",
    }
    write_json(artifact_dir / "metric_fixture.json", fixture)
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    (artifact_dir / "full_test_suite.log").write_text(
        f"exit_code={process.returncode}\n{process.stdout}{process.stderr}", encoding="utf-8"
    )
    status = "PASS" if fixture_ok and process.returncode == 0 else "FAIL"
    return record_check(
        status,
        f"Hand fixture Recall@10={actual_recall:.12f}, NDCG@10={actual_ndcg:.12f}; full pytest exit={process.returncode}",
        ["metric_fixture.json", "full_test_suite.log"],
    )


def check_protocol_compliance(
    dataset: FrozenTemporalDataset,
    protocol: dict[str, Any],
    sessions: pd.DataFrame,
    pool: pd.DataFrame,
    primary_labels: dict[tuple[int, str, int], pd.DataFrame],
    artifact_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    labels_by_cell = dict(primary_labels)
    secondary = protocol["dataset"]["sensitivity_label_window_days"]
    labels_by_cell.update(
        load_label_frames(dataset, sessions, protocol["dataset"]["data_seeds"], protocol["dataset"]["scenarios"], secondary)
    )

    for (seed, scenario, delta), labels in labels_by_cell.items():
        bench = pool.loc[(pool["seed"] == seed) & (pool["scenario"] == scenario) & (pool["delta_days"] == delta)].copy()
        valid = labels.loc[labels["label_status"].isin(["positive", "negative"])].copy()
        canonical = ["session_id", "candidate_id", "job_id", "shown_at", "label", "label_available_at", "new_job", "changed_profile", "low_history", "split"]
        valid["label"] = valid["label"].astype(int)
        bench["label"] = bench["label"].astype(int)
        # Contract direction: every benchmark row must be a valid exposed label.
        # Ineligible sessions may be absent from the benchmark pool.
        comparison = bench[canonical].merge(valid[canonical], on=canonical, how="left", indicator=True)
        mismatch = int((comparison["_merge"] != "both").sum())

        eligible_session_ids = set(
            sessions.loc[
                (sessions["seed"] == seed)
                & (sessions["scenario"] == scenario)
                & (sessions["delta_days"] == delta)
                & (sessions["top10_eligible"] == 1),
                "session_id",
            ]
        )
        eligible_valid = valid.loc[valid["session_id"].isin(eligible_session_ids), canonical]
        eligible_bench = bench.loc[bench["session_id"].isin(eligible_session_ids), canonical]
        eligible_comparison = eligible_valid.merge(eligible_bench, on=canonical, how="outer", indicator=True)
        eligible_full_set_mismatch = int((eligible_comparison["_merge"] != "both").sum())

        impressions = pd.read_csv(dataset.cell_path(seed, scenario, "impressions.csv"), usecols=["exposure_id", "session_id", "job_id"])
        exposed_merge = valid[["exposure_id", "session_id", "job_id"]].merge(
            impressions, on=["exposure_id", "session_id", "job_id"], how="left", indicator=True
        )
        unexposed_valid = int((exposed_merge["_merge"] != "both").sum())

        eligible = pd.read_csv(dataset.cell_path(seed, scenario, "eligible_jobs.csv"), usecols=["session_id", "job_id", "shown"])
        unshown = eligible.loc[eligible["shown"] == 0, ["session_id", "job_id"]].drop_duplicates()
        bench_pairs = bench[["session_id", "job_id"]].drop_duplicates()
        unshown_in_pool = len(unshown.merge(bench_pairs, on=["session_id", "job_id"], how="inner"))
        censored_in_pool = int(bench["label"].isna().sum())
        cell_ok = mismatch == 0 and eligible_full_set_mismatch == 0 and unexposed_valid == 0 and unshown_in_pool == 0 and censored_in_pool == 0
        if not cell_ok:
            failures.append(f"protocol mismatch {seed}/{scenario}/d{delta}")
        rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "delta_days": delta,
                "valid_label_rows": len(valid),
                "benchmark_rows": len(bench),
                "valid_vs_pool_mismatches": mismatch,
                "eligible_full_set_mismatches": eligible_full_set_mismatch,
                "valid_rows_without_impression": unexposed_valid,
                "unshown_pairs_in_pool": unshown_in_pool,
                "censored_rows_in_pool": censored_in_pool,
                "status": "PASS" if cell_ok else "FAIL",
            }
        )

    history = protocol.get("history", {})
    config_ok = (
        protocol.get("ranking", {}).get("scope") == "exposed_set"
        and protocol.get("ranking", {}).get("exclude_censored") is True
        and protocol.get("ranking", {}).get("exclude_unexposed") is True
        and history.get("mode") == "rolling_as_of_t_no_retrain"
        and history.get("require_action_at_before_prediction") is True
        and history.get("require_observed_at_before_prediction") is True
        and history.get("exclude_current_session_edges") is True
        and history.get("update_model_weights_on_validation") is False
        and history.get("update_model_weights_on_test") is False
    )
    rows.append(
        {
            "seed": "CONFIG",
            "scenario": "CONFIG",
            "delta_days": "CONFIG",
            "valid_label_rows": "",
            "benchmark_rows": "",
            "valid_vs_pool_mismatches": "",
            "eligible_full_set_mismatches": "",
            "valid_rows_without_impression": "",
            "unshown_pairs_in_pool": "",
            "censored_rows_in_pool": "",
            "status": "PASS" if config_ok else "FAIL",
        }
    )
    if not config_ok:
        failures.append("stage6 protocol flags do not encode required ranking/history policy")
    write_csv(artifact_dir / "protocol_compliance.csv", rows)
    status = "FAIL" if failures else "PASS"
    return record_check(
        status,
        f"Audited exposed-set membership, eligible full candidate sets, and rolling policy across {len(labels_by_cell)} seed/scenario/window cells",
        ["protocol_compliance.csv", "leakage_deterministic_samples.csv", "full_test_suite.log"],
    )


def contains_tbd(value: Any) -> bool:
    if isinstance(value, dict):
        return any(contains_tbd(key) or contains_tbd(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_tbd(item) for item in value)
    return isinstance(value, str) and "TBD" in value.upper()


def check_experiment_contract(experiments: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    expected_models = ["random", "popularity_asof_t", "cbf", "xgboost_static", "xgboost_time", "static_graphsage"]
    models = experiments.get("models", {})
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    if list(models) != expected_models:
        failures.append(f"model list is {list(models)}, expected {expected_models}")

    applicability = {
        "random": ("featureless sanity baseline", "not applicable"),
        "popularity_asof_t": ("rolling interaction counts", "not applicable"),
        "cbf": ("features", "similarity ranking; no fitted loss"),
        "xgboost_static": ("features", "objective"),
        "xgboost_time": ("features", "objective"),
        "static_graphsage": ("features", "loss"),
    }
    for name in expected_models:
        config = models.get(name)
        issues: list[str] = []
        if not isinstance(config, dict):
            issues.append("missing model config")
            config = {}
        budget = config.get("tuning_budget")
        if not isinstance(budget, int) or budget <= 0:
            issues.append("invalid tuning_budget")
        feature_rule, loss_rule = applicability[name]
        if feature_rule == "features" and not config.get("features"):
            issues.append("missing features")
        if loss_rule in {"loss", "objective"} and not config.get(loss_rule):
            issues.append(f"missing {loss_rule}")
        if contains_tbd(config):
            issues.append("contains TBD")
        rows.append(
            {
                "model": name,
                "features": json.dumps(config.get("features", feature_rule), ensure_ascii=False),
                "loss_or_objective": config.get("loss", config.get("objective", loss_rule)),
                "tuning_budget": budget,
                "official_environment": config.get("official_environment", ""),
                "status": "PASS" if not issues else "FAIL",
                "issues": "; ".join(issues),
            }
        )
        failures.extend(f"{name}: {issue}" for issue in issues)

    seeds = experiments.get("seeds", {})
    seed_ok = (
        seeds.get("data") == [53021, 53022]
        and isinstance(seeds.get("tuning_model_seed"), int)
        and seeds.get("final_model_seeds") == [42, 43, 44]
    )
    rows.append(
        {
            "model": "GLOBAL_SEEDS",
            "features": "not applicable",
            "loss_or_objective": "not applicable",
            "tuning_budget": "not applicable",
            "official_environment": "not applicable",
            "status": "PASS" if seed_ok else "FAIL",
            "issues": "" if seed_ok else "data/tuning/final seeds incomplete",
        }
    )
    if not seed_ok:
        failures.append("seed contract incomplete")
    if contains_tbd(experiments):
        failures.append("experiments.yaml contains TBD")
    write_csv(artifact_dir / "experiment_contract_completeness.csv", rows)
    return record_check(
        "FAIL" if failures else "PASS",
        "Model list, applicable feature/loss definitions, seeds, and tuning budgets are explicit" if not failures else "; ".join(failures),
        ["experiment_contract_completeness.csv"],
    )


def git_commit(root: Path) -> str:
    process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return process.stdout.strip() if process.returncode == 0 else "UNAVAILABLE_NOT_A_GIT_WORKTREE"


def render_report(results: dict[str, Any], artifact_dir: Path) -> str:
    lines = [
        "# DSS Stage 6 Data & Evaluation Preflight",
        "",
        f"Run UTC: `{results['run']['started_at_utc']}`  ",
        f"Dataset: `{results['run']['dataset_version']}` (frozen, read-only)  ",
        f"Overall: **{results['overall_status']}** — **{results['passed_checks']}/6 PASS**  ",
        f"Eligible for GĐ6.1: **{'YES' if results['eligible_for_stage_6_1'] else 'NO'}**",
        "",
        "No model was trained and no prediction file was opened. Δ14 is the primary window; Δ7 was used only to verify handoff/ranking consistency.",
        "",
        "## Gate results",
        "",
        "| Check | Status | Evidence-backed result |",
        "|---|---:|---|",
    ]
    for key, name in CHECK_NAMES.items():
        check = results["checks"][key]
        evidence = ", ".join(f"`artifacts/preflight/{item}`" for item in check["evidence"])
        lines.append(f"| {key[-1]}. {name} | **{check['status']}** | {check['summary']} Evidence: {evidence}. |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            results["decision"],
            "",
            "## Reproduction",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\run_stage6_preflight.py",
            ".\\.venv\\Scripts\\python.exe -m pytest -q",
            "```",
            "",
            "Machine-readable gate output: `artifacts/preflight/preflight_results.json`. Run metadata and hashes are stored beside it.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the reproducible DSS Stage 6.0 preflight gate")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    protocol_path = root / "configs" / "stage6_protocol.yaml"
    experiments_path = root / "configs" / "experiments.yaml"
    paths_path = root / "configs" / "paths.local.yaml"
    protocol = read_yaml(protocol_path)
    experiments = read_yaml(experiments_path)
    paths = read_yaml(paths_path)
    dataset_root = Path(paths["dataset_root"]).resolve()
    handoff_path = Path(paths["handoff_checksums"]).resolve()
    artifact_dir = (root / paths.get("artifacts_dir", "artifacts") / "preflight").resolve()
    report_path = (root / paths.get("reports_dir", "reports") / "DSS_STAGE6_DATA_PREFLIGHT.md").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    dataset = FrozenTemporalDataset(dataset_root)
    checks: dict[str, Any] = {}
    checks["check_1"] = check_manifest_hashes(root, dataset_root, handoff_path, artifact_dir)
    checks["check_2"], sessions, pool = check_loader(dataset, protocol, artifact_dir)
    checks["check_3"], primary_labels = check_statistics(dataset, protocol, sessions, pool, artifact_dir)
    checks["check_4"] = check_metric_fixture(root, artifact_dir)
    checks["check_5"] = check_protocol_compliance(dataset, protocol, sessions, pool, primary_labels, artifact_dir)
    checks["check_6"] = check_experiment_contract(experiments, artifact_dir)

    passed = sum(check["status"] == "PASS" for check in checks.values())
    overall = "PASS" if passed == 6 else ("FAIL" if any(check["status"] == "FAIL" for check in checks.values()) else "BLOCKED")
    eligible = passed == 6
    decision = (
        "All six gates passed; the repository is eligible to proceed to GĐ6.1."
        if eligible
        else "Not eligible to proceed to GĐ6.1. Resolve every FAIL/BLOCKED item and rerun the complete gate; do not infer approval from prior GĐ5 reports."
    )
    results = {
        "schema_version": "stage6-preflight-v1",
        "run": {
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_version": protocol["dataset"]["version"],
            "dataset_root": str(dataset_root),
            "primary_label_window_days": protocol["dataset"]["primary_label_window_days"],
            "sensitivity_label_window_days": protocol["dataset"]["sensitivity_label_window_days"],
            "data_seeds": protocol["dataset"]["data_seeds"],
            "model_seeds": {
                "tuning": experiments["seeds"]["tuning_model_seed"],
                "final": experiments["seeds"]["final_model_seeds"],
            },
            "git_commit": git_commit(root),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "checks": checks,
        "passed_checks": passed,
        "total_checks": 6,
        "overall_status": overall,
        "eligible_for_stage_6_1": eligible,
        "decision": decision,
        "constraints_observed": {
            "trained_model": False,
            "opened_test_predictions": False,
            "modified_frozen_dataset": False,
        },
    }
    results_path = artifact_dir / "preflight_results.json"
    write_json(results_path, results)
    report_path.write_text(render_report(results, artifact_dir), encoding="utf-8")

    metadata = {
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": results["run"]["git_commit"],
        "config_sha256": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (protocol_path, experiments_path, paths_path)
        },
        "code_sha256": {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted((root / "src" / "dss_stage6").rglob("*.py"))
        },
        "seeds": {"data": protocol["dataset"]["data_seeds"], **experiments["seeds"]},
        "environment": {"python": sys.version, "platform": platform.platform(), "executable": sys.executable},
    }
    write_json(artifact_dir / "run_metadata.json", metadata)
    output_hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted([*artifact_dir.glob("*"), report_path])
        if path.is_file() and path.name != "output_hashes.json"
    }
    write_json(artifact_dir / "output_hashes.json", output_hashes)
    print(json.dumps({"overall_status": overall, "passed_checks": passed, "eligible_for_stage_6_1": eligible}, ensure_ascii=False))
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
