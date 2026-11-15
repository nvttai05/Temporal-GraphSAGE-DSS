from __future__ import annotations

import hashlib
import json

from dss_stage6.preflight import (
    HashVerifier,
    canonical_json_sha256,
    validate_hash_invariants,
)


def test_same_path_and_hash_method_with_conflicting_actual_hashes_fails():
    rows = [
        {
            "path": "C:/handoff/candidate_v03_inputs.json",
            "hash_method": "raw_file_sha256",
            "actual_sha256": "a" * 64,
        },
        {
            "path": "C:/handoff/candidate_v03_inputs.json",
            "hash_method": "raw_file_sha256",
            "actual_sha256": "b" * 64,
        },
    ]

    violations = validate_hash_invariants(rows)

    assert len(violations) == 1
    assert violations[0]["reason"] == "multiple_actual_hashes_for_same_path_and_method"


def test_different_hash_methods_are_not_treated_as_same_measurement():
    rows = [
        {
            "path": "C:/handoff/candidate_v03_inputs.json",
            "hash_method": "raw_file_sha256",
            "actual_sha256": "a" * 64,
        },
        {
            "path": "C:/handoff/candidate_v03_inputs.json",
            "hash_method": "recorded_provenance_sha256",
            "actual_sha256": "b" * 64,
        },
    ]

    assert validate_hash_invariants(rows) == []


def test_raw_file_hash_is_computed_once_and_expected_is_not_copied(tmp_path):
    artifact = tmp_path / "receipt.json"
    artifact.write_text('{"config_sha256":"' + "c" * 64 + '"}', encoding="utf-8")
    calls = 0

    def counting_hasher(path):
        nonlocal calls
        calls += 1
        return hashlib.sha256(path.read_bytes()).hexdigest()

    verifier = HashVerifier(raw_hasher=counting_hasher)
    verifier.verify_raw("manifest", artifact, "0" * 64)
    verifier.verify_raw("handoff", artifact, "1" * 64)

    assert calls == 1
    actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert {row["actual_sha256"] for row in verifier.rows} == {actual}
    assert all(row["actual_sha256"] != row["expected_sha256"] for row in verifier.rows)
    assert verifier.invariant_report()["status"] == "PASS"


def test_canonical_and_recorded_json_hash_methods_are_explicit(tmp_path):
    artifact = tmp_path / "receipt.json"
    value = {"z": 1, "config_sha256": "d" * 64, "a": [2, 3]}
    artifact.write_text(json.dumps(value, indent=2), encoding="utf-8")
    verifier = HashVerifier()

    verifier.verify_canonical_json(
        "canonical",
        artifact,
        canonical_json_sha256(value),
    )
    verifier.verify_recorded_provenance(
        "recorded",
        artifact,
        "d" * 64,
        "config_sha256",
    )

    assert [row["hash_method"] for row in verifier.rows] == [
        "canonical_json_sha256",
        "recorded_provenance_sha256",
    ]
    assert all(row["status"] == "MATCH" for row in verifier.rows)
