from __future__ import annotations

import json
import time

import pandas as pd
import pytest

from dss_stage6.data import HistoryStore, StaticGraphBuilder
from dss_stage6.data.session import SessionRecord
from dss_stage6.evaluation import RankingEvaluator
from dss_stage6.features import FeatureBuilder
from dss_stage6.registry import RunRegistry


def _actions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("prior", "e1", "old", "c", "j1", "view", "2027-01-01T09:59:59Z", "2027-01-01T09:59:59Z"),
            ("at-t", "e2", "old", "c", "j2", "view", "2027-01-01T10:00:00Z", "2027-01-01T09:59:59Z"),
            ("observed-at-t", "e3", "old", "c", "j3", "view", "2027-01-01T09:00:00Z", "2027-01-01T10:00:00Z"),
            ("current", "e4", "current", "c", "j4", "view", "2027-01-01T09:00:00Z", "2027-01-01T09:00:00Z"),
        ],
        columns=[
            "action_id", "exposure_id", "session_id", "candidate_id", "job_id",
            "action_type", "action_at", "observed_at",
        ],
    )


def _session() -> SessionRecord:
    candidates = pd.DataFrame(
        {
            "job_id": [f"job-{index:02d}" for index in range(12)],
            "label": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "new_job": [0] * 12,
            "changed_profile": [0] * 12,
            "low_history": [1] * 12,
        }
    )
    return SessionRecord(
        53021, "S0", "validation", 14, "current", "c",
        pd.Timestamp("2027-01-01T10:00:00Z"), 12, 2, True, "scoreable", candidates,
    )


def test_history_store_and_graph_exclude_boundaries_and_current_session():
    history = HistoryStore(_actions(), seed=53021, scenario="S0")
    snapshot = StaticGraphBuilder(history).build(_session())
    assert snapshot.edges["action_id"].tolist() == ["prior"]
    assert snapshot.graph.number_of_edges() == 1


def test_feature_builder_refuses_non_train_fit_and_does_not_mutate_on_transform():
    train = pd.DataFrame(
        {"split": ["train", "train"], "number": [1.0, 3.0], "category": ["a", "b"]}
    )
    validation = pd.DataFrame(
        {"split": ["validation"], "number": [5.0], "category": ["unseen"]}
    )
    builder = FeatureBuilder(numeric_columns=["number"], categorical_columns=["category"])
    builder.fit(train)
    state_hash = builder.state_hash
    transformed = builder.transform(validation)
    assert builder.state_hash == state_hash
    assert transformed["category==__UNKNOWN__"].iloc[0] == 1.0
    with pytest.raises(ValueError, match="train only"):
        FeatureBuilder(numeric_columns=["number"]).fit(validation)


def test_ranking_evaluator_requires_exact_exposed_set_and_uses_shared_metrics():
    session = _session()
    scores = {job_id: 1.0 for job_id in session.exposed_job_ids}
    result = RankingEvaluator(10).evaluate_session(session, scores)
    assert result["recall_at_10"] == 0.5
    with pytest.raises(ValueError, match="Predictions must equal exposed set"):
        RankingEvaluator(10).evaluate_session(session, {"unexposed": 1.0})


def test_run_registry_records_required_provenance(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("evidence", encoding="utf-8")
    registry = RunRegistry(tmp_path, tmp_path / "runs")
    destination = registry.register(
        run_id="run-1",
        config={"k": 10},
        seeds={"data": [53021, 53022]},
        started_monotonic=time.monotonic(),
        artifacts=[artifact],
        status="PASS",
    )
    receipt = json.loads(destination.read_text(encoding="utf-8"))
    assert receipt["config_sha256"]
    assert receipt["code_version"]["source_tree_sha256"]
    assert receipt["runtime_seconds"] >= 0
    assert receipt["artifact_sha256"]["artifact.txt"]
