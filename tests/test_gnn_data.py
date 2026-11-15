from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from dss_stage6.data import DatasetIndex
from dss_stage6.evaluation import RankingEvaluator
from dss_stage6.gnn import GNNGraphAdapter


ROOT = Path(__file__).resolve().parents[1]


def _index() -> DatasetIndex:
    protocol = yaml.safe_load((ROOT / "configs" / "stage6_protocol.yaml").read_text(encoding="utf-8"))
    paths = yaml.safe_load((ROOT / "configs" / "paths.local.yaml").read_text(encoding="utf-8"))
    return DatasetIndex(paths["dataset_root"], protocol)


def test_adapter_excludes_future_and_current_session_edges_and_keeps_exposed_targets():
    adapter = GNNGraphAdapter(_index(), 53021, "S1")
    session = adapter.sessions("validation", eligible_only=True, max_sessions=1)[0]
    example = adapter.build(session)
    assert example.audit["future_or_current_violations"] == 0
    assert example.audit["target_pair_count"] == len(session.candidates)
    assert example.labels.tolist() == list(map(float, session.labels))
    assert (example.edge_age_days >= 0).all()
    assert (example.edge_age_days[~example.edge_is_event] == 0).all()


def test_adapter_rejects_injected_future_and_current_session_events():
    adapter = GNNGraphAdapter(_index(), 53021, "S1")
    session = adapter.sessions("validation", eligible_only=True, max_sessions=1)[0]
    baseline = adapter.build(session)
    prediction = pd.Timestamp(session.prediction_at)
    job_id = session.exposed_job_ids[0]

    current_action = adapter.history._actions.iloc[0].copy()
    current_action["action_id"] = "injected-current-action"
    current_action["session_id"] = session.session_id
    current_action["candidate_id"] = session.candidate_id
    current_action["job_id"] = job_id
    current_action["action_at"] = prediction - pd.Timedelta(days=1)
    current_action["observed_at"] = prediction - pd.Timedelta(hours=1)
    future_action = current_action.copy()
    future_action["action_id"] = "injected-future-action"
    future_action["session_id"] = "other-session"
    future_action["action_at"] = prediction + pd.Timedelta(seconds=1)
    future_action["observed_at"] = prediction + pd.Timedelta(seconds=1)
    adapter.history._actions = pd.concat(
        [adapter.history._actions, pd.DataFrame([current_action, future_action])],
        ignore_index=True,
    )

    current_impression = adapter.impressions.iloc[0].copy()
    current_impression["session_id"] = session.session_id
    current_impression["candidate_id"] = session.candidate_id
    current_impression["job_id"] = job_id
    current_impression["shown_at"] = prediction - pd.Timedelta(days=1)
    future_impression = current_impression.copy()
    future_impression["session_id"] = "other-session"
    future_impression["shown_at"] = prediction + pd.Timedelta(seconds=1)
    adapter.impressions = pd.concat(
        [adapter.impressions, pd.DataFrame([current_impression, future_impression])],
        ignore_index=True,
    )

    rebuilt = adapter.build(session)
    assert rebuilt.audit["future_or_current_violations"] == 0
    assert rebuilt.audit["event_edge_count"] == baseline.audit["event_edge_count"]


def test_gnn_predictions_use_shared_metric_correctly():
    adapter = GNNGraphAdapter(_index(), 53021, "S1")
    session = adapter.sessions("validation", eligible_only=True, max_sessions=1)[0]
    scores = {
        job_id: float(index)
        for index, job_id in enumerate(session.exposed_job_ids)
    }
    metric = RankingEvaluator(10).evaluate_session(session, scores)
    assert 0.0 <= metric["ndcg_at_10"] <= 1.0
    assert 0.0 <= metric["recall_at_10"] <= 1.0
