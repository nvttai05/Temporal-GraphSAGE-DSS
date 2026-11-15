from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from dss_stage6.data import DatasetIndex, HistoryStore, SessionLoader, SessionRecord
from dss_stage6.models import (
    ContentBasedFilter,
    ContentRepository,
    PopularityAsOfT,
    RandomBaseline,
    default_cbf_configs,
)


ROOT = Path(__file__).resolve().parents[1]


def _index() -> DatasetIndex:
    protocol = yaml.safe_load((ROOT / "configs" / "stage6_protocol.yaml").read_text(encoding="utf-8"))
    paths = yaml.safe_load((ROOT / "configs" / "paths.local.yaml").read_text(encoding="utf-8"))
    return DatasetIndex(paths["dataset_root"], protocol)


def test_random_seed_42_is_deterministic_on_exposed_set():
    index = _index()
    session = next(SessionLoader(index).iter_sessions(53021, "S0", "validation", 14, eligible_only=True))
    model = RandomBaseline(42)
    assert model.score(session) == model.score(session)
    assert set(model.score(session)) == set(session.exposed_job_ids)


def test_popularity_counts_only_prior_observed_apply_and_excludes_current_session():
    actions = pd.DataFrame(
        [
            ("a1", "e1", "old", "c", "job-00", "apply", "2027-01-01T09:00:00Z", "2027-01-01T09:00:00Z"),
            ("a2", "e2", "old", "c", "job-01", "apply", "2027-01-01T10:00:00Z", "2027-01-01T09:00:00Z"),
            ("a3", "e3", "old", "c", "job-02", "apply", "2027-01-01T09:00:00Z", "2027-01-01T10:00:00Z"),
            ("a4", "e4", "current", "c", "job-03", "apply", "2027-01-01T09:00:00Z", "2027-01-01T09:00:00Z"),
        ],
        columns=[
            "action_id", "exposure_id", "session_id", "candidate_id", "job_id",
            "action_type", "action_at", "observed_at",
        ],
    )
    candidates = pd.DataFrame(
        {
            "job_id": [f"job-{index:02d}" for index in range(12)],
            "label": [1] + [0] * 11,
            "new_job": [0] * 12,
            "changed_profile": [0] * 12,
            "low_history": [0] * 12,
        }
    )
    session = SessionRecord(
        53021, "S0", "validation", 14, "current", "c",
        pd.Timestamp("2027-01-01T10:00:00Z"), 12, 1, True, "scoreable", candidates,
    )
    scores = PopularityAsOfT(HistoryStore(actions, seed=53021, scenario="S0")).score(session)
    assert scores["job-00"] == 1.0
    assert scores["job-01"] == scores["job-02"] == scores["job-03"] == 0.0


def test_cbf_has_six_configs_and_validation_transform_does_not_refit():
    assert len(default_cbf_configs()) == 6
    index = _index()
    loader = SessionLoader(index)
    session = next(loader.iter_sessions(53021, "S0", "validation", 14, eligible_only=True))
    model = ContentBasedFilter(ContentRepository(index, 53021, "S0")).fit()
    state_hash = model.vocabulary_hash
    scores = model.score(session, default_cbf_configs()[0])
    assert model.vocabulary_hash == state_hash
    assert set(scores) == set(session.exposed_job_ids)
    assert all(0.0 <= score <= 1.0 for score in scores.values())
