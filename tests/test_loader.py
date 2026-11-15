from __future__ import annotations

import pandas as pd

from dss_stage6.data import filter_actions_as_of


def _actions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("prior", "e1", "old", "c", "j1", "view", "2027-01-01T09:59:59Z", "2027-01-01T09:59:59Z"),
            ("action-boundary", "e2", "old", "c", "j2", "view", "2027-01-01T10:00:00Z", "2027-01-01T09:59:59Z"),
            ("observed-boundary", "e3", "old", "c", "j3", "view", "2027-01-01T09:59:59Z", "2027-01-01T10:00:00Z"),
            ("future-action", "e4", "old", "c", "j4", "view", "2027-01-01T10:00:01Z", "2027-01-01T10:00:01Z"),
            ("future-observation", "e5", "old", "c", "j5", "view", "2027-01-01T09:00:00Z", "2027-01-02T00:00:00Z"),
            ("current-session", "e6", "current", "c", "j6", "view", "2027-01-01T09:00:00Z", "2027-01-01T09:00:00Z"),
        ],
        columns=[
            "action_id", "exposure_id", "session_id", "candidate_id", "job_id",
            "action_type", "action_at", "observed_at",
        ],
    )


def test_as_of_t_excludes_boundaries_future_and_current_session():
    snapshot = filter_actions_as_of(
        _actions(), "2027-01-01T10:00:00Z", current_session_id="current"
    )
    assert snapshot["action_id"].tolist() == ["prior"]


def test_as_of_t_fixture_is_deterministic():
    first = filter_actions_as_of(_actions(), "2027-01-01T10:00:00Z", current_session_id="current")
    second = filter_actions_as_of(_actions(), "2027-01-01T10:00:00Z", current_session_id="current")
    pd.testing.assert_frame_equal(first, second)
