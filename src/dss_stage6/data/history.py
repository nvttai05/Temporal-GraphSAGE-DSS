from __future__ import annotations

from pathlib import Path

import pandas as pd

from .loader import ACTION_COLUMNS, FrozenTemporalDataset, _require_columns


class HistoryStore:
    """Cell-local rolling history with strict dual-time as-of-t filtering."""

    def __init__(self, actions: pd.DataFrame, *, seed: int, scenario: str):
        _require_columns(actions, ACTION_COLUMNS, Path("<history actions>"))
        self.seed = int(seed)
        self.scenario = str(scenario)
        self._actions = actions.copy()
        self._actions["action_at"] = pd.to_datetime(self._actions["action_at"], utc=True)
        self._actions["observed_at"] = pd.to_datetime(self._actions["observed_at"], utc=True)
        self._actions = self._actions.sort_values(
            ["action_at", "observed_at", "action_id"], kind="stable"
        ).reset_index(drop=True)

    @classmethod
    def from_dataset(
        cls, dataset: FrozenTemporalDataset, seed: int, scenario: str
    ) -> "HistoryStore":
        return cls(dataset.actions(seed, scenario), seed=seed, scenario=scenario)

    def as_of(
        self, prediction_at: object, *, current_session_id: str | None = None
    ) -> pd.DataFrame:
        prediction = pd.Timestamp(prediction_at)
        if prediction.tzinfo is None:
            prediction = prediction.tz_localize("UTC")
        else:
            prediction = prediction.tz_convert("UTC")
        mask = (
            (self._actions["action_at"] < prediction)
            & (self._actions["observed_at"] < prediction)
        )
        if current_session_id is not None:
            mask &= self._actions["session_id"].astype(str) != str(current_session_id)
        return self._actions.loc[mask].copy().reset_index(drop=True)
