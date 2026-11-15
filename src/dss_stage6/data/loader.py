from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SESSION_COLUMNS = {
    "seed",
    "scenario",
    "delta_days",
    "session_id",
    "candidate_id",
    "prediction_at",
    "split",
    "valid_label_count",
    "positive_count",
    "top10_eligible",
    "reason",
}
POOL_COLUMNS = {
    "seed",
    "scenario",
    "delta_days",
    "split",
    "session_id",
    "candidate_id",
    "job_id",
    "shown_at",
    "label",
    "label_available_at",
    "new_job",
    "changed_profile",
    "low_history",
}
ACTION_COLUMNS = {
    "action_id",
    "exposure_id",
    "session_id",
    "candidate_id",
    "job_id",
    "action_type",
    "action_at",
    "observed_at",
}


def _require_columns(frame: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Schema mismatch in {source}: missing columns {missing}")


def filter_actions_as_of(
    actions: pd.DataFrame,
    prediction_at: object,
    *,
    current_session_id: str | None = None,
) -> pd.DataFrame:
    """Return only graph edges observable strictly before prediction time t.

    Both event time and observation time must be before t. The current session is
    excluded even if a malformed fixture gives it an earlier timestamp.
    """

    _require_columns(actions, ACTION_COLUMNS, Path("<actions frame>"))
    prediction = pd.Timestamp(prediction_at)
    action_at = pd.to_datetime(actions["action_at"], utc=True, errors="raise")
    observed_at = pd.to_datetime(actions["observed_at"], utc=True, errors="raise")
    mask = (action_at < prediction) & (observed_at < prediction)
    if current_session_id is not None:
        mask &= actions["session_id"].astype(str) != str(current_session_id)
    return actions.loc[mask].copy().reset_index(drop=True)


@dataclass(frozen=True)
class FrozenTemporalDataset:
    """Read-only loader for the frozen MEDAJob temporal v0.3 handoff."""

    dataset_root: Path

    def __init__(self, dataset_root: str | Path):
        object.__setattr__(self, "dataset_root", Path(dataset_root).resolve())

    @property
    def candidate_root(self) -> Path:
        return self.dataset_root / "data" / "candidate55" / "v0_3"

    @property
    def log_root(self) -> Path:
        return self.dataset_root / "data" / "candidate55" / "v0_3_logs"

    def _read_csv(self, path: Path, required: Iterable[str]) -> pd.DataFrame:
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        _require_columns(frame, required, path)
        return frame

    def session_splits(
        self,
        *,
        seed: int | None = None,
        scenario: str | None = None,
        split: str | None = None,
        delta_days: int | None = None,
    ) -> pd.DataFrame:
        frame = self._read_csv(self.candidate_root / "session_splits.csv", SESSION_COLUMNS)
        for column, value in (
            ("seed", seed),
            ("scenario", scenario),
            ("split", split),
            ("delta_days", delta_days),
        ):
            if value is not None:
                frame = frame.loc[frame[column] == value]
        return frame.copy().reset_index(drop=True)

    def benchmark_pool(
        self,
        *,
        seed: int | None = None,
        scenario: str | None = None,
        split: str | None = None,
        delta_days: int | None = None,
    ) -> pd.DataFrame:
        frame = self._read_csv(self.candidate_root / "benchmark_pool.csv", POOL_COLUMNS)
        for column, value in (
            ("seed", seed),
            ("scenario", scenario),
            ("split", split),
            ("delta_days", delta_days),
        ):
            if value is not None:
                frame = frame.loc[frame[column] == value]
        return frame.copy().reset_index(drop=True)

    def cell_path(self, seed: int, scenario: str, filename: str) -> Path:
        return self.log_root / f"seed_{seed}" / scenario / filename

    def actions(self, seed: int, scenario: str) -> pd.DataFrame:
        return self._read_csv(self.cell_path(seed, scenario, "actions.csv"), ACTION_COLUMNS)

    def graph_snapshot(
        self,
        seed: int,
        scenario: str,
        prediction_at: object,
        *,
        current_session_id: str | None = None,
    ) -> pd.DataFrame:
        return filter_actions_as_of(
            self.actions(seed, scenario),
            prediction_at,
            current_session_id=current_session_id,
        )
