from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from .index import DatasetIndex


@dataclass(frozen=True)
class SessionRecord:
    seed: int
    scenario: str
    split: str
    delta_days: int
    session_id: str
    candidate_id: str
    prediction_at: pd.Timestamp
    valid_label_count: int
    positive_count: int
    top10_eligible: bool
    reason: str
    candidates: pd.DataFrame

    @property
    def exposed_job_ids(self) -> tuple[str, ...]:
        return tuple(self.candidates["job_id"].astype(str))

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(self.candidates["label"].astype(int))

    @property
    def cohorts(self) -> pd.DataFrame:
        return self.candidates[["job_id", "new_job", "changed_profile", "low_history"]].copy()


class SessionLoader:
    """Shared exposed-set session loader backed only by the frozen benchmark pool."""

    def __init__(self, index: DatasetIndex):
        self.index = index
        self._key_columns = ["seed", "scenario", "split", "delta_days", "session_id"]
        self._session_lookup = index.sessions.set_index(self._key_columns, drop=False)
        self._candidate_groups = index.pool.groupby(
            self._key_columns, sort=False
        ).indices

    def load(
        self,
        seed: int,
        scenario: str,
        split: str,
        delta_days: int,
        session_id: str,
    ) -> SessionRecord:
        key = (seed, scenario, split, delta_days, session_id)
        try:
            row = self._session_lookup.loc[key]
        except KeyError as error:
            raise KeyError(f"Session is not indexed: {session_id}") from error
        if isinstance(row, pd.DataFrame):
            raise KeyError(f"Expected one indexed session, found {len(row)}: {session_id}")
        positions = self._candidate_groups.get(key, [])
        candidates = self.index.pool.iloc[list(positions)].copy()
        candidates = candidates.sort_values("job_id", kind="stable").reset_index(drop=True)
        if candidates["job_id"].duplicated().any():
            raise ValueError(f"Duplicate exposed job in {session_id}")
        if not candidates["label"].isin([0, 1]).all():
            raise ValueError(f"Non-binary or censored label entered exposed set: {session_id}")
        if len(candidates) != int(row["valid_label_count"]):
            raise ValueError(f"Candidate count disagrees with session index: {session_id}")
        if int(candidates["label"].sum()) != int(row["positive_count"]):
            raise ValueError(f"Positive count disagrees with session index: {session_id}")
        return SessionRecord(
            seed=int(seed),
            scenario=str(scenario),
            split=str(split),
            delta_days=int(delta_days),
            session_id=str(session_id),
            candidate_id=str(row["candidate_id"]),
            prediction_at=pd.Timestamp(row["prediction_at"]),
            valid_label_count=int(row["valid_label_count"]),
            positive_count=int(row["positive_count"]),
            top10_eligible=bool(row["top10_eligible"]),
            reason=str(row["reason"]),
            candidates=candidates,
        )

    def iter_sessions(
        self,
        seed: int,
        scenario: str,
        split: str,
        delta_days: int,
        *,
        eligible_only: bool = False,
    ) -> Iterator[SessionRecord]:
        metadata = self.index.slice_frame(seed, scenario, split, delta_days)
        if eligible_only:
            metadata = metadata.loc[metadata["top10_eligible"] == 1]
        for session_id in metadata.sort_values(["prediction_at", "session_id"])["session_id"]:
            yield self.load(seed, scenario, split, delta_days, str(session_id))
