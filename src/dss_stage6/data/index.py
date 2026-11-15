from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .loader import FrozenTemporalDataset


@dataclass(frozen=True, order=True)
class DataCell:
    seed: int
    scenario: str


@dataclass(frozen=True, order=True)
class DatasetSlice:
    seed: int
    scenario: str
    split: str
    delta_days: int
    session_count: int
    candidate_count: int


class DatasetIndex:
    """Validated index over the frozen seed/scenario/split/window lattice."""

    def __init__(self, dataset_root: str | Path, protocol: dict[str, Any]):
        self.dataset = FrozenTemporalDataset(dataset_root)
        self.protocol = protocol
        self.sessions = self.dataset.session_splits()
        self.pool = self.dataset.benchmark_pool()
        self._validate_domains()
        session_counts = self.sessions.groupby(
            ["seed", "scenario", "split", "delta_days"]
        ).size()
        candidate_counts = self.pool.groupby(
            ["seed", "scenario", "split", "delta_days"]
        ).size()
        self.slices = tuple(
            DatasetSlice(
                seed=seed,
                scenario=scenario,
                split=split,
                delta_days=delta,
                session_count=int(session_counts.get((seed, scenario, split, delta), 0)),
                candidate_count=int(candidate_counts.get((seed, scenario, split, delta), 0)),
            )
            for seed in self.expected_seeds
            for scenario in self.expected_scenarios
            for split in self.expected_splits
            for delta in self.expected_deltas
        )
        empty = [entry for entry in self.slices if entry.session_count == 0 or entry.candidate_count == 0]
        if empty:
            raise ValueError(f"Dataset index contains empty slices: {empty}")

    @property
    def expected_seeds(self) -> tuple[int, ...]:
        return tuple(self.protocol["dataset"]["data_seeds"])

    @property
    def expected_scenarios(self) -> tuple[str, ...]:
        return tuple(self.protocol["dataset"]["scenarios"])

    @property
    def expected_splits(self) -> tuple[str, ...]:
        return tuple(self.protocol["dataset"]["splits"])

    @property
    def expected_deltas(self) -> tuple[int, ...]:
        return (
            int(self.protocol["dataset"]["sensitivity_label_window_days"]),
            int(self.protocol["dataset"]["primary_label_window_days"]),
        )

    @property
    def data_cells(self) -> tuple[DataCell, ...]:
        return tuple(
            DataCell(seed, scenario)
            for seed in self.expected_seeds
            for scenario in self.expected_scenarios
        )

    def _validate_domains(self) -> None:
        expected = {
            "seed": set(self.expected_seeds),
            "scenario": set(self.expected_scenarios),
            "split": set(self.expected_splits),
            "delta_days": set(self.expected_deltas),
        }
        for name, values in expected.items():
            actual = set(self.sessions[name].unique())
            if actual != values:
                raise ValueError(f"Unexpected {name} domain: actual={actual}, expected={values}")
        pool_lattice = set(
            map(
                tuple,
                self.pool[["seed", "scenario", "split", "delta_days"]]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        expected_lattice = {
            (seed, scenario, split, delta)
            for seed in self.expected_seeds
            for scenario in self.expected_scenarios
            for split in self.expected_splits
            for delta in self.expected_deltas
        }
        if pool_lattice != expected_lattice:
            raise ValueError("benchmark_pool does not cover the protocol lattice")

    def slice_frame(
        self, seed: int, scenario: str, split: str, delta_days: int
    ) -> pd.DataFrame:
        return self.sessions.loc[
            (self.sessions["seed"] == seed)
            & (self.sessions["scenario"] == scenario)
            & (self.sessions["split"] == split)
            & (self.sessions["delta_days"] == delta_days)
        ].copy()

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([entry.__dict__ for entry in self.slices])
