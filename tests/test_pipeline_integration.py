from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from dss_stage6.data import DatasetIndex, SessionLoader
from dss_stage6.smoke import run_six_cell_smoke, smoke_digest


ROOT = Path(__file__).resolve().parents[1]


def _index() -> DatasetIndex:
    protocol = yaml.safe_load((ROOT / "configs" / "stage6_protocol.yaml").read_text(encoding="utf-8"))
    paths = yaml.safe_load((ROOT / "configs" / "paths.local.yaml").read_text(encoding="utf-8"))
    return DatasetIndex(paths["dataset_root"], protocol)


def test_dataset_index_and_session_loader_cover_protocol_lattice():
    index = _index()
    assert len(index.data_cells) == 6
    assert len(index.slices) == 36
    assert set(index.to_frame()["split"]) == {"train", "validation", "test"}
    assert set(index.to_frame()["delta_days"]) == {7, 14}
    loader = SessionLoader(index)
    for cell in index.data_cells:
        session = next(loader.iter_sessions(cell.seed, cell.scenario, "validation", 14, eligible_only=True))
        assert len(session.candidates) >= 10
        assert session.positive_count >= 1
        assert session.candidates["label"].isin([0, 1]).all()


def test_six_cell_smoke_is_complete_and_deterministic():
    index = _index()
    first = run_six_cell_smoke(index)
    second = run_six_cell_smoke(index)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 6
    assert first["future_edge_violations"].sum() == 0
    assert first["feature_state_unchanged"].all()
    assert smoke_digest(first) == smoke_digest(second)
