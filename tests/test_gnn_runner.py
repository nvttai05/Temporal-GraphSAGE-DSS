from __future__ import annotations

import pytest

from dss_stage6.gnn.runner import filter_model_variants


VARIANTS = [
    ("M0_static_graphsage", False, 0.0),
    ("M1_temporal_graphsage_hl7", True, 7.0),
    ("M1_temporal_graphsage_hl14", True, 14.0),
    ("M1_temporal_graphsage_hl30", True, 30.0),
]


def test_model_filter_defaults_to_static_and_all_temporal_variants():
    assert filter_model_variants(VARIANTS, ["static", "temporal"]) == VARIANTS


def test_model_filter_static_only():
    assert filter_model_variants(VARIANTS, ["static"]) == [VARIANTS[0]]


def test_model_filter_temporal_only_keeps_all_requested_half_lives():
    assert filter_model_variants(VARIANTS, ["temporal"]) == VARIANTS[1:]


def test_model_filter_rejects_unknown_or_empty_selection():
    with pytest.raises(ValueError, match="Unknown"):
        filter_model_variants(VARIANTS, ["graph_transformer"])
    with pytest.raises(ValueError, match="At least one"):
        filter_model_variants(VARIANTS, [])
