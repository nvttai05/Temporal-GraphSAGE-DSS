from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import yaml

from dss_stage6.data import DatasetIndex
from dss_stage6.models import Stage6XGBoost, scale_pos_weight
from dss_stage6.stage6_3 import prepare_local_output_dirs
from dss_stage6.xgboost_features import F2_FEATURES, XGBoostFeaturePipeline


ROOT = Path(__file__).resolve().parents[1]


def _index() -> DatasetIndex:
    protocol = yaml.safe_load((ROOT / "configs" / "stage6_protocol.yaml").read_text(encoding="utf-8"))
    paths = yaml.safe_load((ROOT / "configs" / "paths.local.yaml").read_text(encoding="utf-8"))
    return DatasetIndex(paths["dataset_root"], protocol)


def test_locked_search_has_twenty_valid_deterministic_trials():
    config = yaml.safe_load((ROOT / "configs" / "stage6_3_xgboost.yaml").read_text(encoding="utf-8"))
    trials = config["trials"]
    assert len(trials) == 20
    assert len({trial["trial_id"] for trial in trials}) == 20
    assert all(200 <= trial["n_estimators"] <= 800 for trial in trials)
    assert all(trial["learning_rate"] in [0.03, 0.05, 0.10] for trial in trials)
    assert config["created_with_seed"] == 42


def test_static_and_time_feature_schemas_are_train_fit_and_leakage_safe():
    pipeline = XGBoostFeaturePipeline(_index(), 53021, "S0")
    train_sessions = pipeline.sessions("train", eligible_only=False, max_sessions=2)
    validation_sessions = pipeline.sessions("validation", eligible_only=True, max_sessions=1)
    train_raw = pipeline.raw_frame(train_sessions)
    validation_raw = pipeline.raw_frame(validation_sessions)
    pipeline.fit(train_raw)
    static_hash = pipeline.static_builder.state_hash
    time_hash = pipeline.time_builder.state_hash
    static = pipeline.transform(validation_raw, "static")
    timed = pipeline.transform(validation_raw, "time")
    assert pipeline.static_builder.state_hash == static_hash
    assert pipeline.time_builder.state_hash == time_hash
    assert not set(F2_FEATURES) & set(static.raw_features.columns)
    assert set(F2_FEATURES).issubset(timed.raw_features.columns)
    assert len(timed.transformed.columns) > len(static.transformed.columns)
    assert sum(row["future_or_current_violations"] for row in pipeline.leakage_audit) == 0


def test_xgboost_seed_42_is_deterministic_and_checkpoint_roundtrips(tmp_path):
    features = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    trial = {
        "trial_id": "unit",
        "n_estimators": 8,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "gamma": 0,
    }
    weight = scale_pos_weight(labels)
    first = Stage6XGBoost(trial, model_seed=42, class_weight=weight).fit(features, labels)
    second = Stage6XGBoost(trial, model_seed=42, class_weight=weight).fit(features, labels)
    first_scores = first.predict_scores(features)
    assert np.array_equal(first_scores, second.predict_scores(features))
    checkpoint = tmp_path / "model.json"
    first.save(checkpoint)
    assert np.array_equal(first_scores, Stage6XGBoost.load(checkpoint).predict_scores(features))


def test_local_runner_prepares_checkpoint_and_prediction_directories(tmp_path):
    checkpoint_dir, prediction_dir = prepare_local_output_dirs(tmp_path / "smoke")
    assert checkpoint_dir.is_dir()
    assert prediction_dir.is_dir()


def test_colab_notebook_enforces_commit_manifest_tests_and_resume():
    notebook = json.loads(
        (ROOT / "notebooks" / "STAGE6_3_COLAB_RUNNER.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )
    assert "drive.mount" in source
    assert "git', 'rev-parse', 'HEAD" in source
    assert "EXPECTED_MANIFEST_SHA256" in source
    assert "tests.check_returncode()" in source
    assert "--mode', 'colab-full'" in source
    assert "--resume" in source
