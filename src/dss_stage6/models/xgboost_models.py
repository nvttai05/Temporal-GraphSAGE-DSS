from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from xgboost import XGBClassifier


def scale_pos_weight(labels: np.ndarray) -> float:
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0:
        raise ValueError("Training labels contain no positive class")
    return float(negatives / positives)


class Stage6XGBoost:
    """Deterministic Stage 6 binary logistic XGBoost wrapper."""

    def __init__(self, trial: dict[str, Any], *, model_seed: int, class_weight: float):
        params = {key: value for key, value in trial.items() if key != "trial_id"}
        self.trial_id = str(trial["trial_id"])
        self.model = XGBClassifier(
            **params,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="logloss",
            random_state=int(model_seed),
            seed=int(model_seed),
            scale_pos_weight=float(class_weight),
            n_jobs=1,
            verbosity=0,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "Stage6XGBoost":
        self.model.fit(features, labels, verbose=False)
        return self

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(destination)

    @classmethod
    def load(cls, path: str | Path) -> "Stage6XGBoost":
        instance = cls.__new__(cls)
        instance.trial_id = "loaded"
        instance.model = XGBClassifier()
        instance.model.load_model(Path(path))
        return instance
