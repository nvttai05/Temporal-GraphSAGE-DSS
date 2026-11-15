from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dss_stage6.data import DatasetIndex, HistoryStore, SessionRecord
from dss_stage6.features import FeatureBuilder


class RandomBaseline:
    """Seeded sanity-only scorer; not a trainable model."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def score(self, session: SessionRecord) -> dict[str, float]:
        return {
            job_id: int(
                hashlib.sha256(
                    f"{self.seed}|{session.seed}|{session.scenario}|{session.session_id}|{job_id}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:16],
                16,
            )
            / float(16**16 - 1)
            for job_id in session.exposed_job_ids
        }


class PopularityAsOfT:
    """Count observed apply actions available strictly before each session time."""

    def __init__(self, history: HistoryStore):
        self.history = history

    def score(self, session: SessionRecord) -> dict[str, float]:
        return self.score_from_snapshot(session, self.observable_actions(session))

    def observable_actions(self, session: SessionRecord) -> pd.DataFrame:
        return self.history.as_of(
            session.prediction_at, current_session_id=session.session_id
        )

    @staticmethod
    def score_from_snapshot(
        session: SessionRecord, snapshot: pd.DataFrame
    ) -> dict[str, float]:
        counts = snapshot.loc[snapshot["action_type"] == "apply", "job_id"].value_counts()
        return {job_id: float(counts.get(job_id, 0)) for job_id in session.exposed_job_ids}


@dataclass(frozen=True)
class CBFConfig:
    config_id: str
    field_weights: dict[str, float]
    representation: str = "binary"
    complexity: int = 0


def default_cbf_configs() -> tuple[CBFConfig, ...]:
    return (
        CBFConfig("cbf_01_balanced", {"skills": 1, "occupation": 1, "location": 1, "experience": 1}, complexity=4),
        CBFConfig("cbf_02_skills2", {"skills": 2, "occupation": 1, "location": 1, "experience": 1}, complexity=5),
        CBFConfig("cbf_03_occupation2", {"skills": 1, "occupation": 2, "location": 1, "experience": 1}, complexity=5),
        CBFConfig("cbf_04_geo_experience2", {"skills": 1, "occupation": 1, "location": 2, "experience": 2}, complexity=6),
        CBFConfig("cbf_05_skills3_occupation2", {"skills": 3, "occupation": 2, "location": 1, "experience": 1}, complexity=7),
        CBFConfig("cbf_06_compact", {"skills": 1, "occupation": 1, "location": 0, "experience": 0}, complexity=2),
    )


class ContentRepository:
    """As-of-t F0 content access without free text or salary."""

    fields = ("skills", "occupation", "location", "experience")

    def __init__(self, index: DatasetIndex, seed: int, scenario: str):
        self.index = index
        self.seed = int(seed)
        self.scenario = str(scenario)
        root = index.dataset.dataset_root
        self.jobs = pd.read_csv(root / "data" / "processed" / "jobs_clean.csv").rename(
            columns={"source_id": "job_id"}
        )
        self.jobs = self.jobs.set_index("job_id", drop=False)
        self.candidates = pd.read_csv(
            root / "data" / "processed" / "candidate_profiles_clean.csv"
        ).set_index("candidate_id", drop=False)
        self.versions = pd.read_csv(
            index.dataset.cell_path(seed, scenario, "profile_versions.csv")
        )
        self.versions["valid_from"] = pd.to_datetime(self.versions["valid_from"], utc=True)
        self.versions["valid_to"] = pd.to_datetime(
            self.versions["valid_to"], utc=True, errors="coerce"
        )

    def candidate_at(self, candidate_id: str, prediction_at: object, split: str) -> dict[str, str]:
        prediction = pd.Timestamp(prediction_at)
        if prediction.tzinfo is None:
            prediction = prediction.tz_localize("UTC")
        else:
            prediction = prediction.tz_convert("UTC")
        versions = self.versions.loc[
            (self.versions["candidate_id"] == candidate_id)
            & (self.versions["valid_from"] <= prediction)
            & (self.versions["valid_to"].isna() | (self.versions["valid_to"] > prediction))
        ].sort_values(["valid_from", "profile_version_id"])
        if versions.empty:
            raise KeyError(f"No profile version as of {prediction}: {candidate_id}")
        version = versions.iloc[-1]
        if candidate_id in self.candidates.index:
            base = self.candidates.loc[candidate_id]
            occupation = str(base["occupation"])
            location = str(base["location"])
            experience = str(base["experience"])
        else:
            occupation = location = experience = "unknown"
        return {
            "split": split,
            "skills": str(version["skills"]),
            "occupation": occupation,
            "location": location,
            "experience": experience,
        }

    def jobs_frame(self, job_ids: Iterable[str], split: str) -> pd.DataFrame:
        ordered = list(job_ids)
        missing = sorted(set(ordered) - set(self.jobs.index))
        if missing:
            raise KeyError(f"Jobs missing F0 content: {missing[:5]}")
        result = self.jobs.loc[ordered, list(self.fields)].copy().reset_index(drop=True)
        result.insert(0, "split", split)
        return result

    def training_content(self) -> pd.DataFrame:
        delta = int(self.index.protocol["dataset"]["primary_label_window_days"])
        sessions = self.index.sessions.loc[
            (self.index.sessions["seed"] == self.seed)
            & (self.index.sessions["scenario"] == self.scenario)
            & (self.index.sessions["split"] == "train")
            & (self.index.sessions["delta_days"] == delta)
        ]
        candidate_rows = [
            self.candidate_at(str(row.candidate_id), row.prediction_at, "train")
            for row in sessions.itertuples(index=False)
        ]
        train_jobs = self.index.pool.loc[
            (self.index.pool["seed"] == self.seed)
            & (self.index.pool["scenario"] == self.scenario)
            & (self.index.pool["split"] == "train")
            & (self.index.pool["delta_days"] == delta),
            "job_id",
        ].drop_duplicates()
        jobs = self.jobs_frame(train_jobs.astype(str), "train")
        return pd.concat([pd.DataFrame(candidate_rows), jobs], ignore_index=True)


class ContentBasedFilter:
    """Train-fitted F0 binary vector cosine scorer with field weights."""

    def __init__(self, repository: ContentRepository):
        self.repository = repository
        self.features = FeatureBuilder(
            categorical_columns=["occupation", "location", "experience"],
            multivalue_columns=["skills"],
        )

    def fit(self) -> "ContentBasedFilter":
        self.features.fit(self.repository.training_content())
        return self

    @property
    def vocabulary_hash(self) -> str:
        return self.features.state_hash

    def _weights(self, config: CBFConfig) -> np.ndarray:
        weights = []
        for column in self.features.state.output_columns:
            field = column.split("==", 1)[0].split("~=", 1)[0]
            weights.append(math_sqrt_nonnegative(config.field_weights.get(field, 0.0)))
        return np.asarray(weights, dtype=float)

    def score(self, session: SessionRecord, config: CBFConfig) -> dict[str, float]:
        if not self.features.fitted:
            raise RuntimeError("CBF must fit its vocabulary on train before scoring")
        candidate_frame = pd.DataFrame(
            [
                self.repository.candidate_at(
                    session.candidate_id, session.prediction_at, session.split
                )
            ]
        )
        job_frame = self.repository.jobs_frame(session.exposed_job_ids, session.split)
        state_hash = self.features.state_hash
        candidate_vector = self.features.transform(candidate_frame).to_numpy()[0]
        job_vectors = self.features.transform(job_frame).to_numpy()
        if self.features.state_hash != state_hash:
            raise AssertionError("Validation transform changed fitted vocabulary")
        weights = self._weights(config)
        candidate_vector = candidate_vector * weights
        job_vectors = job_vectors * weights
        candidate_norm = float(np.linalg.norm(candidate_vector))
        job_norms = np.linalg.norm(job_vectors, axis=1)
        numerators = job_vectors @ candidate_vector
        denominators = job_norms * candidate_norm
        similarities = np.divide(
            numerators,
            denominators,
            out=np.zeros_like(numerators, dtype=float),
            where=denominators > 0,
        )
        return {
            job_id: float(score)
            for job_id, score in zip(session.exposed_job_ids, similarities, strict=True)
        }


def math_sqrt_nonnegative(value: float) -> float:
    if value < 0:
        raise ValueError("CBF field weights must be non-negative")
    return float(np.sqrt(value))
