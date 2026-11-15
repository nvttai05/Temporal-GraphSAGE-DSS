from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from dss_stage6.data import DatasetIndex, HistoryStore, SessionLoader, SessionRecord
from dss_stage6.features import FeatureBuilder
from dss_stage6.models import CBFConfig, ContentBasedFilter, ContentRepository, default_cbf_configs


F0_FEATURES = (
    "cbf_score_locked",
    "skill_overlap_count",
    "skill_jaccard",
    "skill_binary_cosine",
    "occupation_match",
    "location_match",
    "experience_match",
    "experience_distance",
)

F1_FEATURES = (
    "candidate_cumulative_impression_count",
    "candidate_cumulative_view_count",
    "candidate_cumulative_save_count",
    "candidate_cumulative_apply_count",
    "job_cumulative_impression_count",
    "job_cumulative_view_count",
    "job_cumulative_save_count",
    "job_cumulative_apply_count",
    "candidate_job_prior_interaction_count",
    "candidate_job_prior_view_count",
    "candidate_job_prior_save_count",
    "candidate_job_prior_apply_count",
    "candidate_job_prior_interaction_flag",
    "candidate_degree",
    "job_degree",
)

RECENCY_BASES = (
    "candidate_last_view_days",
    "candidate_last_save_days",
    "candidate_last_apply_days",
    "job_last_interaction_days",
    "job_last_apply_days",
)

WINDOW_FEATURES = tuple(
    f"{entity}_{kind}_{window}d"
    for window in (7, 14, 30)
    for entity in ("candidate", "job", "candidate_job")
    for kind in ("interaction_count", "apply_count")
)

F2_FEATURES = (
    "job_age_days",
    "profile_version_age_days",
    *RECENCY_BASES,
    *(f"{name}_missing" for name in RECENCY_BASES),
    *WINDOW_FEATURES,
    "candidate_interaction_exp_decay",
    "job_interaction_exp_decay",
    "candidate_job_interaction_exp_decay",
)


@dataclass(frozen=True)
class BuiltFeatures:
    identifiers: pd.DataFrame
    raw_features: pd.DataFrame
    transformed: pd.DataFrame
    labels: np.ndarray


def _schema_hash(columns: Iterable[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(columns), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class XGBoostFeaturePipeline:
    """Shared train-fitted F0/F1/F2 feature pipeline for one data cell."""

    EXPERIENCE_ORDINAL = {
        "unknown": 0.0,
        "entry": 0.0,
        "one_to_four_years": 1.0,
        "mid_unspecified_years": 1.5,
        "senior_or_5plus": 2.0,
    }

    def __init__(
        self,
        index: DatasetIndex,
        seed: int,
        scenario: str,
        *,
        recency_sentinel_days: float = 100000.0,
        decay_half_life_days: float = 14.0,
    ):
        self.index = index
        self.seed = int(seed)
        self.scenario = str(scenario)
        self.loader = SessionLoader(index)
        self.history = HistoryStore.from_dataset(index.dataset, seed, scenario)
        self.repository = ContentRepository(index, seed, scenario)
        self.cbf = ContentBasedFilter(self.repository).fit()
        self.cbf_config = next(
            config
            for config in default_cbf_configs()
            if config.config_id == "cbf_05_skills3_occupation2"
        )
        self.recency_sentinel_days = float(recency_sentinel_days)
        self.decay_half_life_days = float(decay_half_life_days)
        self.impressions = pd.read_csv(
            index.dataset.cell_path(seed, scenario, "impressions.csv")
        )
        self.impressions["shown_at"] = pd.to_datetime(self.impressions["shown_at"], utc=True)
        self.lifecycle = pd.read_csv(
            index.dataset.cell_path(seed, scenario, "job_lifecycle.csv")
        ).set_index("job_id", drop=False)
        self.lifecycle["open_at"] = pd.to_datetime(self.lifecycle["open_at"], utc=True)
        self.static_builder = FeatureBuilder(numeric_columns=[*F0_FEATURES, *F1_FEATURES])
        self.time_builder = FeatureBuilder(
            numeric_columns=[*F0_FEATURES, *F1_FEATURES, *F2_FEATURES]
        )
        self._fitted = False
        self.leakage_audit: list[dict[str, Any]] = []

    @property
    def static_columns(self) -> tuple[str, ...]:
        return (*F0_FEATURES, *F1_FEATURES)

    @property
    def time_columns(self) -> tuple[str, ...]:
        return (*F0_FEATURES, *F1_FEATURES, *F2_FEATURES)

    def sessions(
        self,
        split: str,
        *,
        eligible_only: bool,
        max_sessions: int | None = None,
    ) -> list[SessionRecord]:
        result: list[SessionRecord] = []
        for session in self.loader.iter_sessions(
            self.seed, self.scenario, split, 14, eligible_only=eligible_only
        ):
            if len(session.candidates) == 0:
                continue
            result.append(session)
            if max_sessions is not None and len(result) >= max_sessions:
                break
        return result

    @staticmethod
    def _skills(value: str) -> set[str]:
        return {token for token in str(value).split("|") if token}

    def _content_features(
        self, session: SessionRecord, cbf_scores: dict[str, float]
    ) -> dict[str, dict[str, float]]:
        candidate = self.repository.candidate_at(
            session.candidate_id, session.prediction_at, session.split
        )
        result: dict[str, dict[str, float]] = {}
        candidate_skills = self._skills(candidate["skills"])
        candidate_exp = self.EXPERIENCE_ORDINAL.get(candidate["experience"], 0.0)
        for job_id in session.exposed_job_ids:
            job = self.repository.jobs.loc[job_id]
            job_skills = self._skills(str(job["skills"]))
            overlap = len(candidate_skills & job_skills)
            union = len(candidate_skills | job_skills)
            denom = math_sqrt(len(candidate_skills) * len(job_skills))
            job_exp = self.EXPERIENCE_ORDINAL.get(str(job["experience"]), 0.0)
            result[job_id] = {
                "cbf_score_locked": float(cbf_scores[job_id]),
                "skill_overlap_count": float(overlap),
                "skill_jaccard": float(overlap / union) if union else 0.0,
                "skill_binary_cosine": float(overlap / denom) if denom else 0.0,
                "occupation_match": float(candidate["occupation"] == str(job["occupation"])),
                "location_match": float(candidate["location"] == str(job["location"])),
                "experience_match": float(candidate["experience"] == str(job["experience"])),
                "experience_distance": abs(candidate_exp - job_exp),
            }
        return result

    @staticmethod
    def _action_count(frame: pd.DataFrame, action_type: str) -> float:
        return float((frame["action_type"] == action_type).sum())

    def _recency(
        self, frame: pd.DataFrame, prediction: pd.Timestamp, action_type: str | None
    ) -> tuple[float, float]:
        subset = frame if action_type is None else frame.loc[frame["action_type"] == action_type]
        if subset.empty:
            return self.recency_sentinel_days, 1.0
        days = (prediction - subset["action_at"].max()).total_seconds() / 86400.0
        return float(days), 0.0

    def _window_count(
        self, frame: pd.DataFrame, prediction: pd.Timestamp, days: int, apply_only: bool
    ) -> float:
        cutoff = prediction - pd.Timedelta(days=days)
        subset = frame.loc[frame["action_at"] >= cutoff]
        if apply_only:
            subset = subset.loc[subset["action_type"] == "apply"]
        return float(len(subset))

    def _decay_count(self, frame: pd.DataFrame, prediction: pd.Timestamp) -> float:
        if frame.empty:
            return 0.0
        ages = (prediction - frame["action_at"]).dt.total_seconds() / 86400.0
        return float(np.power(0.5, ages / self.decay_half_life_days).sum())

    def _session_rows(self, session: SessionRecord) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prediction = pd.Timestamp(session.prediction_at)
        if prediction.tzinfo is None:
            prediction = prediction.tz_localize("UTC")
        else:
            prediction = prediction.tz_convert("UTC")
        actions = self.history.as_of(prediction, current_session_id=session.session_id)
        impressions = self.impressions.loc[
            (self.impressions["shown_at"] < prediction)
            & (self.impressions["session_id"].astype(str) != session.session_id)
        ]
        violations = int(
            (
                (actions["action_at"] >= prediction)
                | (actions["observed_at"] >= prediction)
                | (actions["session_id"].astype(str) == session.session_id)
            ).sum()
        ) + int(
            (
                (impressions["shown_at"] >= prediction)
                | (impressions["session_id"].astype(str) == session.session_id)
            ).sum()
        )
        cbf_scores = self.cbf.score(session, self.cbf_config)
        content = self._content_features(session, cbf_scores)
        candidate_actions = actions.loc[actions["candidate_id"] == session.candidate_id]
        candidate_impressions = impressions.loc[
            impressions["candidate_id"] == session.candidate_id
        ]
        candidate_degree = float(candidate_actions["job_id"].nunique())
        profile_versions = self.repository.versions.loc[
            (self.repository.versions["candidate_id"] == session.candidate_id)
            & (self.repository.versions["valid_from"] <= prediction)
            & (
                self.repository.versions["valid_to"].isna()
                | (self.repository.versions["valid_to"] > prediction)
            )
        ]
        if profile_versions.empty:
            raise KeyError(f"No profile version for {session.session_id}")
        profile_valid_from = profile_versions["valid_from"].max()
        rows: list[dict[str, Any]] = []
        for candidate_row in session.candidates.itertuples(index=False):
            job_id = str(candidate_row.job_id)
            job_actions = actions.loc[actions["job_id"] == job_id]
            job_impressions = impressions.loc[impressions["job_id"] == job_id]
            pair_actions = candidate_actions.loc[candidate_actions["job_id"] == job_id]
            pair_impressions = candidate_impressions.loc[
                candidate_impressions["job_id"] == job_id
            ]
            row: dict[str, Any] = {
                "split": session.split,
                "session_id": session.session_id,
                "candidate_id": session.candidate_id,
                "job_id": job_id,
                "label": int(candidate_row.label),
                **content[job_id],
                "candidate_cumulative_impression_count": float(len(candidate_impressions)),
                "candidate_cumulative_view_count": self._action_count(candidate_actions, "view"),
                "candidate_cumulative_save_count": self._action_count(candidate_actions, "save"),
                "candidate_cumulative_apply_count": self._action_count(candidate_actions, "apply"),
                "job_cumulative_impression_count": float(len(job_impressions)),
                "job_cumulative_view_count": self._action_count(job_actions, "view"),
                "job_cumulative_save_count": self._action_count(job_actions, "save"),
                "job_cumulative_apply_count": self._action_count(job_actions, "apply"),
                "candidate_job_prior_interaction_count": float(len(pair_actions)),
                "candidate_job_prior_view_count": self._action_count(pair_actions, "view"),
                "candidate_job_prior_save_count": self._action_count(pair_actions, "save"),
                "candidate_job_prior_apply_count": self._action_count(pair_actions, "apply"),
                "candidate_job_prior_interaction_flag": float(
                    len(pair_actions) > 0 or len(pair_impressions) > 0
                ),
                "candidate_degree": candidate_degree,
                "job_degree": float(job_actions["candidate_id"].nunique()),
                "job_age_days": float(
                    (prediction - self.lifecycle.loc[job_id, "open_at"]).total_seconds()
                    / 86400.0
                ),
                "profile_version_age_days": float(
                    (prediction - profile_valid_from).total_seconds() / 86400.0
                ),
            }
            for feature, frame, action_type in (
                ("candidate_last_view_days", candidate_actions, "view"),
                ("candidate_last_save_days", candidate_actions, "save"),
                ("candidate_last_apply_days", candidate_actions, "apply"),
                ("job_last_interaction_days", job_actions, None),
                ("job_last_apply_days", job_actions, "apply"),
            ):
                value, missing = self._recency(frame, prediction, action_type)
                row[feature] = value
                row[f"{feature}_missing"] = missing
            for window in (7, 14, 30):
                for prefix, frame in (
                    ("candidate", candidate_actions),
                    ("job", job_actions),
                    ("candidate_job", pair_actions),
                ):
                    row[f"{prefix}_interaction_count_{window}d"] = self._window_count(
                        frame, prediction, window, False
                    )
                    row[f"{prefix}_apply_count_{window}d"] = self._window_count(
                        frame, prediction, window, True
                    )
            row["candidate_interaction_exp_decay"] = self._decay_count(
                candidate_actions, prediction
            )
            row["job_interaction_exp_decay"] = self._decay_count(job_actions, prediction)
            row["candidate_job_interaction_exp_decay"] = self._decay_count(
                pair_actions, prediction
            )
            rows.append(row)
        audit = {
            "seed": self.seed,
            "scenario": self.scenario,
            "split": session.split,
            "session_id": session.session_id,
            "prediction_at": prediction.isoformat(),
            "max_action_at": actions["action_at"].max().isoformat() if not actions.empty else "",
            "max_observed_at": actions["observed_at"].max().isoformat() if not actions.empty else "",
            "max_shown_at": impressions["shown_at"].max().isoformat() if not impressions.empty else "",
            "future_or_current_violations": violations,
            "status": "PASS" if violations == 0 else "FAIL",
        }
        self.leakage_audit.append(audit)
        return rows, audit

    def raw_frame(self, sessions: list[SessionRecord]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for session in sessions:
            session_rows, _ = self._session_rows(session)
            rows.extend(session_rows)
        return pd.DataFrame(rows)

    def fit(self, train_raw: pd.DataFrame) -> None:
        if set(train_raw["split"].unique()) != {"train"}:
            raise ValueError("XGBoost feature fitting is train-only")
        self.static_builder.fit(train_raw[["split", *self.static_columns]])
        self.time_builder.fit(train_raw[["split", *self.time_columns]])
        self._fitted = True

    def transform(self, raw: pd.DataFrame, model_kind: str) -> BuiltFeatures:
        if not self._fitted:
            raise RuntimeError("Feature pipeline must fit on train first")
        if model_kind == "static":
            builder = self.static_builder
            columns = self.static_columns
        elif model_kind == "time":
            builder = self.time_builder
            columns = self.time_columns
        else:
            raise ValueError(model_kind)
        state_hash = builder.state_hash
        transformed = builder.transform(raw[["split", *columns]])
        if builder.state_hash != state_hash:
            raise AssertionError("Transform mutated train-fitted feature state")
        identifiers = raw[["split", "session_id", "candidate_id", "job_id"]].copy()
        return BuiltFeatures(
            identifiers=identifiers,
            raw_features=raw[[*columns]].copy(),
            transformed=transformed,
            labels=raw["label"].astype(int).to_numpy(),
        )

    def schema_receipt(self) -> dict[str, Any]:
        return {
            "static": {
                "groups": ["F0", "F1"],
                "raw_columns": list(self.static_columns),
                "transformed_columns": list(self.static_builder.state.output_columns),
                "schema_sha256": _schema_hash(self.static_builder.state.output_columns),
                "contains_f2": False,
            },
            "time": {
                "groups": ["F0", "F1", "F2"],
                "raw_columns": list(self.time_columns),
                "transformed_columns": list(self.time_builder.state.output_columns),
                "schema_sha256": _schema_hash(self.time_builder.state.output_columns),
                "contains_f2": True,
            },
            "fit_split": "train",
            "static_fit_state_sha256": self.static_builder.state_hash,
            "time_fit_state_sha256": self.time_builder.state_hash,
            "cbf_vocabulary_sha256": self.cbf.vocabulary_hash,
            "locked_cbf_config": self.cbf_config.config_id,
            "excluded": ["label", "censored", "future_actions", "salary", "title", "description"],
        }


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value)) if value > 0 else 0.0
