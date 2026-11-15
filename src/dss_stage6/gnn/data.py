from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from dss_stage6.data import DatasetIndex, HistoryStore, SessionLoader, SessionRecord
from dss_stage6.models.baselines import ContentRepository


RELATIONS = (
    "candidate_to_skill",
    "skill_to_candidate",
    "job_to_skill",
    "skill_to_job",
    "impression_candidate_to_job",
    "impression_job_to_candidate",
    "view_candidate_to_job",
    "view_job_to_candidate",
    "save_candidate_to_job",
    "save_job_to_candidate",
    "apply_candidate_to_job",
    "apply_job_to_candidate",
)
RELATION_TO_ID = {name: index for index, name in enumerate(RELATIONS)}
EVENT_RELATIONS = set(range(4, len(RELATIONS)))


@dataclass
class GraphExample:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_age_days: torch.Tensor
    edge_is_event: torch.Tensor
    target_candidate_index: torch.Tensor
    target_job_indices: torch.Tensor
    labels: torch.Tensor
    session: SessionRecord
    node_ids: tuple[str, ...]
    audit: dict[str, Any]

    def to(self, device: torch.device | str) -> "GraphExample":
        return GraphExample(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            edge_age_days=self.edge_age_days.to(device),
            edge_is_event=self.edge_is_event.to(device),
            target_candidate_index=self.target_candidate_index.to(device),
            target_job_indices=self.target_job_indices.to(device),
            labels=self.labels.to(device),
            session=self.session,
            node_ids=self.node_ids,
            audit=self.audit,
        )


class GNNGraphAdapter:
    """Adapt frozen rolling snapshots to relation-aware PyTorch graphs."""

    skill_hash_dim = 32
    node_type_dim = 3

    def __init__(self, index: DatasetIndex, seed: int, scenario: str):
        self.index = index
        self.seed = int(seed)
        self.scenario = str(scenario)
        self.loader = SessionLoader(index)
        self.history = HistoryStore.from_dataset(index.dataset, seed, scenario)
        self.content = ContentRepository(index, seed, scenario)
        self.impressions = pd.read_csv(
            index.dataset.cell_path(seed, scenario, "impressions.csv")
        )
        self.impressions["shown_at"] = pd.to_datetime(
            self.impressions["shown_at"], utc=True
        )

    @property
    def input_dim(self) -> int:
        return self.node_type_dim + self.skill_hash_dim

    @staticmethod
    def _prediction_time(session: SessionRecord) -> pd.Timestamp:
        prediction = pd.Timestamp(session.prediction_at)
        return (
            prediction.tz_localize("UTC")
            if prediction.tzinfo is None
            else prediction.tz_convert("UTC")
        )

    @staticmethod
    def _skill_tokens(value: str) -> tuple[str, ...]:
        return tuple(sorted({token for token in str(value).split("|") if token}))

    def _skill_slot(self, skill: str) -> int:
        return int(hashlib.sha256(skill.encode("utf-8")).hexdigest()[:8], 16) % self.skill_hash_dim

    def _feature(self, node_type: str, skills: tuple[str, ...]) -> np.ndarray:
        feature = np.zeros(self.input_dim, dtype=np.float32)
        type_index = {"candidate": 0, "job": 1, "skill": 2}[node_type]
        feature[type_index] = 1.0
        for skill in skills:
            feature[self.node_type_dim + self._skill_slot(skill)] = 1.0
        return feature

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

    def build(self, session: SessionRecord) -> GraphExample:
        prediction = self._prediction_time(session)
        actions = self.history.as_of(
            prediction, current_session_id=session.session_id
        )
        impressions = self.impressions.loc[
            (self.impressions["shown_at"] < prediction)
            & (self.impressions["session_id"].astype(str) != session.session_id)
        ].copy()
        candidate_ids = set(actions["candidate_id"].astype(str))
        candidate_ids.update(impressions["candidate_id"].astype(str))
        candidate_ids.add(session.candidate_id)
        job_ids = set(actions["job_id"].astype(str))
        job_ids.update(impressions["job_id"].astype(str))
        job_ids.update(session.exposed_job_ids)

        candidate_skills: dict[str, tuple[str, ...]] = {}
        for candidate_id in sorted(candidate_ids):
            content = self.content.candidate_at(candidate_id, prediction, session.split)
            candidate_skills[candidate_id] = self._skill_tokens(content["skills"])
        job_skills: dict[str, tuple[str, ...]] = {}
        for job_id in sorted(job_ids):
            job_skills[job_id] = self._skill_tokens(
                str(self.content.jobs.loc[job_id, "skills"])
            )
        skill_ids = sorted(
            {
                skill
                for skills in [*candidate_skills.values(), *job_skills.values()]
                for skill in skills
            }
        )
        node_ids = tuple(
            [f"candidate:{value}" for value in sorted(candidate_ids)]
            + [f"job:{value}" for value in sorted(job_ids)]
            + [f"skill:{value}" for value in skill_ids]
        )
        node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        features: list[np.ndarray] = []
        for node_id in node_ids:
            kind, value = node_id.split(":", 1)
            if kind == "candidate":
                skills = candidate_skills[value]
            elif kind == "job":
                skills = job_skills[value]
            else:
                skills = (value,)
            features.append(self._feature(kind, skills))

        src: list[int] = []
        dst: list[int] = []
        relation: list[int] = []
        age_days: list[float] = []
        is_event: list[bool] = []

        def add_edge(
            source: str,
            target: str,
            relation_name: str,
            age: float,
            event: bool,
        ) -> None:
            src.append(node_index[source])
            dst.append(node_index[target])
            relation.append(RELATION_TO_ID[relation_name])
            age_days.append(max(0.0, float(age)))
            is_event.append(bool(event))

        for candidate_id, skills in candidate_skills.items():
            for skill in skills:
                add_edge(
                    f"candidate:{candidate_id}",
                    f"skill:{skill}",
                    "candidate_to_skill",
                    0.0,
                    False,
                )
                add_edge(
                    f"skill:{skill}",
                    f"candidate:{candidate_id}",
                    "skill_to_candidate",
                    0.0,
                    False,
                )
        for job_id, skills in job_skills.items():
            for skill in skills:
                add_edge(
                    f"job:{job_id}", f"skill:{skill}", "job_to_skill", 0.0, False
                )
                add_edge(
                    f"skill:{skill}", f"job:{job_id}", "skill_to_job", 0.0, False
                )
        for row in impressions.itertuples(index=False):
            age = (prediction - row.shown_at).total_seconds() / 86400.0
            add_edge(
                f"candidate:{row.candidate_id}",
                f"job:{row.job_id}",
                "impression_candidate_to_job",
                age,
                True,
            )
            add_edge(
                f"job:{row.job_id}",
                f"candidate:{row.candidate_id}",
                "impression_job_to_candidate",
                age,
                True,
            )
        for row in actions.itertuples(index=False):
            age = (prediction - row.action_at).total_seconds() / 86400.0
            add_edge(
                f"candidate:{row.candidate_id}",
                f"job:{row.job_id}",
                f"{row.action_type}_candidate_to_job",
                age,
                True,
            )
            add_edge(
                f"job:{row.job_id}",
                f"candidate:{row.candidate_id}",
                f"{row.action_type}_job_to_candidate",
                age,
                True,
            )
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
        audit = {
            "seed": session.seed,
            "scenario": session.scenario,
            "split": session.split,
            "session_id": session.session_id,
            "prediction_at": prediction.isoformat(),
            "node_count": len(node_ids),
            "edge_count": len(src),
            "event_edge_count": int(sum(is_event)),
            "content_edge_count": int(len(is_event) - sum(is_event)),
            "future_or_current_violations": violations,
            "target_pair_count": len(session.candidates),
            "status": "PASS" if violations == 0 else "FAIL",
        }
        if violations:
            raise AssertionError(f"GNN graph leakage detected: {audit}")
        return GraphExample(
            node_features=torch.tensor(np.stack(features), dtype=torch.float32),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            edge_type=torch.tensor(relation, dtype=torch.long),
            edge_age_days=torch.tensor(age_days, dtype=torch.float32),
            edge_is_event=torch.tensor(is_event, dtype=torch.bool),
            target_candidate_index=torch.tensor(
                node_index[f"candidate:{session.candidate_id}"], dtype=torch.long
            ),
            target_job_indices=torch.tensor(
                [node_index[f"job:{job_id}"] for job_id in session.exposed_job_ids],
                dtype=torch.long,
            ),
            labels=torch.tensor(session.labels, dtype=torch.float32),
            session=session,
            node_ids=node_ids,
            audit=audit,
        )
