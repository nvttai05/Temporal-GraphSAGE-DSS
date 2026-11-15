from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from dss_stage6.data.session import SessionRecord

from .metrics import ndcg_at_k, recall_at_k


@dataclass(frozen=True)
class EvaluationResult:
    session_metrics: pd.DataFrame
    cell_metrics: pd.DataFrame
    summary: dict[str, float | int]


class RankingEvaluator:
    """The sole Stage 6 exposed-set implementation of Recall/NDCG at K."""

    def __init__(self, k: int = 10):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)

    def evaluate_session(
        self,
        session: SessionRecord,
        scores: Mapping[str, float] | Sequence[float],
    ) -> dict[str, object]:
        if not session.top10_eligible:
            raise ValueError(f"Session is not Top-{self.k} eligible: {session.session_id}")
        if len(session.candidates) < self.k or session.positive_count < 1:
            raise ValueError("Session eligibility metadata conflicts with candidate labels")
        job_ids = list(session.candidates["job_id"].astype(str))
        if isinstance(scores, Mapping):
            score_keys = set(map(str, scores.keys()))
            expected_keys = set(job_ids)
            if score_keys != expected_keys:
                missing = sorted(expected_keys - score_keys)
                extra = sorted(score_keys - expected_keys)
                raise ValueError(f"Predictions must equal exposed set; missing={missing}, extra={extra}")
            ordered_scores = [float(scores[job_id]) for job_id in job_ids]
        else:
            ordered_scores = [float(value) for value in scores]
            if len(ordered_scores) != len(job_ids):
                raise ValueError("One prediction is required for every exposed candidate")
        if not np.isfinite(ordered_scores).all():
            raise ValueError("Scores must be finite")
        labels = list(session.candidates["label"].astype(int))
        new_job_labels = list(
            (
                (session.candidates["label"].astype(int) == 1)
                & (session.candidates["new_job"].astype(int) == 1)
            ).astype(int)
        )
        has_positive_new_job = bool(sum(new_job_labels))
        return {
            "seed": session.seed,
            "scenario": session.scenario,
            "split": session.split,
            "delta_days": session.delta_days,
            "session_id": session.session_id,
            "candidate_count": len(job_ids),
            "positive_count": int(sum(labels)),
            f"recall_at_{self.k}": recall_at_k(labels, ordered_scores, self.k, job_ids),
            f"ndcg_at_{self.k}": ndcg_at_k(labels, ordered_scores, self.k, job_ids),
            "changed_profile": int(session.candidates["changed_profile"].max()),
            "low_history": int(session.candidates["low_history"].max()),
            "has_positive_new_job": int(has_positive_new_job),
            f"new_job_recall_at_{self.k}": (
                recall_at_k(new_job_labels, ordered_scores, self.k, job_ids)
                if has_positive_new_job
                else float("nan")
            ),
            f"new_job_ndcg_at_{self.k}": (
                ndcg_at_k(new_job_labels, ordered_scores, self.k, job_ids)
                if has_positive_new_job
                else float("nan")
            ),
        }

    def evaluate(
        self,
        sessions: Sequence[SessionRecord],
        score_provider: Callable[[SessionRecord], Mapping[str, float] | Sequence[float]],
    ) -> EvaluationResult:
        rows = [self.evaluate_session(session, score_provider(session)) for session in sessions]
        if not rows:
            raise ValueError("At least one scoreable session is required")
        session_metrics = pd.DataFrame(rows).sort_values(
            ["seed", "scenario", "split", "session_id"], kind="stable"
        ).reset_index(drop=True)
        metric_columns = [f"ndcg_at_{self.k}", f"recall_at_{self.k}"]
        cell_metrics = (
            session_metrics.groupby(["seed", "scenario"], as_index=False)[metric_columns]
            .mean()
            .sort_values(["seed", "scenario"])
            .reset_index(drop=True)
        )
        summary: dict[str, float | int] = {
            "session_count": len(session_metrics),
            "data_cell_count": len(cell_metrics),
            f"macro_over_cells_ndcg_at_{self.k}": float(cell_metrics[f"ndcg_at_{self.k}"].mean()),
            f"macro_over_cells_recall_at_{self.k}": float(cell_metrics[f"recall_at_{self.k}"].mean()),
            f"pooled_by_session_ndcg_at_{self.k}": float(session_metrics[f"ndcg_at_{self.k}"].mean()),
            f"pooled_by_session_recall_at_{self.k}": float(session_metrics[f"recall_at_{self.k}"].mean()),
        }
        return EvaluationResult(session_metrics, cell_metrics, summary)
