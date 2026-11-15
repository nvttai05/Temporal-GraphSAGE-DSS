from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from dss_stage6.data import (
    DatasetIndex,
    HistoryStore,
    SessionLoader,
    StaticGraphBuilder,
)
from dss_stage6.evaluation import RankingEvaluator
from dss_stage6.features import FeatureBuilder


def deterministic_smoke_scores(job_ids: tuple[str, ...]) -> dict[str, float]:
    """Stable non-model scores used only to exercise the evaluation engine."""

    return {
        job_id: int(hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:16], 16)
        / float(16**16 - 1)
        for job_id in job_ids
    }


def run_six_cell_smoke(index: DatasetIndex) -> pd.DataFrame:
    loader = SessionLoader(index)
    evaluator = RankingEvaluator(k=10)
    primary_delta = int(index.protocol["dataset"]["primary_label_window_days"])
    rows: list[dict[str, object]] = []
    for cell in index.data_cells:
        train_session = next(
            loader.iter_sessions(
                cell.seed, cell.scenario, "train", primary_delta, eligible_only=True
            )
        )
        validation_session = next(
            loader.iter_sessions(
                cell.seed,
                cell.scenario,
                "validation",
                primary_delta,
                eligible_only=True,
            )
        )
        lifecycle = pd.read_csv(
            index.dataset.cell_path(cell.seed, cell.scenario, "job_lifecycle.csv"),
            usecols=["job_id", "skills", "cohort"],
        )
        train_features = train_session.candidates[["job_id", "split"]].merge(
            lifecycle, on="job_id", how="left", validate="one_to_one"
        )
        validation_features = validation_session.candidates[["job_id", "split"]].merge(
            lifecycle, on="job_id", how="left", validate="one_to_one"
        )
        features = FeatureBuilder(categorical_columns=["skills", "cohort"])
        features.fit(train_features)
        state_before = features.state_hash
        transformed = features.transform(validation_features)
        state_after = features.state_hash

        history = HistoryStore.from_dataset(index.dataset, cell.seed, cell.scenario)
        snapshot = StaticGraphBuilder(history).build(validation_session)
        scores = deterministic_smoke_scores(validation_session.exposed_job_ids)
        metrics = evaluator.evaluate_session(validation_session, scores)
        rows.append(
            {
                "seed": cell.seed,
                "scenario": cell.scenario,
                "split": "validation",
                "delta_days": primary_delta,
                "session_id": validation_session.session_id,
                "candidate_count": len(validation_session.candidates),
                "history_edges": len(snapshot.edges),
                "graph_nodes": snapshot.graph.number_of_nodes(),
                "graph_edges": snapshot.graph.number_of_edges(),
                "feature_columns": transformed.shape[1],
                "feature_state_unchanged": state_before == state_after,
                "ndcg_at_10": metrics["ndcg_at_10"],
                "recall_at_10": metrics["recall_at_10"],
                "future_edge_violations": int(
                    (
                        (snapshot.edges["action_at"] >= snapshot.prediction_at)
                        | (snapshot.edges["observed_at"] >= snapshot.prediction_at)
                        | (
                            snapshot.edges["session_id"].astype(str)
                            == validation_session.session_id
                        )
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "scenario"]).reset_index(drop=True)


def smoke_digest(frame: pd.DataFrame) -> str:
    csv_bytes = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode(
        "utf-8"
    )
    return hashlib.sha256(csv_bytes).hexdigest()
