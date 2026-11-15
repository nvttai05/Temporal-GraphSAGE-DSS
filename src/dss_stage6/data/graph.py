from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd

from .history import HistoryStore
from .session import SessionRecord


@dataclass(frozen=True)
class StaticGraphSnapshot:
    graph: nx.MultiDiGraph
    edges: pd.DataFrame
    prediction_at: pd.Timestamp
    current_session_id: str


class StaticGraphBuilder:
    """Build a static candidate-job graph from strictly observable history."""

    def __init__(self, history: HistoryStore):
        self.history = history

    def build(self, session: SessionRecord) -> StaticGraphSnapshot:
        edges = self.history.as_of(
            session.prediction_at, current_session_id=session.session_id
        )
        prediction = pd.Timestamp(session.prediction_at)
        if prediction.tzinfo is None:
            prediction = prediction.tz_localize("UTC")
        else:
            prediction = prediction.tz_convert("UTC")
        if (
            (edges["action_at"] >= prediction).any()
            or (edges["observed_at"] >= prediction).any()
            or (edges["session_id"].astype(str) == session.session_id).any()
        ):
            raise AssertionError("Static graph contains future or current-session edges")
        graph = nx.MultiDiGraph()
        for row in edges.itertuples(index=False):
            candidate_node = f"candidate:{row.candidate_id}"
            job_node = f"job:{row.job_id}"
            graph.add_node(candidate_node, node_type="candidate", source_id=row.candidate_id)
            graph.add_node(job_node, node_type="job", source_id=row.job_id)
            graph.add_edge(
                candidate_node,
                job_node,
                key=row.action_id,
                action_type=row.action_type,
                action_at=row.action_at.isoformat(),
                observed_at=row.observed_at.isoformat(),
                session_id=row.session_id,
            )
        return StaticGraphSnapshot(graph, edges, prediction, session.session_id)
