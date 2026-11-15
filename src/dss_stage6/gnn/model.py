from __future__ import annotations

import math

import torch
from torch import nn

from .data import GraphExample, RELATIONS


def decay_weights(
    age_days: torch.Tensor,
    is_event: torch.Tensor,
    half_life_days: float,
) -> torch.Tensor:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    if (age_days < 0).any():
        raise ValueError("edge ages must be non-negative")
    decayed = torch.exp(-math.log(2.0) * age_days / float(half_life_days))
    return torch.where(is_event, decayed, torch.ones_like(decayed))


class RelationWeightedSAGELayer(nn.Module):
    """Relation-aware weighted-mean aggregation implemented with index_add_."""

    def __init__(self, input_dim: int, output_dim: int, num_relations: int):
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.relation_linears = nn.ModuleList(
            nn.Linear(input_dim, output_dim, bias=False)
            for _ in range(num_relations)
        )
        self.num_relations = int(num_relations)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if not (edge_index.shape[1] == edge_type.numel() == edge_weight.numel()):
            raise ValueError("edge tensors must have the same edge count")
        node_count = x.shape[0]
        aggregate = x.new_zeros((node_count, self.self_linear.out_features))
        active_relations = x.new_zeros((node_count, 1))
        source, destination = edge_index
        eps = torch.finfo(x.dtype).eps
        for relation_id, transform in enumerate(self.relation_linears):
            mask = edge_type == relation_id
            if not bool(mask.any()):
                continue
            relation_source = source[mask]
            relation_destination = destination[mask]
            weights = edge_weight[mask].to(dtype=x.dtype).unsqueeze(1)
            messages = transform(x[relation_source]) * weights
            summed = x.new_zeros((node_count, messages.shape[1]))
            denominator = x.new_zeros((node_count, 1))
            summed.index_add_(0, relation_destination, messages)
            denominator.index_add_(0, relation_destination, weights)
            has_relation = denominator > 0
            aggregate = aggregate + summed / denominator.clamp_min(eps)
            active_relations = active_relations + has_relation.to(x.dtype)
        aggregate = aggregate / active_relations.clamp_min(1.0)
        return self.self_linear(x) + aggregate


class PairScorer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate: torch.Tensor, jobs: torch.Tensor) -> torch.Tensor:
        candidate_expanded = candidate.unsqueeze(0).expand_as(jobs)
        pair = torch.cat(
            [
                candidate_expanded,
                jobs,
                candidate_expanded * jobs,
                torch.abs(candidate_expanded - jobs),
            ],
            dim=1,
        )
        return self.network(pair).squeeze(1)


class GraphSAGERecommender(nn.Module):
    """Shared two-layer encoder/scorer for M0 static and M1 temporal variants."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        half_life_days: float = 14.0,
        temporal: bool = False,
    ):
        super().__init__()
        self.temporal = bool(temporal)
        self.half_life_days = float(half_life_days)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                RelationWeightedSAGELayer(hidden_dim, hidden_dim, len(RELATIONS)),
                RelationWeightedSAGELayer(hidden_dim, hidden_dim, len(RELATIONS)),
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.scorer = PairScorer(hidden_dim)

    def edge_weights(self, example: GraphExample) -> torch.Tensor:
        if not self.temporal:
            return torch.ones_like(example.edge_age_days)
        return decay_weights(
            example.edge_age_days, example.edge_is_event, self.half_life_days
        )

    def forward(
        self,
        example: GraphExample,
        *,
        edge_weight_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = (
            self.edge_weights(example)
            if edge_weight_override is None
            else edge_weight_override
        )
        x = torch.relu(self.encoder(example.node_features))
        for layer in self.layers:
            x = torch.relu(
                layer(x, example.edge_index, example.edge_type, weights)
            )
            x = self.dropout(x)
        candidate = x[example.target_candidate_index]
        jobs = x[example.target_job_indices]
        return self.scorer(candidate, jobs)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
