from __future__ import annotations

import torch

from dss_stage6.gnn.model import RelationWeightedSAGELayer, decay_weights


def test_decay_weight_is_monotonic_and_content_membership_is_one():
    ages = torch.tensor([0.0, 7.0, 14.0, 28.0, 100.0])
    events = torch.tensor([True, True, True, True, False])
    weights = decay_weights(ages, events, half_life_days=14.0)
    assert weights[0] == 1.0
    assert weights[0] > weights[1] > weights[2] > weights[3]
    assert torch.isclose(weights[2], torch.tensor(0.5))
    assert weights[4] == 1.0


def test_weighted_aggregation_with_all_ones_matches_static_mean():
    layer = RelationWeightedSAGELayer(2, 2, num_relations=1)
    with torch.no_grad():
        layer.self_linear.weight.zero_()
        layer.self_linear.bias.zero_()
        layer.relation_linears[0].weight.copy_(torch.eye(2))
    x = torch.tensor([[1.0, 0.0], [3.0, 2.0], [0.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [2, 2]])
    edge_type = torch.zeros(2, dtype=torch.long)
    output = layer(x, edge_index, edge_type, torch.ones(2))
    assert torch.allclose(output[2], torch.tensor([2.0, 1.0]))


def test_weighted_aggregation_has_finite_nonzero_gradient():
    layer = RelationWeightedSAGELayer(2, 3, num_relations=1)
    x = torch.randn(4, 2, requires_grad=True)
    output = layer(
        x,
        torch.tensor([[0, 1, 2], [3, 3, 3]]),
        torch.zeros(3, dtype=torch.long),
        torch.tensor([1.0, 0.5, 0.25]),
    )
    loss = output.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
