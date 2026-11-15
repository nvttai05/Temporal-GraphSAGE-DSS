from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from dss_stage6.data import SessionRecord
from dss_stage6.gnn.data import GraphExample
from dss_stage6.gnn.model import GraphSAGERecommender
from dss_stage6.gnn.trainer import weighted_bce_logits


def graph_example(device: str = "cpu") -> GraphExample:
    candidates = pd.DataFrame(
        {
            "job_id": [f"job-{index:02d}" for index in range(10)],
            "label": [1] + [0] * 9,
            "new_job": [0] * 10,
            "changed_profile": [0] * 10,
            "low_history": [1] * 10,
        }
    )
    session = SessionRecord(
        53021,
        "S1",
        "validation",
        14,
        "session-current",
        "candidate-a",
        pd.Timestamp("2027-05-01T00:00:00Z"),
        10,
        1,
        True,
        "scoreable",
        candidates,
    )
    node_count = 12
    example = GraphExample(
        node_features=torch.randn(node_count, 35),
        edge_index=torch.tensor(
            [[0, 11, 0, 1, 0, 2], [11, 0, 1, 0, 2, 0]], dtype=torch.long
        ),
        edge_type=torch.tensor([0, 1, 6, 7, 8, 9], dtype=torch.long),
        edge_age_days=torch.tensor([0.0, 0.0, 1.0, 1.0, 14.0, 14.0]),
        edge_is_event=torch.tensor([False, False, True, True, True, True]),
        target_candidate_index=torch.tensor(0),
        target_job_indices=torch.arange(1, 11),
        labels=torch.tensor([1.0] + [0.0] * 9),
        session=session,
        node_ids=tuple(["candidate:c"] + [f"job:{i}" for i in range(10)] + ["skill:s"]),
        audit={"future_or_current_violations": 0},
    )
    return example.to(device)


def test_static_and_temporal_share_architecture_and_w1_equivalence():
    torch.manual_seed(42)
    static = GraphSAGERecommender(35, hidden_dim=16, dropout=0.0, temporal=False)
    temporal = GraphSAGERecommender(35, hidden_dim=16, dropout=0.0, temporal=True)
    temporal.load_state_dict(static.state_dict())
    example = graph_example()
    static.eval()
    temporal.eval()
    ones = torch.ones(example.edge_index.shape[1])
    assert static.state_dict().keys() == temporal.state_dict().keys()
    assert torch.allclose(
        static(example, edge_weight_override=ones),
        temporal(example, edge_weight_override=ones),
        atol=1e-7,
    )


def test_logits_loss_is_finite_and_gradients_nonzero():
    model = GraphSAGERecommender(35, hidden_dim=16, dropout=0.0, temporal=True)
    example = graph_example()
    logits = model(example)
    loss = weighted_bce_logits(logits, example.labels, torch.tensor(9.0))
    loss.backward()
    gradient_sum = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert torch.isfinite(loss)
    assert gradient_sum > 0


def test_checkpoint_restore_preserves_scores(tmp_path: Path):
    model = GraphSAGERecommender(35, hidden_dim=16, dropout=0.0, temporal=True)
    model.eval()
    example = graph_example()
    expected = model(example).detach()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)
    restored = GraphSAGERecommender(35, hidden_dim=16, dropout=0.0, temporal=True)
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    )
    restored.eval()
    assert torch.equal(expected, restored(example).detach())


def test_model_initialization_and_scores_are_deterministic_for_seed_42():
    example = graph_example()
    scores = []
    states = []
    for _ in range(2):
        torch.manual_seed(42)
        model = GraphSAGERecommender(35, hidden_dim=16, dropout=0.0, temporal=True)
        model.eval()
        scores.append(model(example).detach())
        states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
    assert torch.equal(scores[0], scores[1])
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable locally")
def test_model_and_graph_tensors_share_cuda_device():
    example = graph_example("cuda")
    model = GraphSAGERecommender(35, hidden_dim=16, dropout=0.0, temporal=True).cuda()
    logits = model(example)
    assert logits.device.type == "cuda"
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
