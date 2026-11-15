from __future__ import annotations

import math
from collections.abc import Sequence


def _validate(
    labels: Sequence[int],
    scores: Sequence[float],
    job_ids: Sequence[object] | None,
    k: int,
) -> list[object]:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if len(labels) == 0:
        raise ValueError("candidate set must not be empty")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be binary 0/1")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("scores must be finite")
    if job_ids is None:
        return list(range(len(labels)))
    if len(job_ids) != len(labels):
        raise ValueError("job_ids and labels must have the same length")
    return list(job_ids)


def rank_indices(
    scores: Sequence[float],
    job_ids: Sequence[object] | None = None,
) -> list[int]:
    if len(scores) == 0:
        raise ValueError("scores must not be empty")
    ids = list(range(len(scores))) if job_ids is None else list(job_ids)
    if len(ids) != len(scores):
        raise ValueError("job_ids and scores must have the same length")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("scores must be finite")
    return sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), str(ids[index]), index),
    )


def recall_at_k(
    labels: Sequence[int],
    scores: Sequence[float],
    k: int = 10,
    job_ids: Sequence[object] | None = None,
) -> float:
    ids = _validate(labels, scores, job_ids, k)
    positives = sum(labels)
    if positives == 0:
        raise ValueError("recall is undefined when there is no positive label")
    order = rank_indices(scores, ids)[:k]
    hits = sum(labels[index] for index in order)
    return hits / positives


def ndcg_at_k(
    labels: Sequence[int],
    scores: Sequence[float],
    k: int = 10,
    job_ids: Sequence[object] | None = None,
) -> float:
    ids = _validate(labels, scores, job_ids, k)
    positives = sum(labels)
    if positives == 0:
        raise ValueError("NDCG is undefined when there is no positive label")

    order = rank_indices(scores, ids)[:k]
    dcg = sum(
        (2**labels[index] - 1) / math.log2(rank + 1)
        for rank, index in enumerate(order, start=1)
    )
    ideal_length = min(positives, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_length + 1))
    return dcg / idcg

