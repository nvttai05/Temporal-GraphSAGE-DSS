import math

import pytest

from dss_stage6.evaluation import ndcg_at_k, recall_at_k


def test_hand_calculated_fixture_positive_at_ranks_2_and_11():
    labels = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    scores = list(reversed(range(12)))
    job_ids = [f"job-{index:02d}" for index in range(12)]

    expected_dcg = 1.0 / math.log2(3)
    expected_idcg = 1.0 + 1.0 / math.log2(3)

    assert recall_at_k(labels, scores, 10, job_ids) == 0.5
    assert ndcg_at_k(labels, scores, 10, job_ids) == pytest.approx(
        expected_dcg / expected_idcg,
        abs=1e-12,
    )


def test_job_id_breaks_score_tie_ascending():
    labels = [1, 0, 0]
    scores = [1.0, 1.0, 0.0]
    job_ids = ["b", "a", "c"]

    assert recall_at_k(labels, scores, 1, job_ids) == 0.0
    assert ndcg_at_k(labels, scores, 1, job_ids) == 0.0


def test_fewer_than_ten_candidates_is_supported_by_metric_function():
    labels = [1, 0, 1]
    scores = [3.0, 1.0, 2.0]
    assert recall_at_k(labels, scores, 10) == 1.0
    assert ndcg_at_k(labels, scores, 10) == 1.0


def test_no_positive_is_rejected_instead_of_silently_returning_zero():
    with pytest.raises(ValueError):
        recall_at_k([0, 0], [2.0, 1.0])
    with pytest.raises(ValueError):
        ndcg_at_k([0, 0], [2.0, 1.0])


def test_metric_is_invariant_to_input_row_order():
    labels_a = [1, 0, 1, 0]
    scores_a = [0.7, 0.9, 0.6, 0.1]
    ids_a = ["a", "b", "c", "d"]

    labels_b = [0, 1, 0, 1]
    scores_b = [0.1, 0.6, 0.9, 0.7]
    ids_b = ["d", "c", "b", "a"]

    assert recall_at_k(labels_a, scores_a, 3, ids_a) == recall_at_k(
        labels_b, scores_b, 3, ids_b
    )
    assert ndcg_at_k(labels_a, scores_a, 3, ids_a) == ndcg_at_k(
        labels_b, scores_b, 3, ids_b
    )

