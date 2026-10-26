from __future__ import annotations

import pytest

from src.metrics.retrieval import evaluate


def test_complete_metric_grid() -> None:
    qrels = {
        "q1": {"d1": 1, "d2": 1},
        "q2": {"d3": 1},
    }
    ranking = {
        "q1": ["wrong", "d1", "d2"],
        "q2": ["d3"],
    }
    metrics = evaluate(
        qrels,
        ranking,
        {
            "cutoffs": [1, 2, 3],
            "mrr_cutoff": 2,
            "ndcg_cutoff": 2,
            "map_cutoff": 3,
            "precision_cutoff": 1,
            "recall_cutoff": 3,
            "save_per_query": False,
        },
    )

    assert metrics["queries"] == 2
    assert metrics["queries_ranked"] == 2
    assert metrics["mrr_at_2"] == pytest.approx(0.75)
    assert metrics["recall_at_3"] == pytest.approx(1.0)
    assert metrics["precision_at_1"] == pytest.approx(0.5)
    assert metrics["map_at_3"] == pytest.approx((7 / 12 + 1) / 2)
    assert metrics["mrr"] == metrics["mrr_at_2"]
