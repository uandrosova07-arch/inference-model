from __future__ import annotations

import json

import numpy as np
import pytest

from src.index.query_router import (
    FEATURE_NAMES,
    feature_matrix,
    load_router_model,
    predict_policy_index,
    query_features,
    route_vector,
    train_router_tree,
)


def test_query_features_are_finite_and_concentration_sensitive() -> None:
    concentrated = query_features({"a": 4.0, "b": 1.0})
    diffuse = query_features({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0})

    assert tuple(concentrated) == FEATURE_NAMES
    assert concentrated["nnz"] == 2
    assert concentrated["top1_mass"] == pytest.approx(0.8)
    assert concentrated["normalized_entropy"] < diffuse["normalized_entropy"]
    assert all(np.isfinite(value) for value in concentrated.values())


def test_router_tree_learns_small_interpretable_policy_split() -> None:
    rows = []
    labels = []
    for nnz in range(1, 101):
        features = {name: 0.0 for name in FEATURE_NAMES}
        features["nnz"] = float(nnz)
        rows.append(features)
        labels.append(0 if nnz <= 50 else 2)

    tree = train_router_tree(
        feature_matrix(rows),
        labels,
        max_depth=1,
        min_samples_leaf=10,
        safety_quantile=0.95,
    )

    assert tree["type"] == "split"
    assert tree["feature"] == "nnz"
    assert predict_policy_index(tree, rows[10]) == 0
    assert predict_policy_index(tree, rows[90]) == 2


def test_router_model_round_trip_and_policy_selection(tmp_path) -> None:
    model = {
        "schema_version": 1,
        "model_type": "ordinal_cart",
        "feature_names": list(FEATURE_NAMES),
        "policies": [
            {"name": "fast", "parameters": {"query_cut": 6}},
            {"name": "safe", "parameters": {"query_cut": 10}},
        ],
        "tree": {"type": "leaf", "policy_index": 1, "samples": 1},
    }
    path = tmp_path / "router.json"
    path.write_text(json.dumps(model), encoding="utf-8")

    routed = route_vector(load_router_model(path), {"token": 1.0})

    assert routed["policy"] == "safe"
    assert routed["parameters"] == {"query_cut": 10}
