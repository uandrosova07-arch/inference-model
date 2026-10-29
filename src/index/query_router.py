from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_NAMES = (
    "nnz",
    "l1",
    "l2",
    "max_weight",
    "mean_weight",
    "std_weight",
    "top1_mass",
    "top2_mass",
    "top4_mass",
    "normalized_entropy",
    "effective_terms",
    "top_weight_gap",
)
ALLOWED_POLICY_PARAMETERS = {"query_cut", "heap_factor", "n_knn", "first_sorted"}


def query_features(vector: Mapping[str, float]) -> dict[str, float]:
    """Return cheap, index-independent features for one SPLADE query vector."""
    weights = np.asarray([float(value) for value in vector.values()], dtype=np.float64)
    if weights.size == 0:
        return {name: 0.0 for name in FEATURE_NAMES}
    if not np.all(np.isfinite(weights)):
        raise ValueError("Query vector contains a non-finite weight")
    if np.any(weights < 0):
        raise ValueError("Query router expects non-negative SPLADE weights")

    ordered = np.sort(weights)[::-1]
    total = float(ordered.sum())
    probabilities = ordered / total if total > 0 else np.zeros_like(ordered)
    positive = probabilities[probabilities > 0]
    entropy = float(-np.sum(positive * np.log(positive))) if positive.size else 0.0
    normalized_entropy = entropy / math.log(len(ordered)) if len(ordered) > 1 else 0.0
    maximum = float(ordered[0])
    second = float(ordered[1]) if len(ordered) > 1 else 0.0

    def mass(count: int) -> float:
        return float(ordered[:count].sum() / total) if total > 0 else 0.0

    return {
        "nnz": float(len(ordered)),
        "l1": total,
        "l2": float(np.linalg.norm(ordered)),
        "max_weight": maximum,
        "mean_weight": float(ordered.mean()),
        "std_weight": float(ordered.std()),
        "top1_mass": mass(1),
        "top2_mass": mass(2),
        "top4_mass": mass(4),
        "normalized_entropy": normalized_entropy,
        "effective_terms": float(math.exp(entropy)),
        "top_weight_gap": (maximum - second) / maximum if maximum > 0 else 0.0,
    }


def feature_matrix(rows: Sequence[Mapping[str, float]]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows])


def _leaf(labels: np.ndarray, safety_quantile: float) -> dict[str, Any]:
    ordered = np.sort(labels.astype(np.int64, copy=False))
    position = min(len(ordered) - 1, math.ceil(safety_quantile * (len(ordered) - 1)))
    counts = np.bincount(ordered)
    return {
        "type": "leaf",
        "policy_index": int(ordered[position]),
        "samples": len(ordered),
        "label_mean": float(ordered.mean()),
        "label_counts": {str(index): int(count) for index, count in enumerate(counts)},
    }


def train_router_tree(
    matrix: np.ndarray,
    labels: Sequence[int],
    *,
    max_depth: int = 3,
    min_samples_leaf: int = 64,
    safety_quantile: float = 0.95,
) -> dict[str, Any]:
    """Fit a tiny ordinal CART tree and return a JSON-serializable representation."""
    matrix = np.asarray(matrix, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected feature matrix with {len(FEATURE_NAMES)} columns")
    if len(matrix) != len(label_array) or not len(label_array):
        raise ValueError("Features and labels must have the same non-zero length")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be positive")
    if not 0.5 <= safety_quantile <= 1.0:
        raise ValueError("safety_quantile must be in [0.5, 1.0]")

    def squared_error(values: np.ndarray) -> float:
        if not len(values):
            return 0.0
        return float(np.square(values - values.mean()).sum())

    def build(indices: np.ndarray, depth: int) -> dict[str, Any]:
        node_labels = label_array[indices]
        if (
            depth >= max_depth
            or len(indices) < 2 * min_samples_leaf
            or np.all(node_labels == node_labels[0])
        ):
            return _leaf(node_labels, safety_quantile)

        parent_error = squared_error(node_labels)
        best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
        for feature_index in range(matrix.shape[1]):
            values = matrix[indices, feature_index]
            order = np.argsort(values, kind="stable")
            sorted_indices = indices[order]
            sorted_values = values[order]
            sorted_labels = label_array[sorted_indices].astype(np.float64)
            cumulative = np.cumsum(sorted_labels)
            cumulative_square = np.cumsum(np.square(sorted_labels))
            positions = np.arange(min_samples_leaf, len(indices) - min_samples_leaf + 1)
            if not len(positions):
                continue
            positions = positions[sorted_values[positions - 1] < sorted_values[positions]]
            if not len(positions):
                continue

            left_sum = cumulative[positions - 1]
            left_square = cumulative_square[positions - 1]
            left_error = left_square - np.square(left_sum) / positions
            right_count = len(indices) - positions
            right_sum = cumulative[-1] - left_sum
            right_square = cumulative_square[-1] - left_square
            right_error = right_square - np.square(right_sum) / right_count
            gains = parent_error - left_error - right_error
            candidate_index = int(np.argmax(gains))
            gain = float(gains[candidate_index])
            if best is not None and gain <= best[0]:
                continue
            split_position = int(positions[candidate_index])
            threshold = float(
                (sorted_values[split_position - 1] + sorted_values[split_position]) / 2
            )
            best = (
                gain,
                feature_index,
                threshold,
                sorted_indices[:split_position],
                sorted_indices[split_position:],
            )

        if best is None or best[0] <= 1e-12:
            return _leaf(node_labels, safety_quantile)
        gain, feature_index, threshold, left_indices, right_indices = best
        return {
            "type": "split",
            "feature": FEATURE_NAMES[feature_index],
            "threshold": threshold,
            "gain": gain,
            "samples": len(indices),
            "left": build(left_indices, depth + 1),
            "right": build(right_indices, depth + 1),
        }

    return build(np.arange(len(label_array)), 0)


def predict_policy_index(tree: Mapping[str, Any], features: Mapping[str, float]) -> int:
    node = tree
    while node.get("type") == "split":
        feature = str(node["feature"])
        if feature not in features:
            raise ValueError(f"Router feature is missing: {feature}")
        branch = "left" if float(features[feature]) <= float(node["threshold"]) else "right"
        node = node[branch]
    if node.get("type") != "leaf":
        raise ValueError("Invalid router tree node")
    return int(node["policy_index"])


def validate_router_model(model: Mapping[str, Any]) -> None:
    if int(model.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported query router schema")
    if tuple(model.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Query router feature schema does not match this checkout")
    policies = model.get("policies")
    if not isinstance(policies, list) or not policies:
        raise ValueError("Query router must contain at least one policy")
    names = set()
    for policy in policies:
        name = str(policy.get("name", ""))
        if not name or name in names:
            raise ValueError("Query router policy names must be non-empty and unique")
        names.add(name)
        parameters = policy.get("parameters", {})
        unknown = set(parameters).difference(ALLOWED_POLICY_PARAMETERS)
        if unknown:
            raise ValueError(f"Unsupported adaptive search parameters: {sorted(unknown)}")
        if "query_cut" in parameters and int(parameters["query_cut"]) <= 0:
            raise ValueError("query_cut must be positive")
        if "heap_factor" in parameters and float(parameters["heap_factor"]) <= 0:
            raise ValueError("heap_factor must be positive")
        if "n_knn" in parameters and int(parameters["n_knn"]) < 0:
            raise ValueError("n_knn must be non-negative")
    if not isinstance(model.get("tree"), Mapping):
        raise TypeError("Query router tree is missing")


def load_router_model(path: Path) -> dict[str, Any]:
    model = json.loads(path.read_text(encoding="utf-8"))
    validate_router_model(model)
    return model


def route_vector(model: Mapping[str, Any], vector: Mapping[str, float]) -> dict[str, Any]:
    features = query_features(vector)
    policy_index = predict_policy_index(model["tree"], features)
    policies = model["policies"]
    if not 0 <= policy_index < len(policies):
        raise ValueError(f"Router predicted invalid policy index {policy_index}")
    policy = policies[policy_index]
    return {
        "policy_index": policy_index,
        "policy": str(policy["name"]),
        "parameters": dict(policy.get("parameters", {})),
        "features": features,
    }
