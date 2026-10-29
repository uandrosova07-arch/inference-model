"""Train a tiny query-adaptive SEISMIC policy router from fixed-policy runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.datasets import load_qrels
from src.index.query_router import (
    FEATURE_NAMES,
    feature_matrix,
    predict_policy_index,
    query_features,
    train_router_tree,
    validate_router_model,
)
from src.metrics import query_metrics
from src.utils.io import atomic_write_json, atomic_write_text, file_fingerprint, jsonl_lines

EVALUATION_CONFIG = {
    "cutoffs": [10, 100, 1000],
    "mrr_cutoff": 10,
    "ndcg_cutoff": 10,
    "map_cutoff": 1000,
    "precision_cutoff": 10,
    "recall_cutoff": 1000,
    "save_per_query": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--query-vectors", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-assignments", type=Path, required=True)
    parser.add_argument("--output-test-qrels", type=Path, required=True)
    parser.add_argument("--qrels-source", choices=("pyserini", "file"), default="pyserini")
    parser.add_argument("--qrels", type=Path, default=Path("unused-qrels.tsv"))
    parser.add_argument("--pyserini-qrels", default="msmarco-passage-dev-subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=64)
    parser.add_argument("--leaf-safety-quantile", type=float, default=0.95)
    parser.add_argument(
        "--quality-gate",
        action="append",
        default=[],
        metavar="METRIC=TOLERANCE",
        help="Per-query loss allowed relative to the reference policy; repeatable.",
    )
    return parser.parse_args()


def _read_vectors(path: Path) -> dict[str, dict[str, float]]:
    records: dict[str, dict[str, float]] = {}
    for line_number, line in enumerate(jsonl_lines(path), 1):
        record = json.loads(line)
        if "id" not in record or not isinstance(record.get("vector"), dict):
            raise ValueError(f"Invalid query vector row {line_number} in {path}")
        records[str(record["id"])] = {
            str(component): float(value) for component, value in record["vector"].items()
        }
    if not records:
        raise ValueError(f"No query vectors in {path}")
    return records


def _split(query_id: str, seed: int, train_fraction: float, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{query_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def _parse_gates(values: list[str]) -> dict[str, float]:
    if not values:
        return {"recall_at_1000": 0.0, "mrr_at_10": 0.0}
    gates = {}
    for value in values:
        metric, separator, tolerance = value.partition("=")
        if not separator or not metric:
            raise ValueError(f"Invalid --quality-gate {value!r}; expected METRIC=TOLERANCE")
        gates[metric] = float(tolerance)
        if gates[metric] < 0:
            raise ValueError("Quality tolerances must be non-negative")
    return gates


def _search_seconds(policy: dict[str, Any]) -> float | None:
    configured = policy.get("search_seconds")
    if configured is not None:
        return float(configured)
    report_path = policy.get("report")
    if not report_path or not Path(report_path).is_file():
        return None
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    search = report.get("stages", {}).get("search", {})
    value = search.get("search_seconds")
    return float(value) if value is not None else None


def _run_query_metrics(
    path: Path,
    qrels: dict[str, dict[str, int]],
    cutoffs: list[int],
) -> dict[str, dict[str, float]]:
    """Stream a query-grouped TREC run without retaining millions of document IDs."""
    output: dict[str, dict[str, float]] = {}
    current_query: str | None = None
    documents: list[str] = []
    seen_documents: set[str] = set()

    def finish() -> None:
        if current_query is not None and current_query in qrels:
            output[current_query] = query_metrics(
                qrels[current_query], documents, cutoffs, int(EVALUATION_CONFIG["map_cutoff"])
            )

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.split()
            if len(fields) == 3:
                query_id, document_id, _ = fields
            elif len(fields) >= 6:
                query_id, _, document_id = fields[:3]
            else:
                raise ValueError(f"Invalid run row {line_number} in {path}")
            query_id = str(query_id)
            if current_query is None:
                current_query = query_id
            elif query_id != current_query:
                finish()
                if query_id in output:
                    raise ValueError(f"Run is not grouped by query ID: {query_id} appears twice")
                current_query = query_id
                documents = []
                seen_documents = set()
            if len(documents) < max(1000, max(cutoffs)) and document_id not in seen_documents:
                documents.append(str(document_id))
                seen_documents.add(str(document_id))
    finish()
    return output


def _aggregate_query_metrics(
    query_ids: list[str],
    rows: dict[str, dict[str, float]],
) -> dict[str, float | int]:
    totals: dict[str, float] = {}
    for query_id in query_ids:
        for name, value in rows.get(query_id, {}).items():
            totals[name] = totals.get(name, 0.0) + float(value)
    count = len(query_ids)
    aggregate: dict[str, float | int] = {
        "queries": count,
        "queries_ranked": sum(query_id in rows for query_id in query_ids),
        **{name: value / max(count, 1) for name, value in sorted(totals.items())},
    }
    aggregate["mrr"] = aggregate["mrr_at_10"]
    aggregate["ndcg"] = aggregate["ndcg_at_10"]
    aggregate["map"] = aggregate["map_at_1000"]
    aggregate["precision"] = aggregate["precision_at_10"]
    aggregate["recall"] = aggregate["recall_at_1000"]
    return aggregate


def main() -> None:
    args = parse_args()
    if not 0 < args.train_fraction < 1:
        raise ValueError("train-fraction must be in (0, 1)")
    if not 0 < args.validation_fraction < 1 - args.train_fraction:
        raise ValueError("validation-fraction must leave a non-empty test fraction")

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    policies = plan.get("policies")
    if not isinstance(policies, list) or len(policies) < 2:
        raise ValueError("Training plan must contain at least two policies, fastest first")
    names = [str(policy["name"]) for policy in policies]
    if len(set(names)) != len(names):
        raise ValueError("Policy names must be unique")
    for policy in policies:
        run_path = Path(policy["run"])
        if not run_path.is_file():
            raise FileNotFoundError(run_path)

    vectors = _read_vectors(args.query_vectors)
    qrels = load_qrels(args.qrels_source, args.qrels, args.pyserini_qrels)
    query_ids = sorted(set(vectors).intersection(qrels))
    if not query_ids:
        raise ValueError("No query IDs overlap between vectors and qrels")
    gates = _parse_gates(args.quality_gate)
    cutoffs = sorted(
        {10, 100, 1000, *[int(metric.rsplit("_", 1)[-1]) for metric in gates]}
    )
    reference_index = len(policies) - 1

    feature_rows = {query_id: query_features(vectors[query_id]) for query_id in query_ids}
    per_policy = {
        name: _run_query_metrics(Path(policy["run"]), qrels, cutoffs)
        for name, policy in zip(names, policies)
    }
    missing = {
        name: len(set(query_ids).difference(rows)) for name, rows in per_policy.items()
    }
    if any(missing.values()):
        raise ValueError(f"Policy runs are missing configured queries: {missing}")

    oracle_labels: dict[str, int] = {}
    for query_id in query_ids:
        reference = per_policy[names[reference_index]][query_id]
        chosen = reference_index
        for policy_index, name in enumerate(names):
            candidate = per_policy[name][query_id]
            if all(
                candidate[metric] >= reference[metric] - tolerance
                for metric, tolerance in gates.items()
            ):
                chosen = policy_index
                break
        oracle_labels[query_id] = chosen

    splits = {
        query_id: _split(
            query_id, args.seed, args.train_fraction, args.validation_fraction
        )
        for query_id in query_ids
    }
    split_ids = {
        split: [query_id for query_id in query_ids if splits[query_id] == split]
        for split in ("train", "validation", "test")
    }
    if any(not values for values in split_ids.values()):
        raise ValueError("Hash split unexpectedly produced an empty partition")

    train_ids = split_ids["train"]
    tree = train_router_tree(
        feature_matrix([feature_rows[query_id] for query_id in train_ids]),
        [oracle_labels[query_id] for query_id in train_ids],
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        safety_quantile=args.leaf_safety_quantile,
    )
    predicted = {
        query_id: predict_policy_index(tree, feature_rows[query_id]) for query_id in query_ids
    }

    policy_models = [
        {"name": name, "parameters": dict(policy.get("parameters", {}))}
        for name, policy in zip(names, policies)
    ]
    model = {
        "schema_version": 1,
        "model_type": "ordinal_cart",
        "feature_names": list(FEATURE_NAMES),
        "policies": policy_models,
        "tree": tree,
        "training": {
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "test_fraction": 1 - args.train_fraction - args.validation_fraction,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "leaf_safety_quantile": args.leaf_safety_quantile,
            "quality_gates": gates,
            "reference_policy": names[reference_index],
            "queries": len(query_ids),
            "query_vectors": {
                "path": str(args.query_vectors),
                **file_fingerprint(args.query_vectors),
            },
            "policy_runs": [
                {
                    "name": name,
                    "path": str(policy["run"]),
                    **file_fingerprint(Path(policy["run"])),
                }
                for name, policy in zip(names, policies)
            ],
        },
    }
    validate_router_model(model)

    policy_seconds = {name: _search_seconds(policy) for name, policy in zip(names, policies)}
    report: dict[str, Any] = {
        "schema_version": 1,
        "model": str(args.output_model),
        "query_count": len(query_ids),
        "splits": {name: len(values) for name, values in split_ids.items()},
        "quality_gates": gates,
        "policies": [
            {
                "name": name,
                "parameters": policy.get("parameters", {}),
                "search_seconds": policy_seconds[name],
            }
            for name, policy in zip(names, policies)
        ],
        "partitions": {},
    }
    for split, ids in split_ids.items():
        router_rows = {
            query_id: per_policy[names[predicted[query_id]]][query_id] for query_id in ids
        }
        oracle_rows = {
            query_id: per_policy[names[oracle_labels[query_id]]][query_id]
            for query_id in ids
        }
        predicted_counts = Counter(names[predicted[query_id]] for query_id in ids)
        estimated_seconds = None
        if all(policy_seconds[name] is not None for name in names):
            estimated_seconds = sum(
                predicted_counts[name] / len(ids) * float(policy_seconds[name]) for name in names
            )
        violations = 0
        for query_id in ids:
            reference = per_policy[names[reference_index]][query_id]
            selected = per_policy[names[predicted[query_id]]][query_id]
            if any(
                selected[metric] < reference[metric] - tolerance
                for metric, tolerance in gates.items()
            ):
                violations += 1
        report["partitions"][split] = {
            "queries": len(ids),
            "oracle_policy_distribution": dict(
                Counter(names[oracle_labels[query_id]] for query_id in ids)
            ),
            "router_policy_distribution": dict(predicted_counts),
            "router_matches_oracle": sum(
                predicted[query_id] == oracle_labels[query_id] for query_id in ids
            )
            / len(ids),
            "quality_gate_violations": violations,
            "quality_gate_violation_rate": violations / len(ids),
            "estimated_search_seconds": estimated_seconds,
            "fixed_policy_metrics": {
                name: _aggregate_query_metrics(ids, per_policy[name]) for name in names
            },
            "oracle_metrics": _aggregate_query_metrics(ids, oracle_rows),
            "router_metrics": _aggregate_query_metrics(ids, router_rows),
        }

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_model, model)
    atomic_write_json(args.output_report, report)
    assignment_lines = []
    for query_id in query_ids:
        assignment_lines.append(
            json.dumps(
                {
                    "query_id": query_id,
                    "split": splits[query_id],
                    "oracle_policy": names[oracle_labels[query_id]],
                    "predicted_policy": names[predicted[query_id]],
                    "features": feature_rows[query_id],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
    atomic_write_text(args.output_assignments, "".join(assignment_lines))
    test_qrels = []
    for query_id in split_ids["test"]:
        for document_id, relevance in qrels[query_id].items():
            test_qrels.append(f"{query_id}\t0\t{document_id}\t{relevance}\n")
    atomic_write_text(args.output_test_qrels, "".join(test_qrels))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
