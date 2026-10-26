"""Compute the standard MS MARCO retrieval metric grid from a run file."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_RUN = Path(__file__).resolve().parent.parent / "outputs" / "baseline" / "run.dev.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels-name", default="msmarco-passage-dev-subset")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_builtin_qrels(name: str) -> dict[str, set[str]]:
    from pyserini.search import get_qrels, get_topics

    qrels = get_qrels(name)
    topic_ids = {str(query_id) for query_id in get_topics(name)}
    return {
        str(query_id): {str(doc_id) for doc_id, relevance in docs.items() if int(relevance) > 0}
        for query_id, docs in qrels.items()
        if str(query_id) in topic_ids
    }


def load_run(path: Path) -> dict[str, list[str]]:
    ranking: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.split()
            if len(fields) == 3:  # MS MARCO: qid, docid, rank
                query_id, doc_id, rank = fields
            elif len(fields) >= 6:  # TREC: qid, Q0, docid, rank, score, tag
                query_id, _, doc_id, rank = fields[:4]
            else:
                raise ValueError(f"Invalid run row {line_number}: {line.rstrip()}")
            ranking[query_id].append((int(rank), doc_id))
    return {query_id: [doc_id for _, doc_id in sorted(rows)] for query_id, rows in ranking.items()}


def evaluate(
    relevant: dict[str, set[str]], ranking: dict[str, list[str]]
) -> dict[str, float | int]:
    # Keep this compatibility function for the older benchmark scripts, but
    # calculate the same complete metric grid as the unified pipeline.
    from src.metrics.retrieval import evaluate as evaluate_metrics

    qrels = {
        query_id: {document_id: 1 for document_id in document_ids}
        for query_id, document_ids in relevant.items()
    }
    return evaluate_metrics(
        qrels,
        ranking,
        {
            "cutoffs": [10, 100, 1000],
            "mrr_cutoff": 10,
            "ndcg_cutoff": 10,
            "map_cutoff": 1000,
            "precision_cutoff": 10,
            "recall_cutoff": 1000,
            "save_per_query": False,
        },
    )


def main() -> None:
    args = parse_args()
    relevant = load_builtin_qrels(args.qrels_name)
    metrics = evaluate(relevant, load_run(args.run))
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
