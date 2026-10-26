"""Compute MS MARCO MRR@10 and macro Recall@1000 from a run file."""

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
    from pyserini.search import get_qrels

    qrels = get_qrels(name)
    return {
        str(query_id): {str(doc_id) for doc_id, relevance in docs.items() if int(relevance) > 0}
        for query_id, docs in qrels.items()
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


def evaluate(relevant: dict[str, set[str]], ranking: dict[str, list[str]]) -> dict[str, float | int]:
    reciprocal_ranks = []
    recalls = []
    for query_id, relevant_docs in relevant.items():
        retrieved = ranking.get(query_id, [])
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(retrieved[:10], 1):
            if doc_id in relevant_docs:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)
        recalls.append(len(relevant_docs.intersection(retrieved[:1000])) / len(relevant_docs))
    return {
        "queries": len(relevant),
        "queries_ranked": len(set(relevant).intersection(ranking)),
        "mrr_at_10": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "recall_at_1000": sum(recalls) / len(recalls),
    }


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
