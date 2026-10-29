from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_run(path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.split()
            if len(fields) == 3:
                query_id, document_id, rank = fields
            elif len(fields) >= 6:
                query_id, _, document_id, rank = fields[:4]
            else:
                raise ValueError(f"Invalid run row {line_number} in {path}")
            rows[str(query_id)].append((int(rank), str(document_id)))

    rankings = {}
    for query_id, ranking in rows.items():
        seen = set()
        documents = []
        for _, document_id in sorted(ranking):
            if document_id not in seen:
                documents.append(document_id)
                seen.add(document_id)
        rankings[query_id] = documents
    return rankings


def _dcg(relevances: list[int]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 1) for rank, relevance in enumerate(relevances, 1)
    )


def query_metrics(
    judged: dict[str, int],
    ranking: list[str],
    cutoffs: list[int],
    map_cutoff: int,
) -> dict[str, float]:
    relevant = {document_id for document_id, relevance in judged.items() if relevance > 0}
    output: dict[str, float] = {}
    for cutoff in cutoffs:
        retrieved = ranking[:cutoff]
        hits = [document_id for document_id in retrieved if document_id in relevant]
        reciprocal_rank = next(
            (
                1.0 / rank
                for rank, document_id in enumerate(retrieved, 1)
                if document_id in relevant
            ),
            0.0,
        )
        gains = [max(0, judged.get(document_id, 0)) for document_id in retrieved]
        ideal = sorted((max(0, value) for value in judged.values()), reverse=True)[:cutoff]
        ideal_dcg = _dcg(ideal)
        output[f"mrr_at_{cutoff}"] = reciprocal_rank
        output[f"recall_at_{cutoff}"] = len(hits) / max(len(relevant), 1)
        output[f"precision_at_{cutoff}"] = len(hits) / cutoff
        output[f"success_at_{cutoff}"] = float(bool(hits))
        output[f"ndcg_at_{cutoff}"] = _dcg(gains) / ideal_dcg if ideal_dcg else 0.0

    hits = 0
    precision_sum = 0.0
    for rank, document_id in enumerate(ranking[:map_cutoff], 1):
        if document_id in relevant:
            hits += 1
            precision_sum += hits / rank
    output[f"map_at_{map_cutoff}"] = precision_sum / max(len(relevant), 1)
    return output


def evaluate(
    qrels: dict[str, dict[str, int]],
    ranking: dict[str, list[str]],
    config: dict[str, Any],
    per_query_path: Path | None = None,
) -> dict[str, float | int]:
    cutoffs = sorted({int(value) for value in config["cutoffs"]})
    for key in (
        "mrr_cutoff",
        "ndcg_cutoff",
        "precision_cutoff",
        "recall_cutoff",
    ):
        cutoffs.append(int(config[key]))
    cutoffs = sorted(set(cutoffs))
    map_cutoff = int(config["map_cutoff"])

    totals: dict[str, float] = defaultdict(float)
    query_rows = []
    for query_id, judged in qrels.items():
        values = query_metrics(judged, ranking.get(query_id, []), cutoffs, map_cutoff)
        for name, value in values.items():
            totals[name] += value
        query_rows.append({"query_id": query_id, **values})

    query_count = len(qrels)
    aggregate: dict[str, float | int] = {
        "queries": query_count,
        "queries_ranked": len(set(qrels).intersection(ranking)),
    }
    aggregate.update({name: value / max(query_count, 1) for name, value in sorted(totals.items())})

    # Stable headline names, even when the full cutoff grid changes.
    aggregate["mrr"] = aggregate[f"mrr_at_{int(config['mrr_cutoff'])}"]
    aggregate["ndcg"] = aggregate[f"ndcg_at_{int(config['ndcg_cutoff'])}"]
    aggregate["map"] = aggregate[f"map_at_{map_cutoff}"]
    aggregate["precision"] = aggregate[f"precision_at_{int(config['precision_cutoff'])}"]
    aggregate["recall"] = aggregate[f"recall_at_{int(config['recall_cutoff'])}"]

    if per_query_path is not None and bool(config.get("save_per_query", True)):
        per_query_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = per_query_path.with_suffix(per_query_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as output:
            for row in query_rows:
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(per_query_path)
    return aggregate
