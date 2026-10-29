from __future__ import annotations

import json
import sys

from scripts import train_query_router
from src.index.query_router import load_router_model


def test_train_query_router_end_to_end_with_file_qrels(tmp_path, monkeypatch) -> None:
    query_ids = [str(value) for value in range(100, 200)]
    vectors = tmp_path / "queries.jsonl"
    vectors.write_text(
        "".join(
            json.dumps(
                {
                    "id": query_id,
                    "vector": {
                        f"t{index}": 1.0 for index in range(1 + int(query_id) % 4)
                    },
                }
            )
            + "\n"
            for query_id in query_ids
        ),
        encoding="utf-8",
    )
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text(
        "".join(f"{query_id}\t0\trel-{query_id}\t1\n" for query_id in query_ids),
        encoding="utf-8",
    )
    fast_run = tmp_path / "fast.trec"
    safe_run = tmp_path / "safe.trec"
    fast_rows = []
    safe_rows = []
    for query_id in query_ids:
        fast_documents = (
            [f"rel-{query_id}", f"other-{query_id}"]
            if int(query_id) % 2 == 0
            else [f"miss-{query_id}", f"other-{query_id}"]
        )
        safe_documents = [f"rel-{query_id}", f"other-{query_id}"]
        for rank, document_id in enumerate(fast_documents, 1):
            fast_rows.append(f"{query_id}\tQ0\t{document_id}\t{rank}\t1.0\tfast\n")
        for rank, document_id in enumerate(safe_documents, 1):
            safe_rows.append(f"{query_id}\tQ0\t{document_id}\t{rank}\t1.0\tsafe\n")
    fast_run.write_text("".join(fast_rows), encoding="utf-8")
    safe_run.write_text("".join(safe_rows), encoding="utf-8")

    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "policies": [
                    {
                        "name": "fast",
                        "parameters": {"query_cut": 6},
                        "run": str(fast_run),
                        "search_seconds": 6.0,
                    },
                    {
                        "name": "safe",
                        "parameters": {"query_cut": 10},
                        "run": str(safe_run),
                        "search_seconds": 10.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    model = tmp_path / "router.json"
    report = tmp_path / "training_report.json"
    assignments = tmp_path / "assignments.jsonl"
    test_qrels = tmp_path / "test.qrels.tsv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_query_router.py",
            "--plan",
            str(plan),
            "--query-vectors",
            str(vectors),
            "--output-model",
            str(model),
            "--output-report",
            str(report),
            "--output-assignments",
            str(assignments),
            "--output-test-qrels",
            str(test_qrels),
            "--qrels-source",
            "file",
            "--qrels",
            str(qrels),
            "--min-samples-leaf",
            "4",
        ],
    )

    train_query_router.main()

    trained = load_router_model(model)
    training_report = json.loads(report.read_text(encoding="utf-8"))
    assert len(trained["policies"]) == 2
    assert sum(training_report["splits"].values()) == len(query_ids)
    assert len(assignments.read_text(encoding="utf-8").splitlines()) == len(query_ids)
    assert test_qrels.read_text(encoding="utf-8")
