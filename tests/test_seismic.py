from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("seismic")

from src.index.seismic import build_index, search_index


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_build_and_search_tiny_index(tmp_path) -> None:
    documents = tmp_path / "documents.jsonl"
    queries = tmp_path / "queries.jsonl"
    _write_jsonl(
        documents,
        [
            {"id": "d1", "vector": {"cat": 2.0, "pet": 1.0}},
            {"id": "d2", "vector": {"dog": 2.0, "pet": 1.0}},
            {"id": "d3", "vector": {"car": 3.0}},
        ],
    )
    _write_jsonl(queries, [{"id": "q1", "vector": {"cat": 1.0}}])
    prefix = tmp_path / "index" / "tiny"
    build = build_index(
        documents,
        prefix,
        {
            "variant": "standard",
            "n_postings": 10,
            "centroid_fraction": 0.2,
            "min_cluster_size": 2,
            "summary_energy": 0.5,
            "max_fraction": 6.0,
            "doc_cut": 15,
            "nknn": 0,
            "knn_path": None,
            "batched_indexing": None,
            "load_content": False,
            "threads": 1,
            "expected_documents": 3,
        },
        temporary_dir=tmp_path,
    )
    assert build["documents"] == 3

    run = tmp_path / "run.trec"
    search = search_index(
        Path(build["index_file"]),
        queries,
        run,
        "standard",
        {
            "top_k": 2,
            "query_cut": 10,
            "heap_factor": 0.8,
            "n_knn": 0,
            "first_sorted": False,
            "batch_size": 1,
            "warmup_queries": 0,
            "threads": 1,
            "run_name": "test",
        },
    )
    assert search["queries"] == 1
    assert run.read_text(encoding="utf-8").split()[2] == "d1"
