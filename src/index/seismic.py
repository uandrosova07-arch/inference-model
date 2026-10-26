from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import directory_size, jsonl_lines, seismic_document_input

INDEX_SUFFIX = ".index.seismic"


def _index_class(variant: str):
    if variant == "standard":
        from seismic import SeismicIndex

        return SeismicIndex
    if variant == "large_vocab":
        from seismic import SeismicIndexLV

        return SeismicIndexLV
    raise ValueError(f"Unsupported SEISMIC variant={variant!r}")


def normalized_prefix(path: Path) -> Path:
    rendered = str(path)
    rendered = rendered.removesuffix(INDEX_SUFFIX)
    return Path(rendered)


def index_file(prefix: Path) -> Path:
    return Path(f"{normalized_prefix(prefix)}{INDEX_SUFFIX}")


def build_index(
    documents: Path,
    prefix: Path,
    config: dict[str, Any],
    temporary_dir: Path | None,
) -> dict[str, Any]:
    index_type = _index_class(str(config["variant"]))
    prefix = normalized_prefix(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with seismic_document_input(documents, temporary_dir) as input_path:
        index = index_type.build(
            str(input_path),
            n_postings=int(config["n_postings"]),
            centroid_fraction=float(config["centroid_fraction"]),
            min_cluster_size=int(config["min_cluster_size"]),
            summary_energy=float(config["summary_energy"]),
            max_fraction=float(config["max_fraction"]),
            doc_cut=int(config["doc_cut"]),
            nknn=int(config.get("nknn", 0)),
            knn_path=config.get("knn_path"),
            batched_indexing=config.get("batched_indexing"),
            load_content=bool(config.get("load_content", False)),
            num_threads=int(config.get("threads", 0)),
        )
        expected = config.get("expected_documents")
        if expected is not None and int(index.len) != int(expected):
            raise RuntimeError(
                f"SEISMIC indexed {int(index.len):,} documents, expected {int(expected):,}"
            )
        index.save(str(prefix))
    seconds = time.perf_counter() - started
    output_path = index_file(prefix)
    return {
        "seconds": seconds,
        "documents": int(index.len),
        "dimensions": int(index.dim),
        "non_zero_terms": int(index.nnz),
        "documents_per_second": int(index.len) / max(seconds, 1e-9),
        "index_bytes": output_path.stat().st_size,
        "index_file": str(output_path),
        "source_vector_bytes": directory_size(documents),
    }


def _chunks(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _read_queries(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(jsonl_lines(path), 1):
        record = json.loads(line)
        if "id" not in record or "vector" not in record:
            raise ValueError(f"Missing id/vector at query row {line_number} in {path}")
        records.append(record)
    if not records:
        raise ValueError(f"No query vectors in {path}")
    return records


def _prepare_queries(records: list[dict[str, Any]], string_type: str):
    prepared = []
    for record in records:
        vector = record["vector"]
        prepared.append(
            (
                str(record["id"]),
                np.asarray(list(vector), dtype=string_type),
                np.asarray(list(vector.values()), dtype=np.float32),
            )
        )
    return prepared


def _batch_search(index, batch, config: dict[str, Any]):
    from seismic import get_seismic_string

    string_type = get_seismic_string()
    return index.batch_search(
        queries_ids=np.asarray([query_id for query_id, _, _ in batch], dtype=string_type),
        query_components=[components for _, components, _ in batch],
        query_values=[values for _, _, values in batch],
        k=int(config["top_k"]),
        query_cut=int(config["query_cut"]),
        heap_factor=float(config["heap_factor"]),
        n_knn=int(config.get("n_knn", 0)),
        sorted=bool(config.get("first_sorted", False)),
        num_threads=int(config.get("threads", 0)),
    )


def search_index(
    index_path: Path,
    query_vectors: Path,
    run_path: Path,
    index_variant: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    from seismic import get_seismic_string

    index_type = _index_class(index_variant)
    load_started = time.perf_counter()
    index = index_type.load(str(index_path))
    load_seconds = time.perf_counter() - load_started

    prepare_started = time.perf_counter()
    records = _read_queries(query_vectors)
    prepared = _prepare_queries(records, get_seismic_string())
    prepare_seconds = time.perf_counter() - prepare_started

    warmup_count = min(int(config.get("warmup_queries", 0)), len(prepared))
    if warmup_count:
        _batch_search(index, prepared[:warmup_count], config)

    run_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = run_path.with_suffix(run_path.suffix + ".tmp")
    batch_size = int(config["batch_size"])
    batch_ms_per_query: list[float] = []
    result_count = 0
    started = time.perf_counter()
    with temporary.open("w", encoding="utf-8") as output:
        for batch_number, batch in enumerate(_chunks(prepared, batch_size), 1):
            batch_started = time.perf_counter()
            results = _batch_search(index, batch, config)
            batch_seconds = time.perf_counter() - batch_started
            batch_ms_per_query.extend([batch_seconds * 1000 / len(batch)] * len(batch))
            for query_results in results:
                for rank, (query_id, score, document_id) in enumerate(query_results, 1):
                    output.write(
                        f"{query_id}\tQ0\t{document_id}\t{rank}\t{float(score):.8f}\t"
                        f"{config['run_name']}\n"
                    )
                    result_count += 1
            completed = min(batch_number * batch_size, len(prepared))
            print(f"SEISMIC searched {completed}/{len(prepared)} queries", flush=True)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(run_path)
    seconds = time.perf_counter() - started
    return {
        "queries": len(prepared),
        "results": result_count,
        "index_load_seconds": load_seconds,
        "query_prepare_seconds": prepare_seconds,
        "search_seconds": seconds,
        "queries_per_second": len(prepared) / max(seconds, 1e-9),
        "mean_query_ms_parallel_wall": seconds * 1000 / len(prepared),
        "batch_amortized_query_ms_p50": _percentile(batch_ms_per_query, 0.50),
        "batch_amortized_query_ms_p95": _percentile(batch_ms_per_query, 0.95),
        "batch_amortized_query_ms_p99": _percentile(batch_ms_per_query, 0.99),
        "run_bytes": run_path.stat().st_size,
    }
