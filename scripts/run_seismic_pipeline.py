"""Build and evaluate a SEISMIC index from the existing SPLADE JSONL shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.evaluate_run import evaluate, load_builtin_qrels, load_run

EXPECTED_MSMARCO_DOCUMENTS = 8_841_823
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors-dir", type=Path, required=True)
    parser.add_argument(
        "--queries", type=Path, required=True, help="Pretokenized Pyserini TSV queries."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-docs", type=int, default=EXPECTED_MSMARCO_DOCUMENTS)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--search-batch-size", type=int, default=64)
    parser.add_argument("--hits", type=int, default=1_000)
    parser.add_argument("--n-postings", type=int, default=3_000)
    parser.add_argument("--centroid-fraction", type=float, default=0.2)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--summary-energy", type=float, default=0.5)
    parser.add_argument("--max-fraction", type=float, default=6.0)
    parser.add_argument("--doc-cut", type=int, default=15)
    parser.add_argument("--query-cut", type=int, default=10)
    parser.add_argument("--heap-factor", type=float, default=0.8)
    parser.add_argument("--first-sorted", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("all", "build", "search", "evaluate"),
        default="all",
    )
    parser.add_argument(
        "--temporary-jsonl",
        type=Path,
        default=Path("/kaggle/temp/seismic-documents.jsonl"),
    )
    parser.add_argument("--keep-temporary-jsonl", action="store_true")
    return parser.parse_args()


def vector_shards(path: Path) -> list[Path]:
    shards = sorted(item for item in path.glob("*.jsonl") if not item.name.endswith(".tmp"))
    if not shards:
        raise FileNotFoundError(f"No JSONL vector shards in {path}")
    return shards


def merge_shards(shards: list[Path], destination: Path) -> tuple[int, int]:
    """Concatenate shards without parsing or holding their contents in memory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    documents = 0
    with temporary.open("wb") as output:
        for shard_number, shard in enumerate(shards, 1):
            with shard.open("rb") as source:
                while block := source.read(16 * 1024 * 1024):
                    documents += block.count(b"\n")
                    output.write(block)
            if shard_number % 100 == 0 or shard_number == len(shards):
                print(
                    f"Merged {shard_number}/{len(shards)} shards; documents={documents:,}",
                    flush=True,
                )
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)
    return documents, destination.stat().st_size


def read_queries(path: Path) -> list[tuple[str, dict[str, float]]]:
    queries = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                query_id, weighted_tokens = line.rstrip("\n").split("\t", 1)
            except ValueError as error:
                raise ValueError(f"Invalid query row {line_number} in {path}") from error
            token_counts = Counter(weighted_tokens.split())
            queries.append(
                (query_id, {token: float(value) for token, value in token_counts.items()})
            )
    if not queries:
        raise ValueError(f"No queries in {path}")
    return queries


def chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def search(
    index,
    queries: list[tuple[str, dict[str, float]]],
    run_path: Path,
    args: argparse.Namespace,
) -> dict:
    from seismic import get_seismic_string

    string_type = get_seismic_string()
    run_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = run_path.with_suffix(run_path.suffix + ".tmp")
    started = time.perf_counter()
    result_count = 0
    with temporary.open("w", encoding="utf-8") as output:
        for batch_number, batch in enumerate(chunks(queries, args.search_batch_size), 1):
            query_ids = np.asarray([query_id for query_id, _ in batch], dtype=string_type)
            components = [np.asarray(list(vector), dtype=string_type) for _, vector in batch]
            values = [np.asarray(list(vector.values()), dtype=np.float32) for _, vector in batch]
            results = index.batch_search(
                queries_ids=query_ids,
                query_components=components,
                query_values=values,
                k=args.hits,
                query_cut=args.query_cut,
                heap_factor=args.heap_factor,
                sorted=args.first_sorted,
                n_knn=0,
                num_threads=args.threads,
            )
            for query_results in results:
                for rank, (query_id, _score, document_id) in enumerate(query_results, 1):
                    output.write(f"{query_id}\t{document_id}\t{rank}\n")
                    result_count += 1
            print(
                f"Searched {min(batch_number * args.search_batch_size, len(queries))}/{len(queries)} queries",
                flush=True,
            )
    temporary.replace(run_path)
    seconds = time.perf_counter() - started
    return {
        "queries": len(queries),
        "results": result_count,
        "seconds": seconds,
        "queries_per_second": len(queries) / max(seconds, 1e-9),
        "mean_query_ms_parallel_wall": seconds * 1000 / len(queries),
    }


def main() -> None:
    args = parse_args()
    if args.threads < 1 or args.search_batch_size < 1 or args.hits < 1:
        raise ValueError("threads, search-batch-size and hits must be positive")

    seismic_dir = args.output_dir / "seismic"
    index_prefix = seismic_dir / "msmarco"
    index_path = Path(f"{index_prefix}.index.seismic")
    run_path = seismic_dir / "run.dev.tsv"
    metrics_path = seismic_dir / "retrieval_metrics.json"
    pipeline_path = seismic_dir / "pipeline_metrics.json"
    seismic_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, object] = {}
    if pipeline_path.exists():
        timings = json.loads(pipeline_path.read_text(encoding="utf-8"))
    timings["parameters"] = {
        "n_postings": args.n_postings,
        "centroid_fraction": args.centroid_fraction,
        "min_cluster_size": args.min_cluster_size,
        "summary_energy": args.summary_energy,
        "max_fraction": args.max_fraction,
        "doc_cut": args.doc_cut,
        "query_cut": args.query_cut,
        "heap_factor": args.heap_factor,
        "first_sorted": args.first_sorted,
        "hits": args.hits,
        "threads": args.threads,
    }

    index = None
    if args.stage in ("all", "build"):
        from seismic import SeismicIndex

        if index_path.exists():
            print(f"Skipping existing SEISMIC index: {index_path}", flush=True)
            index = SeismicIndex.load(str(index_path))
        else:
            shards = vector_shards(args.vectors_dir)
            if (
                args.temporary_jsonl == Path("/kaggle/temp/seismic-documents.jsonl")
                and not Path("/kaggle/temp").exists()
            ):
                args.temporary_jsonl = Path(tempfile.gettempdir()) / "seismic-documents.jsonl"
            merge_started = time.perf_counter()
            documents, merged_bytes = merge_shards(shards, args.temporary_jsonl)
            timings["merge_seconds"] = time.perf_counter() - merge_started
            timings["vector_shards"] = len(shards)
            timings["documents"] = documents
            timings["merged_jsonl_bytes"] = merged_bytes
            if args.expected_docs and documents != args.expected_docs:
                raise RuntimeError(
                    f"Merged {documents:,} documents, expected {args.expected_docs:,}; "
                    "refusing to build an incomplete index"
                )
            try:
                build_started = time.perf_counter()
                index = SeismicIndex.build(
                    str(args.temporary_jsonl),
                    n_postings=args.n_postings,
                    centroid_fraction=args.centroid_fraction,
                    min_cluster_size=args.min_cluster_size,
                    summary_energy=args.summary_energy,
                    max_fraction=args.max_fraction,
                    doc_cut=args.doc_cut,
                    nknn=0,
                    load_content=False,
                    num_threads=args.threads,
                )
                timings["build_seconds"] = time.perf_counter() - build_started
                index.save(str(index_prefix))
            finally:
                if not args.keep_temporary_jsonl:
                    args.temporary_jsonl.unlink(missing_ok=True)
        timings["index_bytes"] = index_path.stat().st_size
        timings["index_documents"] = index.len
        timings["index_dimensions"] = index.dim
        timings["index_non_zero"] = index.nnz

    if args.stage in ("all", "search"):
        from seismic import SeismicIndex

        if not index_path.exists():
            raise FileNotFoundError(index_path)
        if index is None:
            index = SeismicIndex.load(str(index_path))
        timings["search"] = search(index, read_queries(args.queries), run_path, args)

    if args.stage in ("all", "evaluate"):
        if not run_path.exists():
            raise FileNotFoundError(run_path)
        metrics = evaluate(load_builtin_qrels("msmarco-passage-dev-subset"), load_run(run_path))
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2), flush=True)

    pipeline_path.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")
    print(f"SEISMIC artifacts: {seismic_dir}", flush=True)


if __name__ == "__main__":
    main()
