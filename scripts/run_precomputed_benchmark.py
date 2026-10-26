"""Benchmark Lucene impact retrieval and SEISMIC on precomputed MS MARCO vectors."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import gc
import json
import multiprocessing
import os
from pathlib import Path
import tarfile
import tempfile
import time

import numpy as np

from scripts.evaluate_run import evaluate, load_builtin_qrels, load_run


EXPECTED_DOCUMENTS = 8_841_823
EXPECTED_QUERIES = 6_980
WEIGHT_RANGE = 5.0
QUANT_RANGE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True, help="documents.tar.gz or JSONL")
    parser.add_argument("--queries", type=Path, required=True, help="queries.tar.gz or JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--search-batch-size", type=int, default=64)
    parser.add_argument("--hits", type=int, default=1_000)
    parser.add_argument("--expected-docs", type=int, default=EXPECTED_DOCUMENTS)
    parser.add_argument("--expected-queries", type=int, default=EXPECTED_QUERIES)
    parser.add_argument("--n-postings", type=int, default=3_000)
    parser.add_argument("--centroid-fraction", type=float, default=0.1)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--summary-energy", type=float, default=0.4)
    parser.add_argument("--max-fraction", type=float, default=6.0)
    parser.add_argument("--doc-cut", type=int, default=15)
    parser.add_argument("--query-cut", type=int, default=8)
    parser.add_argument("--heap-factor", type=float, default=0.7)
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "baseline-index",
            "baseline-search",
            "seismic-build",
            "seismic-search",
            "evaluate",
        ),
        default="all",
    )
    return parser.parse_args()


def files_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def first_tar_member(archive: tarfile.TarFile):
    for member in archive:
        name = Path(member.name).name
        if member.isfile() and not name.startswith("._") and "__MACOSX" not in Path(member.name).parts:
            return member
    raise ValueError(f"No regular file in {archive.name}")


def jsonl_lines(path: Path) -> Iterator[str]:
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            member = first_tar_member(archive)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read {member.name} from {path}")
            for raw_line in source:
                yield raw_line.decode("utf-8")
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            yield from source
    else:
        raise ValueError(f"Expected .tar.gz or .jsonl: {path}")


def read_vectors(path: Path) -> list[dict]:
    return [json.loads(line) for line in jsonl_lines(path)]


def quantize_vector(vector: dict[str, float]) -> dict[str, int]:
    quantized = {}
    for token, value in vector.items():
        weight = round(float(value) / WEIGHT_RANGE * QUANT_RANGE)
        if weight > 0:
            quantized[token] = weight
    return quantized


def weighted_tokens(vector: dict[str, int]) -> str:
    return " ".join(token for token, weight in vector.items() for _ in range(weight))


def chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_baseline_index(documents_path: Path, index_dir: Path, args: argparse.Namespace) -> dict:
    from pyserini.index.lucene import LuceneIndexer
    from pyserini.pyclass import autoclass

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    indexer = LuceneIndexer(
        args=["-index", str(index_dir), "-impact", "-pretokenized"],
        threads=args.threads,
    )
    json_vector_document = autoclass("io.anserini.collection.JsonVectorCollection$Document")
    started = time.perf_counter()
    documents = 0
    accepted_documents = 0
    non_zero = 0
    indexed_impacts = 0
    batch = []
    try:
        for line in jsonl_lines(documents_path):
            record = json.loads(line)
            vector = quantize_vector(record["vector"])
            non_zero += len(vector)
            indexed_impacts += sum(vector.values())
            node = indexer.mapper.readTree(
                json.dumps({"id": str(record["id"]), "vector": vector}, separators=(",", ":"))
            )
            batch.append(json_vector_document(node))
            documents += 1
            if len(batch) == args.batch_size:
                accepted_documents += indexer.object.addJsonDocuments(batch)
                batch.clear()
            if documents % 100_000 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"Lucene indexed {documents:,} documents "
                    f"({documents / max(elapsed, 1e-9):,.0f} docs/s)",
                    flush=True,
                )
        if batch:
            accepted_documents += indexer.object.addJsonDocuments(batch)
    finally:
        indexer.close()
    seconds = time.perf_counter() - started
    if args.expected_docs and documents != args.expected_docs:
        raise RuntimeError(f"Indexed {documents:,} documents, expected {args.expected_docs:,}")
    if accepted_documents != documents:
        raise RuntimeError(f"Lucene accepted {accepted_documents:,}/{documents:,} documents")
    return {
        "seconds": seconds,
        "documents": documents,
        "accepted_documents": accepted_documents,
        "documents_per_second": documents / max(seconds, 1e-9),
        "non_zero_terms_before_impact_expansion": non_zero,
        "indexed_impact_tokens": indexed_impacts,
        "index_bytes": files_size(index_dir),
    }


def prepare_queries(query_records: list[dict], destination: Path) -> dict:
    started = time.perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in query_records:
            vector = quantize_vector(record["vector"])
            output.write(f"{record['id']}\t{weighted_tokens(vector)}\n")
    temporary.replace(destination)
    return {"seconds": time.perf_counter() - started, "queries": len(query_records)}


def baseline_search(
    index_dir: Path,
    queries_path: Path,
    run_path: Path,
    query_count: int,
    args: argparse.Namespace,
) -> dict:
    from pyserini.search.lucene import LuceneImpactSearcher

    query_ids = []
    queries = []
    with queries_path.open(encoding="utf-8") as source:
        for line in source:
            query_id, query = line.rstrip("\n").split("\t", 1)
            query_ids.append(query_id)
            queries.append(query)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    searcher = LuceneImpactSearcher(str(index_dir), None)
    index_load_seconds = time.perf_counter() - load_started
    temporary = run_path.with_suffix(run_path.suffix + ".tmp")
    started = time.perf_counter()
    result_count = 0
    with temporary.open("w", encoding="utf-8") as output:
        paired_queries = list(zip(query_ids, queries, strict=True))
        for batch_number, batch in enumerate(chunks(paired_queries, args.search_batch_size), 1):
            batch_ids = [query_id for query_id, _ in batch]
            batch_queries = [query for _, query in batch]
            results = searcher.batch_search(
                queries=batch_queries,
                qids=batch_ids,
                k=args.hits,
                threads=args.threads,
            )
            for query_id in batch_ids:
                for rank, hit in enumerate(results.get(query_id, []), 1):
                    output.write(f"{query_id}\t{hit.docid}\t{rank}\n")
                    result_count += 1
            completed = min(batch_number * args.search_batch_size, query_count)
            print(f"Lucene searched {completed}/{query_count} queries", flush=True)
    temporary.replace(run_path)
    seconds = time.perf_counter() - started
    return {
        "seconds": seconds,
        "index_load_seconds": index_load_seconds,
        "queries": query_count,
        "results": result_count,
        "queries_per_second": query_count / max(seconds, 1e-9),
        "mean_query_ms_parallel_wall": seconds * 1000 / query_count,
    }


def feed_fifo_from_tar(archive_path: str, fifo_path: str, passes: int) -> None:
    """Feed SEISMIC's two JSONL passes without extracting the 33 GiB archive."""
    for pass_number in range(passes):
        with tarfile.open(archive_path, "r:gz") as archive:
            member = first_tar_member(archive)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read {member.name} from {archive_path}")
            with open(fifo_path, "wb", buffering=0) as output:
                while block := source.read(16 * 1024 * 1024):
                    output.write(block)
        # Give the reader time to observe EOF and reopen the FIFO for its next pass.
        if pass_number + 1 < passes:
            time.sleep(1.0)


def seismic_input(documents_path: Path) -> tuple[Path, multiprocessing.Process | None]:
    if not documents_path.name.endswith(".tar.gz"):
        return documents_path, None
    temporary_root = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path(tempfile.gettempdir())
    fifo_path = temporary_root / "seismic-precomputed-documents.jsonl"
    fifo_path.unlink(missing_ok=True)
    os.mkfifo(fifo_path)
    producer = multiprocessing.Process(
        target=feed_fifo_from_tar,
        args=(str(documents_path), str(fifo_path), 2),
        daemon=True,
    )
    producer.start()
    return fifo_path, producer


def build_seismic(documents_path: Path, index_prefix: Path, args: argparse.Namespace):
    from seismic import SeismicIndex

    input_path, producer = seismic_input(documents_path)
    index_prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        index = SeismicIndex.build(
            str(input_path),
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
        if producer is not None:
            producer.join()
            if producer.exitcode != 0:
                raise RuntimeError(f"Archive producer failed with exit code {producer.exitcode}")
        index.save(str(index_prefix))
    finally:
        if producer is not None and producer.is_alive():
            producer.terminate()
            producer.join()
        if producer is not None:
            input_path.unlink(missing_ok=True)
    seconds = time.perf_counter() - started
    if args.expected_docs and index.len != args.expected_docs:
        raise RuntimeError(f"SEISMIC indexed {index.len:,} documents, expected {args.expected_docs:,}")
    index_path = Path(f"{index_prefix}.index.seismic")
    return index, {
        "seconds": seconds,
        "documents": index.len,
        "documents_per_second": index.len / max(seconds, 1e-9),
        "dimensions": index.dim,
        "non_zero_terms": index.nnz,
        "index_bytes": index_path.stat().st_size,
    }


def seismic_search(index, query_records: list[dict], run_path: Path, args: argparse.Namespace) -> dict:
    from seismic import get_seismic_string

    string_type = get_seismic_string()
    prepare_started = time.perf_counter()
    prepared_queries = []
    for record in query_records:
        prepared_queries.append(
            (
                str(record["id"]),
                np.asarray(list(record["vector"]), dtype=string_type),
                np.asarray(list(record["vector"].values()), dtype=np.float32),
            )
        )
    query_prepare_seconds = time.perf_counter() - prepare_started
    temporary = run_path.with_suffix(run_path.suffix + ".tmp")
    result_count = 0
    started = time.perf_counter()
    with temporary.open("w", encoding="utf-8") as output:
        for batch_number, batch in enumerate(chunks(prepared_queries, args.search_batch_size), 1):
            query_ids = np.asarray([query_id for query_id, _, _ in batch], dtype=string_type)
            components = [query_components for _, query_components, _ in batch]
            values = [query_values for _, _, query_values in batch]
            results = index.batch_search(
                queries_ids=query_ids,
                query_components=components,
                query_values=values,
                k=args.hits,
                query_cut=args.query_cut,
                heap_factor=args.heap_factor,
                sorted=False,
                n_knn=0,
                num_threads=args.threads,
            )
            for query_results in results:
                for rank, (query_id, _score, document_id) in enumerate(query_results, 1):
                    output.write(f"{query_id}\t{document_id}\t{rank}\n")
                    result_count += 1
            completed = min(batch_number * args.search_batch_size, len(query_records))
            print(f"SEISMIC searched {completed}/{len(query_records)} queries", flush=True)
    temporary.replace(run_path)
    seconds = time.perf_counter() - started
    return {
        "seconds": seconds,
        "query_prepare_seconds": query_prepare_seconds,
        "queries": len(query_records),
        "results": result_count,
        "queries_per_second": len(query_records) / max(seconds, 1e-9),
        "mean_query_ms_parallel_wall": seconds * 1000 / len(query_records),
    }


def evaluate_run(run_path: Path) -> dict:
    return evaluate(load_builtin_qrels("msmarco-passage-dev-subset"), load_run(run_path))


def main() -> None:
    args = parse_args()
    if args.threads < 1 or args.batch_size < 1 or args.search_batch_size < 1:
        raise ValueError("threads and batch sizes must be positive")
    if not args.documents.exists() or not args.queries.exists():
        raise FileNotFoundError("Precomputed documents or queries archive is missing")

    output_dir = args.output_dir
    baseline_dir = output_dir / "baseline_lucene"
    seismic_dir = output_dir / "seismic"
    baseline_index_dir = baseline_dir / "index"
    baseline_queries = baseline_dir / "queries.dev.splade.tsv"
    baseline_run = baseline_dir / "run.dev.tsv"
    seismic_prefix = seismic_dir / "msmarco"
    seismic_index_path = Path(f"{seismic_prefix}.index.seismic")
    seismic_run = seismic_dir / "run.dev.tsv"
    report_path = output_dir / "benchmark_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source"] = {
        "documents": str(args.documents),
        "queries": str(args.queries),
        "documents_archive_bytes": args.documents.stat().st_size,
        "queries_archive_bytes": args.queries.stat().st_size,
    }
    report["parameters"] = {
        "threads": args.threads,
        "hits": args.hits,
        "n_postings": args.n_postings,
        "centroid_fraction": args.centroid_fraction,
        "min_cluster_size": args.min_cluster_size,
        "summary_energy": args.summary_energy,
        "max_fraction": args.max_fraction,
        "doc_cut": args.doc_cut,
        "query_cut": args.query_cut,
        "heap_factor": args.heap_factor,
    }

    query_records = None
    if args.stage in ("all", "baseline-search", "seismic-search"):
        query_started = time.perf_counter()
        query_records = read_vectors(args.queries)
        report["query_load_seconds"] = time.perf_counter() - query_started
        if args.expected_queries and len(query_records) != args.expected_queries:
            raise RuntimeError(f"Loaded {len(query_records):,} queries, expected {args.expected_queries:,}")

    if args.stage in ("all", "baseline-index"):
        report["baseline_index"] = build_baseline_index(args.documents, baseline_index_dir, args)

    if args.stage in ("all", "baseline-search"):
        if not baseline_index_dir.exists():
            raise FileNotFoundError(baseline_index_dir)
        report["baseline_query_prepare"] = prepare_queries(query_records, baseline_queries)
        report["baseline_search"] = baseline_search(
            baseline_index_dir,
            baseline_queries,
            baseline_run,
            len(query_records),
            args,
        )

    index = None
    if args.stage in ("all", "seismic-build"):
        index, report["seismic_build"] = build_seismic(args.documents, seismic_prefix, args)

    if args.stage in ("all", "seismic-search"):
        from seismic import SeismicIndex

        if not seismic_index_path.exists():
            raise FileNotFoundError(seismic_index_path)
        if index is not None:
            del index
            gc.collect()
        load_started = time.perf_counter()
        index = SeismicIndex.load(str(seismic_index_path))
        index_load_seconds = time.perf_counter() - load_started
        report["seismic_search"] = seismic_search(index, query_records, seismic_run, args)
        report["seismic_search"]["index_load_seconds"] = index_load_seconds

    if args.stage in ("all", "evaluate"):
        if not baseline_run.exists() or not seismic_run.exists():
            raise FileNotFoundError("Both baseline and SEISMIC run files are required")
        report["baseline_metrics"] = evaluate_run(baseline_run)
        report["seismic_metrics"] = evaluate_run(seismic_run)
        baseline_seconds = float(report["baseline_search"]["seconds"])
        seismic_seconds = float(report["seismic_search"]["seconds"])
        report["comparison"] = {
            "search_speedup_seismic_vs_lucene": baseline_seconds / max(seismic_seconds, 1e-9),
            "mrr_at_10_delta": (
                report["seismic_metrics"]["mrr_at_10"] - report["baseline_metrics"]["mrr_at_10"]
            ),
            "recall_at_1000_delta": (
                report["seismic_metrics"]["recall_at_1000"]
                - report["baseline_metrics"]["recall_at_1000"]
            ),
        }

    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Benchmark report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
