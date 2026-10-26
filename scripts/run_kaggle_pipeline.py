"""Run the complete two-GPU SPLADE baseline pipeline on Kaggle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from config import BASELINE_MODEL_DIR, BASELINE_MODEL_ID, PROJECT_DIR

DEFAULT_DATA_DIR = Path("/kaggle/input/msmarco")
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "baseline"
EXPECTED_MSMARCO_DOCUMENTS = 8_841_823


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model", default=None, help="Local model path; downloaded once when omitted."
    )
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bucket-multiplier", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--index-threads", type=int, default=8)
    parser.add_argument("--search-threads", type=int, default=8)
    parser.add_argument("--max-docs", type=int, default=None, help="Set for a smoke run only.")
    parser.add_argument(
        "--stage",
        choices=("all", "encode", "index", "queries", "search", "evaluate"),
        default="all",
    )
    return parser.parse_args()


def run(command: list[str], env: dict[str, str], timings: dict[str, float], name: str) -> None:
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=PROJECT_DIR, env=env, check=True)
    timings[f"{name}_seconds"] = time.perf_counter() - started


def ensure_model(model: str | None) -> str:
    if model:
        return model
    if not BASELINE_MODEL_DIR.exists():
        print(f"Downloading model {BASELINE_MODEL_ID} once...", flush=True)
        snapshot_download(repo_id=BASELINE_MODEL_ID, local_dir=BASELINE_MODEL_DIR)
    return str(BASELINE_MODEL_DIR)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def summarize_encoding(output_dir: Path, wall_seconds: float) -> dict[str, float | int]:
    totals = {
        "documents": 0,
        "unpadded_tokens": 0,
        "padded_tokens": 0,
        "naive_padded_tokens": 0,
        "fixed_length_tokens": 0,
        "non_zero_terms": 0,
    }
    seconds_by_rank: dict[int, float] = {}
    for path in sorted(output_dir.glob("encoding_metrics.rank-*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                for key in totals:
                    totals[key] += record[key]
                rank = int(record["rank"])
                seconds_by_rank[rank] = seconds_by_rank.get(rank, 0.0) + record["encode_seconds"]
    documents = max(totals["documents"], 1)
    padded = max(totals["padded_tokens"], 1)
    naive = max(totals["naive_padded_tokens"], 1)
    fixed = max(totals["fixed_length_tokens"], 1)
    return {
        **totals,
        "wall_seconds_current_invocation": wall_seconds,
        "estimated_parallel_encode_seconds": max(seconds_by_rank.values(), default=0.0),
        "combined_docs_per_second": totals["documents"]
        / max(max(seconds_by_rank.values(), default=0.0), 1e-9),
        "avg_non_zero_terms": totals["non_zero_terms"] / documents,
        "padding_fraction": (totals["padded_tokens"] - totals["unpadded_tokens"]) / padded,
        "padding_saved_vs_naive_fraction": (totals["naive_padded_tokens"] - totals["padded_tokens"])
        / naive,
        "tokens_saved_vs_fixed_fraction": (totals["fixed_length_tokens"] - totals["padded_tokens"])
        / fixed,
    }


def validate_environment(args: argparse.Namespace) -> None:
    required = ("collection.tsv",)
    missing = [name for name in required if not (args.data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing in {args.data_dir}: {', '.join(missing)}")
    if args.gpus < 1:
        raise ValueError("--gpus must be positive")
    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError(f"Requested {args.gpus} GPUs, found {torch.cuda.device_count()}")


def main() -> None:
    args = parse_args()
    validate_environment(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = ensure_model(args.model)
    python = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_DIR)
    env["MSMARCO_DATA_DIR"] = str(args.data_dir)
    env["BASELINE_MODEL_PATH"] = model
    env["PYTHONUNBUFFERED"] = "1"
    timings: dict[str, float | int | str | list[str]] = {
        "gpus": args.gpus,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(args.gpus)],
        "batch_size_per_gpu": args.batch_size,
        "bucket_multiplier": args.bucket_multiplier,
        "max_length": args.max_length,
        "data_dir": str(args.data_dir),
        "model": model,
    }

    vectors_root = args.output_dir / "vectors"
    vectors_dir = vectors_root / f"world-size-{args.gpus}"
    index_dir = args.output_dir / "index"
    query_file = args.output_dir / "queries.dev.splade.tsv"
    run_file = args.output_dir / "run.dev.tsv"
    retrieval_metrics = args.output_dir / "retrieval_metrics.json"

    if args.stage in ("all", "encode"):
        command = [
            python,
            "scripts/launch_parallel_index.py",
            "--workers",
            str(args.gpus),
            "--",
            "--collection",
            str(args.data_dir / "collection.tsv"),
            "--model",
            model,
            "--output-dir",
            str(vectors_root),
            "--batch-size",
            str(args.batch_size),
            "--bucket-multiplier",
            str(args.bucket_multiplier),
            "--shard-size",
            str(args.shard_size),
            "--max-length",
            str(args.max_length),
            "--precision",
            "fp16",
        ]
        if args.max_docs is not None:
            command.extend(("--max-docs", str(args.max_docs)))
        run(command, env, timings, "encode")
        encoding_summary = summarize_encoding(vectors_root, float(timings["encode_seconds"]))
        (args.output_dir / "encoding_summary.json").write_text(
            json.dumps(encoding_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        expected_documents = args.max_docs or EXPECTED_MSMARCO_DOCUMENTS
        if encoding_summary["documents"] != expected_documents:
            raise RuntimeError(
                f"Encoded {encoding_summary['documents']} documents, expected {expected_documents}. "
                "Do not build the index from an incomplete vector directory."
            )
        timings["encoded_vector_bytes"] = directory_size(vectors_dir)

    if args.stage in ("all", "index"):
        if not vectors_dir.exists():
            raise FileNotFoundError(vectors_dir)
        index_exists = index_dir.exists() and any(index_dir.iterdir())
        if index_exists:
            print(f"Skipping existing Lucene index: {index_dir}")
        else:
            temporary_index = index_dir.with_name(index_dir.name + ".tmp")
            if temporary_index.exists():
                shutil.rmtree(temporary_index)
            command = [
                python,
                "-m",
                "pyserini.index.lucene",
                "--collection",
                "JsonVectorCollection",
                "--input",
                str(vectors_dir),
                "--index",
                str(temporary_index),
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                str(args.index_threads),
                "--impact",
                "--pretokenized",
            ]
            run(command, env, timings, "index")
            if index_dir.exists():
                index_dir.rmdir()
            temporary_index.replace(index_dir)
        timings["lucene_index_bytes"] = directory_size(index_dir)

    if args.stage in ("all", "queries"):
        run(
            [
                python,
                "scripts/encode_queries.py",
                "--model",
                model,
                "--output",
                str(query_file),
                "--batch-size",
                str(args.batch_size),
                "--max-length",
                str(args.max_length),
                "--device",
                "cuda:0",
                "--precision",
                "fp16",
            ],
            env,
            timings,
            "queries",
        )

    if args.stage in ("all", "search"):
        run(
            [
                python,
                "-m",
                "pyserini.search.lucene",
                "--index",
                str(index_dir),
                "--topics",
                str(query_file),
                "--output",
                str(run_file),
                "--output-format",
                "msmarco",
                "--batch-size",
                "16",
                "--threads",
                str(args.search_threads),
                "--hits",
                "1000",
                "--impact",
                "--pretokenized",
            ],
            env,
            timings,
            "search",
        )

    if args.stage in ("all", "evaluate"):
        run(
            [
                python,
                "scripts/evaluate_run.py",
                "--run",
                str(run_file),
                "--output",
                str(retrieval_metrics),
            ],
            env,
            timings,
            "evaluate",
        )

    timings_path = args.output_dir / "pipeline_metrics.json"
    timings_path.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")
    print(f"\nPipeline timings: {timings_path}")


if __name__ == "__main__":
    main()
