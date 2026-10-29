"""Encode MS MARCO passages into Pyserini's sparse-vector JSONL format.

The collection is processed in resumable shards.  Inside every bounded window
documents are sorted by tokenized length, so each micro-batch is padded only to
the longest document in that batch.  This keeps the output independent of the
processing order while avoiding the memory cost of sorting the full corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from sentence_transformers import SparseEncoder
from tqdm import tqdm

from config import baseline_model_path, msmarco_dir

COLLECTION_PATH = msmarco_dir() / "collection.tsv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "baseline" / "vectors"
WEIGHT_RANGE = 5.0
QUANT_RANGE = 256
DEFAULT_COLLECTION_SIZE = 8_841_823


@dataclass
class PaddingStats:
    documents: int = 0
    unpadded_tokens: int = 0
    padded_tokens: int = 0
    naive_padded_tokens: int = 0
    fixed_length_tokens: int = 0
    non_zero_terms: int = 0
    encode_seconds: float = 0.0

    def add(self, other: PaddingStats) -> None:
        for field in asdict(self):
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def report(self) -> dict[str, float | int]:
        padded = max(self.padded_tokens, 1)
        naive = max(self.naive_padded_tokens, 1)
        fixed = max(self.fixed_length_tokens, 1)
        seconds = max(self.encode_seconds, 1e-9)
        return {
            **asdict(self),
            "docs_per_second": self.documents / seconds,
            "avg_non_zero_terms": self.non_zero_terms / max(self.documents, 1),
            "padding_fraction": (self.padded_tokens - self.unpadded_tokens) / padded,
            "padding_saved_vs_naive_fraction": (self.naive_padded_tokens - self.padded_tokens)
            / naive,
            "tokens_saved_vs_fixed_fraction": (self.fixed_length_tokens - self.padded_tokens)
            / fixed,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=COLLECTION_PATH)
    parser.add_argument("--model", default=baseline_model_path())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--bucket-multiplier",
        type=int,
        default=32,
        help="Length-sort this many batches at a time.",
    )
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-docs", type=int, default=None, help="Useful for smoke tests.")
    parser.add_argument("--total-docs", type=int, default=DEFAULT_COLLECTION_SIZE)
    parser.add_argument("--device", default=None, help="cpu, mps, cuda, or auto when omitted.")
    parser.add_argument(
        "--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
        help="auto selects fp16 on CUDA and fp32 elsewhere.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_collection(
    path: Path,
    limit: int | None,
    rank: int = 0,
    world_size: int = 1,
) -> Iterable[tuple[str, str]]:
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if limit is not None and index >= limit:
                break
            if index % world_size != rank:
                continue
            try:
                doc_id, text = line.rstrip("\n").split("\t", 1)
            except ValueError as error:
                raise ValueError(f"Invalid collection row {index + 1}") from error
            yield doc_id, text


def chunks(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def naive_padded_token_count(lengths: list[int], batch_size: int) -> int:
    return sum(max(batch) * len(batch) for batch in chunks(lengths, batch_size))


def encode_window(
    model: SparseEncoder,
    documents: list[tuple[str, str]],
    batch_size: int,
    max_length: int,
    precision: str = "fp32",
) -> tuple[list[tuple[str, dict[str, int]]], PaddingStats]:
    tokenizer = model.tokenizer
    texts = [text for _, text in documents]
    tokenized = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    lengths = [len(ids) for ids in tokenized["input_ids"]]
    stats = PaddingStats(
        documents=len(documents),
        unpadded_tokens=sum(lengths),
        naive_padded_tokens=naive_padded_token_count(lengths, batch_size),
        fixed_length_tokens=len(documents) * max_length,
    )

    examples = []
    for row, (doc_id, _) in enumerate(documents):
        encoded = {key: values[row] for key, values in tokenized.items()}
        examples.append((lengths[row], doc_id, encoded))
    examples.sort(key=lambda item: item[0])

    results: list[tuple[str, dict[str, int]]] = []
    started = time.perf_counter()
    for batch in chunks(examples, batch_size):
        features = tokenizer.pad(
            [encoded for _, _, encoded in batch],
            padding=True,
            return_tensors="pt",
        )
        stats.padded_tokens += int(
            features["attention_mask"].shape[0] * features["attention_mask"].shape[1]
        )
        features = {
            key: value.to(model.device) if isinstance(value, torch.Tensor) else value
            for key, value in features.items()
        }
        features["modality"] = "text"
        device_type = model.device.type
        if precision == "fp16":
            autocast = torch.autocast(device_type=device_type, dtype=torch.float16)
        elif precision == "bf16":
            autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        else:
            autocast = nullcontext()
        with torch.inference_mode(), autocast:
            dense_embeddings = model.forward(features)["sentence_embedding"].detach().cpu()

        for (_, doc_id, _), embedding in zip(batch, dense_embeddings, strict=True):
            indices = torch.nonzero(embedding > 0, as_tuple=False).flatten()
            weights = torch.round(embedding[indices] / WEIGHT_RANGE * QUANT_RANGE).to(torch.int32)
            keep = weights > 0
            indices = indices[keep].tolist()
            weights = weights[keep].tolist()
            tokens = tokenizer.convert_ids_to_tokens(indices)
            vector = dict(zip(tokens, weights, strict=True))
            stats.non_zero_terms += len(vector)
            results.append((doc_id, vector))
    stats.encode_seconds = time.perf_counter() - started
    return results, stats


def encode_shard(
    model: SparseEncoder,
    documents: list[tuple[str, str]],
    destination: Path,
    batch_size: int,
    bucket_multiplier: int,
    max_length: int,
    precision: str,
) -> PaddingStats:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    stats = PaddingStats()
    with temporary.open("w", encoding="utf-8") as output:
        window_size = batch_size * bucket_multiplier
        for window in chunks(documents, window_size):
            encoded, window_stats = encode_window(model, window, batch_size, max_length, precision)
            stats.add(window_stats)
            for doc_id, vector in encoded:
                output.write(
                    json.dumps({"id": doc_id, "vector": vector}, separators=(",", ":")) + "\n"
                )
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)
    return stats


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.bucket_multiplier < 1 or args.shard_size < 1:
        raise ValueError("batch-size, bucket-multiplier and shard-size must be positive")
    if not args.collection.exists():
        raise FileNotFoundError(args.collection)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not 0 <= rank < world_size:
        raise ValueError(f"Invalid distributed rank {rank}/{world_size}")

    device = args.device
    if device is None and torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    precision = args.precision
    if precision == "auto":
        precision = "fp16" if device and device.startswith("cuda") else "fp32"
    if precision == "fp16" and (not device or not device.startswith("cuda")):
        raise ValueError("fp16 indexing is supported only on CUDA; use --precision fp32")

    output_dir = args.output_dir / f"world-size-{world_size}" if world_size > 1 else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir.parent / f"encoding_metrics.rank-{rank:02d}.jsonl"
    model = SparseEncoder(args.model, device=device)
    model.max_seq_length = args.max_length
    model.eval()

    total = PaddingStats()
    stream = read_collection(args.collection, args.max_docs, rank, world_size)
    document_total = args.max_docs if args.max_docs is not None else args.total_docs
    progress_total = max(0, math.ceil((document_total - rank) / world_size))
    progress = tqdm(
        unit="doc",
        total=progress_total,
        desc=f"GPU rank {rank}",
        disable=rank != 0,
    )
    shard_index = 0
    while True:
        documents = []
        for _ in range(args.shard_size):
            try:
                documents.append(next(stream))
            except StopIteration:
                break
        if not documents:
            break

        destination = output_dir / f"part-r{rank:02d}-{shard_index:05d}.jsonl"
        shard_index += 1
        if destination.exists() and not args.overwrite:
            progress.update(len(documents))
            continue

        stats = encode_shard(
            model,
            documents,
            destination,
            args.batch_size,
            args.bucket_multiplier,
            args.max_length,
            precision,
        )
        total.add(stats)
        record = {
            "shard": destination.name,
            "batch_size": args.batch_size,
            "bucket_multiplier": args.bucket_multiplier,
            "max_length": args.max_length,
            "rank": rank,
            "world_size": world_size,
            "device": str(model.device),
            "precision": precision,
            **stats.report(),
        }
        with metrics_path.open("a", encoding="utf-8") as metrics:
            metrics.write(json.dumps(record) + "\n")
        progress.update(len(documents))
        progress.set_postfix(docs_s=f"{stats.report()['docs_per_second']:.1f}")

    progress.close()
    summary = {
        "rank": rank,
        "world_size": world_size,
        "device": str(model.device),
        "precision": precision,
        **total.report(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
