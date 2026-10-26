from __future__ import annotations

import json
import math
import multiprocessing
import os
import time
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SparseEncoder
from tqdm import tqdm

from src.datasets.msmarco import read_documents
from src.utils.io import atomic_write_json


@dataclass
class EncodingStats:
    documents: int = 0
    skipped_documents: int = 0
    unpadded_tokens: int = 0
    padded_tokens: int = 0
    fixed_length_tokens: int = 0
    non_zero_terms: int = 0
    encode_seconds: float = 0.0

    def add(self, other: EncodingStats) -> None:
        for name in asdict(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def report(self) -> dict[str, float | int]:
        encoded = max(self.documents - self.skipped_documents, 0)
        return {
            **asdict(self),
            "encoded_documents": encoded,
            "documents_per_second": encoded / max(self.encode_seconds, 1e-9),
            "average_non_zero_terms": self.non_zero_terms / max(encoded, 1),
            "padding_fraction": (
                (self.padded_tokens - self.unpadded_tokens) / max(self.padded_tokens, 1)
            ),
        }


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def resolve_model_path(config: dict[str, Any], project_root: Path) -> str:
    local_path = Path(str(config["local_path"])).expanduser()
    if not local_path.is_absolute():
        local_path = project_root / local_path
    return str(local_path) if local_path.exists() else str(config["name_or_path"])


def _device_for_rank(config: dict[str, Any], rank: int) -> str:
    configured = str(config.get("device", "auto"))
    if configured != "auto":
        if configured == "cuda":
            return f"cuda:{rank}"
        return configured
    if torch.cuda.is_available():
        return f"cuda:{rank}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _precision(config: dict[str, Any], device: str) -> str:
    precision = str(config.get("precision", "auto"))
    if precision == "auto":
        return "fp16" if device.startswith("cuda") else "fp32"
    if precision == "fp16" and not (device.startswith("cuda") or device == "mps"):
        raise ValueError("fp16 SPLADE encoding requires CUDA or MPS")
    return precision


def _autocast(device: str, precision: str):
    if precision == "fp16":
        return torch.autocast(device_type=device.split(":", 1)[0], dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type=device.split(":", 1)[0], dtype=torch.bfloat16)
    return nullcontext()


def _load_model(config: dict[str, Any], project_root: Path, device: str) -> SparseEncoder:
    model = SparseEncoder(
        resolve_model_path(config, project_root),
        device=device,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    model.max_seq_length = int(config["max_length"])
    model.eval()
    return model


def encode_batch(
    model: SparseEncoder,
    records: list[tuple[str, str]],
    config: dict[str, Any],
    precision: str,
) -> tuple[list[dict[str, Any]], EncodingStats]:
    tokenizer = model.tokenizer
    texts = [text for _, text in records]
    tokenized = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=int(config["max_length"]),
        truncation=True,
        padding=False,
    )
    lengths = [len(row) for row in tokenized["input_ids"]]
    examples = []
    for row, (identifier, text) in enumerate(records):
        features = {name: values[row] for name, values in tokenized.items()}
        examples.append((lengths[row], identifier, text, features))
    examples.sort(key=lambda item: item[0])

    stats = EncodingStats(
        documents=len(records),
        unpadded_tokens=sum(lengths),
        fixed_length_tokens=len(records) * int(config["max_length"]),
    )
    encoded_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch in chunks(examples, int(config["batch_size"])):
        features = tokenizer.pad(
            [row_features for _, _, _, row_features in batch],
            padding=True,
            return_tensors="pt",
        )
        stats.padded_tokens += int(features["attention_mask"].numel())
        features = {
            name: value.to(model.device) if isinstance(value, torch.Tensor) else value
            for name, value in features.items()
        }
        features["modality"] = "text"
        with torch.inference_mode(), _autocast(str(model.device), precision):
            embeddings = model.forward(features)["sentence_embedding"].detach().float().cpu()

        for (_, identifier, text, _), embedding in zip(batch, embeddings, strict=True):
            indices = torch.nonzero(
                embedding > float(config.get("threshold", 0.0)), as_tuple=False
            ).flatten()
            max_terms = config.get("max_terms")
            if max_terms is not None and len(indices) > int(max_terms):
                selected = torch.topk(embedding[indices], k=int(max_terms)).indices
                indices = indices[selected]
            indices = indices.sort().values
            tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
            values = embedding[indices].tolist()
            vector = {token: float(value) for token, value in zip(tokens, values, strict=True)}
            stats.non_zero_terms += len(vector)
            record = {"id": str(identifier), "vector": vector}
            if bool(config.get("include_content", False)):
                record["content"] = text
            encoded_records.append(record)
    stats.encode_seconds = time.perf_counter() - started
    return encoded_records, stats


def _write_encoded_shard(
    model: SparseEncoder,
    records: list[tuple[str, str]],
    destination: Path,
    config: dict[str, Any],
    precision: str,
) -> EncodingStats:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    total = EncodingStats()
    window_size = int(config["batch_size"]) * int(config["bucket_multiplier"])
    with temporary.open("w", encoding="utf-8") as output:
        for window in chunks(records, window_size):
            encoded, stats = encode_batch(model, window, config, precision)
            total.add(stats)
            for record in encoded:
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)
    return total


def _document_worker(
    rank: int,
    world_size: int,
    documents_path: str,
    output_dir: str,
    config: dict[str, Any],
    project_root: str,
    max_documents: int | None,
) -> None:
    root = Path(project_root)
    destination_root = Path(output_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    device = _device_for_rank(config, rank)
    precision = _precision(config, device)
    model = _load_model(config, root, device)
    stream = iter(
        read_documents(
            Path(documents_path),
            limit=max_documents,
            rank=rank,
            world_size=world_size,
        )
    )
    expected_for_rank = None
    if max_documents is not None:
        expected_for_rank = max(0, math.ceil((max_documents - rank) / world_size))
    progress = tqdm(
        total=expected_for_rank, unit="doc", desc=f"SPLADE rank {rank}", disable=rank != 0
    )
    total = EncodingStats()
    shard_number = 0
    while True:
        records = []
        for _ in range(int(config["shard_size"])):
            try:
                records.append(next(stream))
            except StopIteration:
                break
        if not records:
            break
        destination = destination_root / f"part-r{rank:02d}-{shard_number:05d}.jsonl"
        shard_number += 1
        if destination.exists():
            total.documents += len(records)
            total.skipped_documents += len(records)
            progress.update(len(records))
            continue
        stats = _write_encoded_shard(model, records, destination, config, precision)
        total.add(stats)
        progress.update(len(records))
        progress.set_postfix(docs_s=f"{stats.report()['documents_per_second']:.1f}")
    progress.close()
    atomic_write_json(
        destination_root / f"encoding.rank-{rank:02d}.json",
        {
            "rank": rank,
            "world_size": world_size,
            "device": device,
            "precision": precision,
            **total.report(),
        },
    )


def _world_size(config: dict[str, Any]) -> int:
    configured = config.get("devices", "auto")
    if str(configured) == "auto":
        return max(torch.cuda.device_count(), 1)
    return int(configured)


def encode_documents(
    documents_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    project_root: Path,
    max_documents: int | None,
    expected_documents: int | None,
) -> dict[str, Any]:
    world_size = _world_size(config)
    if world_size > 1 and torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"Requested {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = (
        world_size,
        str(documents_path),
        str(output_dir),
        config,
        str(project_root),
        max_documents,
    )
    if world_size == 1:
        _document_worker(0, *arguments)
    else:
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(target=_document_worker, args=(rank, *arguments), daemon=False)
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        failures = [process.exitcode for process in processes if process.exitcode != 0]
        if failures:
            raise RuntimeError(f"SPLADE workers failed with exit codes {failures}")

    rank_reports = [
        json.loads((output_dir / f"encoding.rank-{rank:02d}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    documents = sum(int(report["documents"]) for report in rank_reports)
    expected = max_documents if max_documents is not None else expected_documents
    if expected is not None and documents != int(expected):
        raise RuntimeError(f"Encoded {documents:,} documents, expected {int(expected):,}")
    summary = {
        "world_size": world_size,
        "documents": documents,
        "encoded_documents_current_invocation": sum(
            int(report["encoded_documents"]) for report in rank_reports
        ),
        "non_zero_terms_current_invocation": sum(
            int(report["non_zero_terms"]) for report in rank_reports
        ),
        "rank_reports": rank_reports,
    }
    atomic_write_json(output_dir / "encoding_summary.json", summary)
    return summary


def encode_queries(
    queries: list[tuple[str, str]],
    destination: Path,
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    device = _device_for_rank(config, 0)
    precision = _precision(config, device)
    model = _load_model(config, project_root, device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    total = EncodingStats()
    with temporary.open("w", encoding="utf-8") as output:
        window_size = int(config["batch_size"]) * int(config["bucket_multiplier"])
        for window in chunks(queries, window_size):
            encoded, stats = encode_batch(model, window, config, precision)
            total.add(stats)
            for record in encoded:
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)
    return {"device": device, "precision": precision, **total.report()}
