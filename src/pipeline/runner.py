from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import shutil
import socket
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.datasets import load_qrels, read_queries
from src.index import build_index, search_index
from src.metrics import evaluate, load_run
from src.model import encode_documents, encode_queries
from src.utils.io import (
    atomic_write_json,
    count_lines,
    directory_size,
    resolve_path,
    stable_hash,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class StageResult:
    outputs: list[Path]
    metrics: dict[str, Any]


class PipelineRunner:
    """Run an inclusive interval of resumable, independently configurable stages."""

    def __init__(self, config: DictConfig, project_root: Path):
        self.omega_config = config
        self.config: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # type: ignore[assignment]
        self.project_root = project_root
        self.paths = {
            name: resolve_path(value, project_root) for name, value in self.config["paths"].items()
        }
        self.output_dir = resolve_path(self.config["runtime"]["output_dir"], project_root)
        temporary = self.config["runtime"].get("temporary_dir")
        self.temporary_dir = resolve_path(temporary, project_root) if temporary else None
        self.stages: dict[str, Callable[[], StageResult]] = {
            "prepare": self._prepare,
            "encode_documents": self._encode_documents,
            "encode_queries": self._encode_queries,
            "build_index": self._build_index,
            "search": self._search,
            "metrics": self._metrics,
        }

    def run(self) -> dict[str, Any]:
        pipeline_started = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths["state_dir"].mkdir(parents=True, exist_ok=True)
        self.paths["resolved_config"].parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.omega_config, self.paths["resolved_config"], resolve=True)

        requested = self._requested_stages()
        LOGGER.info("Experiment: %s", self.config["experiment"]["name"])
        LOGGER.info("Output directory: %s", self.output_dir)
        LOGGER.info("Resolved config: %s", self.paths["resolved_config"])
        LOGGER.info("Pipeline stages: %s", " -> ".join(requested))
        summary: dict[str, Any] = {"experiment": self.config["experiment"]["name"], "stages": {}}
        for stage_name in requested:
            signature = self._stage_signature(stage_name)
            state = self._read_state(stage_name)
            forced = stage_name in set(self.config["pipeline"].get("force", []))
            if (
                bool(self.config["pipeline"].get("resume", True))
                and not forced
                and state
                and state.get("status") == "success"
                and state.get("signature") == signature
                and all(Path(path).exists() for path in state.get("outputs", []))
            ):
                LOGGER.info("[%s] already complete; skipping", stage_name)
                summary["stages"][stage_name] = {"status": "skipped", **state.get("metrics", {})}
                continue

            LOGGER.info("[%s] starting", stage_name)
            started = time.perf_counter()
            started_at = datetime.now(timezone.utc).isoformat()
            try:
                result = self.stages[stage_name]()
                elapsed = time.perf_counter() - started
                stage_state = {
                    "stage": stage_name,
                    "status": "success",
                    "signature": signature,
                    "started_at": started_at,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "seconds": elapsed,
                    "outputs": [str(path) for path in result.outputs],
                    "metrics": result.metrics,
                }
                atomic_write_json(self._state_path(stage_name), stage_state)
                summary["stages"][stage_name] = {"status": "success", **result.metrics}
                LOGGER.info("[%s] complete in %.2fs", stage_name, elapsed)
            except Exception as error:
                LOGGER.exception("[%s] failed", stage_name)
                atomic_write_json(
                    self._state_path(stage_name),
                    {
                        "stage": stage_name,
                        "status": "failed",
                        "signature": signature,
                        "started_at": started_at,
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                raise
        invocation_seconds = time.perf_counter() - pipeline_started
        summary["current_invocation_seconds"] = invocation_seconds
        LOGGER.info("Artifacts: %s", self.output_dir)
        LOGGER.info("Pipeline invocation complete in %.2fs", invocation_seconds)
        self._finalize_report(invocation_seconds)
        return summary

    def _finalize_report(self, invocation_seconds: float) -> None:
        if not self.paths["report"].exists():
            return
        report = json.loads(self.paths["report"].read_text(encoding="utf-8"))
        states = {}
        cumulative_seconds = 0.0
        for name in self.config["pipeline"]["stages"]:
            state = self._read_state(name)
            if state and state.get("status") == "success":
                states[name] = {
                    "started_at": state.get("started_at"),
                    "completed_at": state.get("completed_at"),
                    "seconds": state.get("seconds"),
                    **state.get("metrics", {}),
                }
                cumulative_seconds += float(state.get("seconds", 0.0))
        report["stages"] = states
        report["timing"] = {
            "current_invocation_seconds": invocation_seconds,
            "cumulative_stage_seconds": cumulative_seconds,
        }
        report.setdefault("artifacts", {})["pipeline_log"] = {
            "path": str(self.paths["pipeline_log"]),
            "bytes": (
                self.paths["pipeline_log"].stat().st_size
                if self.paths["pipeline_log"].exists()
                else None
            ),
        }
        atomic_write_json(self.paths["report"], report)

    def _requested_stages(self) -> list[str]:
        configured = list(self.config["pipeline"]["stages"])
        unknown = [name for name in configured if name not in self.stages]
        if unknown:
            raise ValueError(f"Unknown pipeline stages: {unknown}")
        start = str(self.config["pipeline"]["start_at"])
        end = str(self.config["pipeline"]["end_at"])
        if start not in configured or end not in configured:
            raise ValueError(f"start_at/end_at must be one of {configured}")
        start_index = configured.index(start)
        end_index = configured.index(end)
        if start_index > end_index:
            raise ValueError("pipeline.start_at must not come after pipeline.end_at")
        return configured[start_index : end_index + 1]

    def _state_path(self, stage_name: str) -> Path:
        return self.paths["state_dir"] / f"{stage_name}.json"

    def _read_state(self, stage_name: str) -> dict[str, Any] | None:
        path = self._state_path(stage_name)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _stage_signature(self, stage_name: str) -> str:
        config = deepcopy(self.config)
        for name in ("start_at", "end_at", "resume", "force"):
            config["pipeline"].pop(name, None)
        fingerprint = {"stage": stage_name, "config": config, "inputs": {}}
        input_names = {
            "prepare": [],
            "encode_documents": [],
            "encode_queries": [],
            "build_index": ["document_vectors"],
            "search": ["query_vectors", "index_file"],
            "metrics": ["run"],
        }[stage_name]
        for name in input_names:
            path = self.paths[name]
            if path.exists():
                stat = path.stat()
                fingerprint["inputs"][name] = {
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
        dataset_input = {
            "prepare": "documents",
            "encode_documents": "documents",
            "encode_queries": "queries",
            "metrics": "qrels",
        }.get(stage_name)
        if dataset_input is not None:
            source_mode = self.config["dataset"].get(
                "query_source" if dataset_input == "queries" else "qrels_source"
            )
            if dataset_input == "documents" or source_mode == "file":
                path = resolve_path(self.config["dataset"][dataset_input], self.project_root)
                if path.exists():
                    stat = path.stat()
                    fingerprint["inputs"][f"dataset.{dataset_input}"] = {
                        "path": str(path),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
        return stable_hash(fingerprint)

    def _prepare(self) -> StageResult:
        dataset = self.config["dataset"]
        documents = resolve_path(dataset["documents"], self.project_root)
        if not documents.exists():
            raise FileNotFoundError(documents)
        checked: dict[str, Any] = {
            "documents": str(documents),
            "documents_bytes": documents.stat().st_size,
        }
        if bool(dataset.get("validate_document_count", True)):
            document_count = count_lines(documents)
            checked["documents_count"] = document_count
            expected = dataset.get("expected_documents")
            max_documents = dataset.get("max_documents")
            if max_documents is None and expected is not None and document_count != int(expected):
                raise RuntimeError(
                    f"Dataset contains {document_count:,} documents, expected {int(expected):,}"
                )
            if max_documents is not None and document_count < int(max_documents):
                raise RuntimeError(
                    f"Dataset contains {document_count:,} documents, fewer than "
                    f"dataset.max_documents={int(max_documents):,}"
                )
        if dataset["query_source"] == "file":
            queries = resolve_path(dataset["queries"], self.project_root)
            if not queries.exists():
                raise FileNotFoundError(queries)
            checked.update(queries=str(queries), queries_bytes=queries.stat().st_size)
        if dataset["qrels_source"] == "file":
            qrels = resolve_path(dataset["qrels"], self.project_root)
            if not qrels.exists():
                raise FileNotFoundError(qrels)
            checked.update(qrels=str(qrels), qrels_bytes=qrels.stat().st_size)
        configured_queries = read_queries(
            source=str(dataset["query_source"]),
            path=resolve_path(dataset["queries"], self.project_root),
            topics_name=str(dataset["pyserini_topics"]),
            limit=dataset.get("max_queries"),
        )
        configured_qrels = load_qrels(
            source=str(dataset["qrels_source"]),
            path=resolve_path(dataset["qrels"], self.project_root),
            qrels_name=str(dataset["pyserini_qrels"]),
        )
        query_ids = {query_id for query_id, _ in configured_queries}
        checked["queries_count"] = len(configured_queries)
        checked["qrels_queries"] = len(
            {query_id for query_id in configured_qrels if query_id in query_ids}
        )
        checked["environment"] = self._environment_info()
        manifest = self.paths["state_dir"] / "dataset.json"
        atomic_write_json(manifest, checked)
        LOGGER.info("Input validation complete: %s", checked)
        return StageResult([manifest], checked)

    def _environment_info(self) -> dict[str, Any]:
        import torch

        cuda_devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                cuda_devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                    }
                )
        if bool(self.config["runtime"].get("require_cuda", False)) and not cuda_devices:
            raise RuntimeError("CUDA is required by this config, but no CUDA device is available")

        package_versions = {}
        for package in (
            "torch",
            "sentence-transformers",
            "transformers",
            "pyseismic-lsr",
            "pyserini",
            "hydra-core",
        ):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = None

        output_disk = shutil.disk_usage(self.output_dir)
        temporary_root = self.temporary_dir or self.output_dir
        temporary_root.mkdir(parents=True, exist_ok=True)
        temporary_disk = shutil.disk_usage(temporary_root)
        return {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "packages": package_versions,
            "cuda_devices": cuda_devices,
            "output_disk_free_bytes": output_disk.free,
            "temporary_disk_free_bytes": temporary_disk.free,
        }

    def _encode_documents(self) -> StageResult:
        dataset = self.config["dataset"]
        result = encode_documents(
            documents_path=resolve_path(dataset["documents"], self.project_root),
            output_dir=self.paths["document_vectors"],
            config=self.config["model"],
            project_root=self.project_root,
            max_documents=dataset.get("max_documents"),
            expected_documents=dataset.get("expected_documents"),
            log_dir=self.paths["log_dir"],
        )
        result["vector_bytes"] = directory_size(self.paths["document_vectors"])
        return StageResult([self.paths["document_vectors"]], result)

    def _encode_queries(self) -> StageResult:
        dataset = self.config["dataset"]
        queries = read_queries(
            source=str(dataset["query_source"]),
            path=resolve_path(dataset["queries"], self.project_root),
            topics_name=str(dataset["pyserini_topics"]),
            limit=dataset.get("max_queries"),
        )
        result = encode_queries(
            queries=queries,
            destination=self.paths["query_vectors"],
            config=self.config["model"],
            project_root=self.project_root,
        )
        result["vector_bytes"] = self.paths["query_vectors"].stat().st_size
        return StageResult([self.paths["query_vectors"]], result)

    def _build_index(self) -> StageResult:
        documents = self.paths["document_vectors"]
        if not documents.exists():
            raise FileNotFoundError(documents)
        config = deepcopy(self.config["index"])
        if self.config["dataset"].get("max_documents") is not None:
            config["expected_documents"] = self.config["dataset"]["max_documents"]
        result = build_index(
            documents=documents,
            prefix=self.paths["index_prefix"],
            config=config,
            temporary_dir=self.temporary_dir,
        )
        built_file = Path(result["index_file"])
        configured_file = self.paths["index_file"]
        if built_file.resolve() != configured_file.resolve():
            raise ValueError(
                f"paths.index_file ({configured_file}) must match the file produced from "
                f"paths.index_prefix ({built_file})"
            )
        return StageResult([built_file], result)

    def _search(self) -> StageResult:
        for path in (self.paths["index_file"], self.paths["query_vectors"]):
            if not path.exists():
                raise FileNotFoundError(path)
        result = search_index(
            index_path=self.paths["index_file"],
            query_vectors=self.paths["query_vectors"],
            run_path=self.paths["run"],
            index_variant=str(self.config["index"]["variant"]),
            config=self.config["index"]["search"],
        )
        return StageResult([self.paths["run"]], result)

    def _metrics(self) -> StageResult:
        if not self.paths["run"].exists():
            raise FileNotFoundError(self.paths["run"])
        dataset = self.config["dataset"]
        qrels = load_qrels(
            source=str(dataset["qrels_source"]),
            path=resolve_path(dataset["qrels"], self.project_root),
            qrels_name=str(dataset["pyserini_qrels"]),
        )
        qrels_queries_before_scope = len(qrels)
        if self.config["metrics"].get("query_scope") == "configured_queries":
            configured_queries = read_queries(
                source=str(dataset["query_source"]),
                path=resolve_path(dataset["queries"], self.project_root),
                topics_name=str(dataset["pyserini_topics"]),
                limit=dataset.get("max_queries"),
            )
            query_ids = {query_id for query_id, _ in configured_queries}
            qrels = {query_id: docs for query_id, docs in qrels.items() if query_id in query_ids}
            if not qrels:
                raise ValueError("No qrels remain after applying metrics.query_scope")
        effectiveness = evaluate(
            qrels=qrels,
            ranking=load_run(self.paths["run"]),
            config=self.config["metrics"],
            per_query_path=self.paths["per_query_metrics"],
        )
        effectiveness["qrels_queries_before_scope"] = qrels_queries_before_scope
        prior_stages = {}
        for name in self.config["pipeline"]["stages"]:
            state = self._read_state(name)
            if state and state.get("status") == "success":
                prior_stages[name] = {
                    "seconds": state.get("seconds"),
                    **state.get("metrics", {}),
                }
        report = {
            "experiment": self.config["experiment"]["name"],
            "config": self.config,
            "effectiveness": effectiveness,
            "stages": prior_stages,
            "artifacts": {
                name: {
                    "path": str(path),
                    "bytes": directory_size(path) if path.exists() else None,
                }
                for name, path in self.paths.items()
                if name
                in {
                    "document_vectors",
                    "query_vectors",
                    "index_file",
                    "run",
                    "log_dir",
                    "pipeline_log",
                }
            },
        }
        atomic_write_json(self.paths["report"], report)
        outputs = [self.paths["report"]]
        if bool(self.config["metrics"].get("save_per_query", True)):
            outputs.append(self.paths["per_query_metrics"])
        print(json.dumps(effectiveness, ensure_ascii=False, indent=2), flush=True)
        return StageResult(outputs, effectiveness)
