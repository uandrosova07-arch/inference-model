from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing
import os
import shutil
import tarfile
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def count_lines(path: Path) -> int:
    lines = 0
    last_byte = b""
    with path.open("rb") as source:
        while block := source.read(16 * 1024 * 1024):
            lines += block.count(b"\n")
            last_byte = block[-1:]
    if last_byte and last_byte != b"\n":
        lines += 1
    return lines


def file_fingerprint(path: Path, *, include_lines: bool = False) -> dict[str, Any]:
    """Return a content fingerprint, optionally counting lines in the same pass."""
    digest = hashlib.sha256()
    size = 0
    lines = 0
    last_byte = b""
    with path.open("rb") as source:
        while block := source.read(16 * 1024 * 1024):
            digest.update(block)
            size += len(block)
            if include_lines:
                lines += block.count(b"\n")
                last_byte = block[-1:]
    result: dict[str, Any] = {
        "type": "file",
        "bytes": size,
        "sha256": digest.hexdigest(),
    }
    if include_lines:
        if last_byte and last_byte != b"\n":
            lines += 1
        result["lines"] = lines
    return result


def path_inventory(path: Path) -> dict[str, Any]:
    """Create a cheap resume fingerprint without reading file contents."""
    if path.is_file():
        stat = path.stat()
        return {
            "type": "file",
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    if not path.is_dir():
        return {"type": "missing"}
    files = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        stat = item.stat()
        files.append(
            {
                "path": item.relative_to(path).as_posix(),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "type": "directory",
        "file_count": len(files),
        "bytes": sum(int(item["bytes"]) for item in files),
        "signature": stable_hash(files),
    }


def path_fingerprint(path: Path) -> dict[str, Any]:
    """Return a deterministic content fingerprint for a file or directory."""
    if path.is_file():
        return file_fingerprint(path)
    if not path.is_dir():
        return {"type": "missing"}

    files = []
    combined = hashlib.sha256()
    total_bytes = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        fingerprint = file_fingerprint(item)
        size = int(fingerprint["bytes"])
        sha256 = str(fingerprint["sha256"])
        files.append({"path": relative, "bytes": size, "sha256": sha256})
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(size).encode("ascii"))
        combined.update(b"\0")
        combined.update(sha256.encode("ascii"))
        combined.update(b"\n")
        total_bytes += size
    return {
        "type": "directory",
        "file_count": len(files),
        "bytes": total_bytes,
        "sha256": combined.hexdigest(),
        "files": files,
    }


def vector_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(
        item
        for item in path.iterdir()
        if item.is_file()
        and (item.name.endswith(".jsonl") or item.name.endswith(".jsonl.gz"))
        and not item.name.endswith(".tmp")
    )
    if not files:
        raise FileNotFoundError(f"No JSONL vector shards in {path}")
    return files


def _first_regular_member(archive: tarfile.TarFile) -> tarfile.TarInfo:
    for member in archive:
        parts = Path(member.name).parts
        if (
            member.isfile()
            and not Path(member.name).name.startswith("._")
            and "__MACOSX" not in parts
        ):
            return member
    raise ValueError(f"No regular file in {archive.name}")


def jsonl_lines(path: Path) -> Iterator[str]:
    if path.is_dir():
        for shard in vector_files(path):
            yield from jsonl_lines(shard)
    elif path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            yield from source
    elif path.name.endswith(".tar.gz") or path.suffix == ".tgz":
        with tarfile.open(path, "r:gz") as archive:
            member = _first_regular_member(archive)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read {member.name} from {path}")
            for line in source:
                yield line.decode("utf-8")
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            yield from source
    else:
        raise ValueError(f"Expected JSONL, JSONL.GZ, a shard directory, or tar.gz: {path}")


def _copy_vector_file(path: Path, output) -> None:
    opener = gzip.open if path.name.endswith(".jsonl.gz") else open
    with opener(path, "rb") as source:
        shutil.copyfileobj(source, output, length=16 * 1024 * 1024)


def _feed_fifo(shards: list[str], fifo_path: str, passes: int) -> None:
    """Feed the two reads performed by Seismic without materializing a merge."""
    for pass_number in range(passes):
        with open(fifo_path, "wb", buffering=0) as output:
            for shard in shards:
                _copy_vector_file(Path(shard), output)
        if pass_number + 1 < passes:
            time.sleep(0.2)


@contextmanager
def seismic_document_input(
    path: Path,
    temporary_dir: Path | None = None,
    shard_input_mode: str = "fifo",
) -> Iterator[Path]:
    """Return a Seismic input while avoiding a duplicate full-size shard merge."""
    is_compressed_jsonl = path.name.endswith(".jsonl.gz")
    if path.is_file() and not is_compressed_jsonl:
        yield path
        return

    shards = vector_files(path)
    root = temporary_dir if temporary_dir else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    if shard_input_mode == "fifo":
        if not hasattr(os, "mkfifo"):
            raise RuntimeError(
                "FIFO shard streaming is unavailable; set index.shard_input_mode=merge"
            )
        fifo_root = Path(tempfile.mkdtemp(prefix="seismic-fifo-", dir=root))
        fifo = fifo_root / "documents.jsonl"
        os.mkfifo(fifo)
        methods = multiprocessing.get_all_start_methods()
        context = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
        producer = context.Process(
            target=_feed_fifo,
            args=([str(shard) for shard in shards], str(fifo), 2),
            daemon=True,
        )
        producer.start()
        try:
            yield fifo
            producer.join(timeout=60)
            if producer.is_alive():
                raise RuntimeError("SEISMIC FIFO producer did not finish its two input passes")
            if producer.exitcode != 0:
                raise RuntimeError(
                    f"SEISMIC FIFO producer failed with exit code {producer.exitcode}"
                )
        finally:
            if producer.is_alive():
                producer.terminate()
                producer.join()
            shutil.rmtree(fifo_root, ignore_errors=True)
        return
    if shard_input_mode != "merge":
        raise ValueError("index.shard_input_mode must be 'fifo' or 'merge'")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix="seismic-documents-", suffix=".jsonl", dir=root
    )
    os.close(descriptor)
    merged = Path(temporary_name)
    try:
        with merged.open("wb") as output:
            for shard in shards:
                _copy_vector_file(shard, output)
            output.flush()
            os.fsync(output.fileno())
        yield merged
    finally:
        merged.unlink(missing_ok=True)


def stable_hash(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()
