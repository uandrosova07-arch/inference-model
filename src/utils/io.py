from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
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


def directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def count_lines(path: Path) -> int:
    lines = 0
    with path.open("rb") as source:
        while block := source.read(16 * 1024 * 1024):
            lines += block.count(b"\n")
    return lines


def vector_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(
        item for item in path.glob("*.jsonl") if item.is_file() and not item.name.endswith(".tmp")
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
        raise ValueError(f"Expected a JSONL file, directory, or tar.gz archive: {path}")


@contextmanager
def seismic_document_input(path: Path, temporary_dir: Path | None = None) -> Iterator[Path]:
    """Return a SEISMIC-readable file, merging a shard directory when required."""
    if path.is_file():
        yield path
        return

    shards = vector_files(path)
    root = temporary_dir if temporary_dir else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="seismic-documents-", suffix=".jsonl", dir=root
    )
    os.close(descriptor)
    merged = Path(temporary_name)
    try:
        with merged.open("wb") as output:
            for shard in shards:
                with shard.open("rb") as source:
                    shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        yield merged
    finally:
        merged.unlink(missing_ok=True)


def stable_hash(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()
