"""Quick structural check for a mounted SPLADE/SEISMIC baseline Dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import jsonl_lines, vector_files  # noqa: I001


REQUIRED_ARTIFACTS = {
    "index_file": Path("seismic/msmarco.index.seismic"),
    "query_vectors": Path("vectors/queries.dev.jsonl"),
    "run": Path("seismic/run.dev.trec"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("/kaggle/input"),
        help="Dataset root, or a directory below which it can be discovered",
    )
    return parser.parse_args()


def find_dataset_root(path: Path) -> Path:
    path = path.resolve()
    if (path / "manifest.json").is_file():
        return path
    candidates = sorted(
        manifest.parent
        for manifest in path.rglob("manifest.json")
        if (manifest.parent / REQUIRED_ARTIFACTS["index_file"]).is_file()
        and (manifest.parent / REQUIRED_ARTIFACTS["query_vectors"]).is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No mounted SPLADE/SEISMIC baseline found below {path}")
    rendered = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise RuntimeError(f"Multiple baseline Datasets found; pass one explicitly:\n{rendered}")


def expected_vector_shards(manifest: dict[str, Any]) -> int | None:
    files = manifest.get("artifacts", {}).get("document_vectors", {}).get("files")
    if not isinstance(files, list):
        return None
    return sum(
        isinstance(item, dict) and str(item.get("path", "")).endswith((".jsonl", ".jsonl.gz"))
        for item in files
    )


def validate_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})

    checked: dict[str, Any] = {
        "root": str(root),
        "run_id": manifest.get("run_id"),
        "configuration_id": manifest.get("configuration_id"),
    }
    for name, relative in REQUIRED_ARTIFACTS.items():
        artifact = root / relative
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        actual_bytes = artifact.stat().st_size
        expected_bytes = artifacts.get(name, {}).get("bytes")
        if expected_bytes is not None and actual_bytes != int(expected_bytes):
            raise RuntimeError(
                f"{relative} has {actual_bytes:,} bytes, manifest expects {int(expected_bytes):,}"
            )
        checked[name] = {"path": str(artifact), "bytes": actual_bytes}

    vectors_path = root / "vectors/documents"
    shards = vector_files(vectors_path)
    expected_shards = expected_vector_shards(manifest)
    if expected_shards is not None and len(shards) != expected_shards:
        raise RuntimeError(
            f"Found {len(shards):,} document-vector shards, manifest expects {expected_shards:,}"
        )
    first_record = json.loads(next(jsonl_lines(shards[0])))
    if "id" not in first_record or "vector" not in first_record:
        raise ValueError(f"First vector record in {shards[0]} has no id/vector")
    layout = "kaggle-expanded-gzip" if shards[0].name.endswith(".jsonl.gz.tmp") else "flat"
    checked["document_vectors"] = {
        "path": str(vectors_path),
        "shards": len(shards),
        "layout": layout,
    }
    return checked


def main() -> None:
    root = find_dataset_root(parse_args().path)
    checked = validate_dataset(root)
    print(json.dumps(checked, ensure_ascii=False, indent=2))
    print("OK: build_index, search, and metrics inputs are present", flush=True)


if __name__ == "__main__":
    main()
