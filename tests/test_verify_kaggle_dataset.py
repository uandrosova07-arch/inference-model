from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_kaggle_dataset import find_dataset_root, validate_dataset


def _write_dataset(root: Path) -> Path:
    index = root / "seismic/msmarco.index.seismic"
    query_vectors = root / "vectors/queries.dev.jsonl"
    run = root / "seismic/run.dev.trec"
    shard = root / "vectors/documents/part-r00-00000.jsonl"
    index.parent.mkdir(parents=True)
    query_vectors.parent.mkdir(parents=True)
    shard.parent.mkdir(parents=True, exist_ok=True)
    index.write_bytes(b"index")
    query_vectors.write_text('{"id":"q1","vector":{"1":1.0}}\n', encoding="utf-8")
    run.write_text("q1 Q0 d1 1 1.0 seismic\n", encoding="utf-8")
    shard.write_text('{"id":"d1","vector":{"1":1.0}}\n', encoding="utf-8")
    manifest = {
        "run_id": "run",
        "configuration_id": "config",
        "artifacts": {
            "index_file": {"bytes": index.stat().st_size},
            "query_vectors": {"bytes": query_vectors.stat().st_size},
            "run": {"bytes": run.stat().st_size},
            "document_vectors": {"files": [{"path": shard.name}]},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_discovers_and_validates_dataset(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "nested/baseline")

    assert find_dataset_root(tmp_path) == root
    checked = validate_dataset(root)
    assert checked["document_vectors"]["shards"] == 1
    assert checked["document_vectors"]["layout"] == "flat"
