from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_pipeline import (
    discover_msmarco_collection,
    has_documents_override,
    override_value,
    should_discover_msmarco_collection,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _requirements(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _write_dataset(directory: Path, *, complete: bool = True) -> Path:
    directory.mkdir(parents=True)
    collection = directory / "collection.tsv"
    collection.touch()
    if complete:
        for name in ("queries.dev.tsv", "qrels.dev.tsv"):
            (directory / name).touch()
    return collection


def test_discovers_nested_kaggle_dataset(tmp_path: Path) -> None:
    collection = _write_dataset(tmp_path / "datasets" / "owner" / "msmarco")

    assert discover_msmarco_collection(tmp_path, {}) == collection.resolve()


def test_prefers_candidate_with_msmarco_companion_files(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "unrelated", complete=False)
    collection = _write_dataset(tmp_path / "datasets" / "owner" / "msmarco")

    assert discover_msmarco_collection(tmp_path, {}) == collection.resolve()


def test_rejects_ambiguous_collections(tmp_path: Path) -> None:
    first = _write_dataset(tmp_path / "first")
    second = _write_dataset(tmp_path / "second")

    with pytest.raises(RuntimeError, match="Multiple possible") as error:
        discover_msmarco_collection(tmp_path, {})

    assert str(first) in str(error.value)
    assert str(second) in str(error.value)


def test_explicit_collection_environment_variable(tmp_path: Path) -> None:
    collection = _write_dataset(tmp_path / "custom", complete=False)

    assert (
        discover_msmarco_collection(tmp_path, {"MSMARCO_COLLECTION": str(collection)})
        == collection.resolve()
    )


@pytest.mark.parametrize(
    "override",
    [
        "dataset.documents=/data/collection.tsv",
        "+dataset.documents=/data/collection.tsv",
        "++dataset.documents=/data/collection.tsv",
    ],
)
def test_detects_explicit_hydra_documents_override(override: str) -> None:
    assert has_documents_override([override])


def test_discovers_collection_for_document_stages_in_kaggle_runtime(tmp_path: Path) -> None:
    assert should_discover_msmarco_collection([], tmp_path)
    assert should_discover_msmarco_collection(["pipeline.start_at=encode_documents"], tmp_path)
    assert not should_discover_msmarco_collection(
        ["dataset.documents=/data/collection.tsv"], tmp_path
    )


@pytest.mark.parametrize("start_at", ["encode_queries", "build_index", "search", "metrics"])
def test_late_stage_run_does_not_require_collection(tmp_path: Path, start_at: str) -> None:
    assert not should_discover_msmarco_collection([f"pipeline.start_at={start_at}"], tmp_path)


def test_reads_last_hydra_override_value() -> None:
    assert override_value(["seed=1", "+seed=7"], "seed", "42") == "7"
    assert override_value([], "seed", "42") == "42"


def test_kaggle_requirements_match_local_pins() -> None:
    local = _requirements(PROJECT_DIR / "requirements.txt")
    kaggle = _requirements(PROJECT_DIR / "requirements-kaggle.txt")

    assert kaggle <= local
