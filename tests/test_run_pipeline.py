from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_pipeline import discover_msmarco_collection, has_documents_override, override_value


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


def test_reads_last_hydra_override_value() -> None:
    assert override_value(["seed=1", "+seed=7"], "seed", "42") == "7"
    assert override_value([], "seed", "42") == "42"
