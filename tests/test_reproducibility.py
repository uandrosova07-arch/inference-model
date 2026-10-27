from __future__ import annotations

from src.utils.io import file_fingerprint, path_fingerprint, path_inventory


def test_file_fingerprint_counts_unterminated_last_line(tmp_path) -> None:
    path = tmp_path / "data.tsv"
    path.write_bytes(b"a\nb")

    fingerprint = file_fingerprint(path, include_lines=True)

    assert fingerprint["bytes"] == 3
    assert fingerprint["lines"] == 2
    assert len(fingerprint["sha256"]) == 64


def test_directory_fingerprint_changes_with_content(tmp_path) -> None:
    directory = tmp_path / "artifacts"
    directory.mkdir()
    artifact = directory / "part.jsonl"
    artifact.write_text("first\n", encoding="utf-8")
    before = path_fingerprint(directory)
    inventory_before = path_inventory(directory)

    artifact.write_text("second\n", encoding="utf-8")
    after = path_fingerprint(directory)
    inventory_after = path_inventory(directory)

    assert before["sha256"] != after["sha256"]
    assert inventory_before != inventory_after
