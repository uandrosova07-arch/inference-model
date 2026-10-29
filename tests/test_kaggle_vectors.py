from __future__ import annotations

import gzip
import io
from pathlib import Path

from src.utils.io import jsonl_lines, seismic_document_input, vector_files


def test_reads_flat_gzip_vector_shards(tmp_path: Path) -> None:
    shard = tmp_path / "part-r00-00000.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as output:
        output.write('{"id":"d1","vector":{"1":1.0}}\n')

    assert vector_files(tmp_path) == [shard]
    assert list(jsonl_lines(tmp_path)) == ['{"id":"d1","vector":{"1":1.0}}\n']


def test_reads_kaggle_expanded_gzip_vector_shards(tmp_path: Path) -> None:
    extracted = tmp_path / "part-r00-00000.jsonl" / "part-r00-00000.jsonl.gz.tmp"
    extracted.parent.mkdir()
    extracted.write_text('{"id":"d1","vector":{"1":1.0}}\n', encoding="utf-8")
    unrelated = tmp_path / "unfinished.jsonl.gz.tmp"
    unrelated.write_text("partial", encoding="utf-8")

    assert vector_files(tmp_path) == [extracted]
    assert list(jsonl_lines(tmp_path)) == ['{"id":"d1","vector":{"1":1.0}}\n']


def test_streams_kaggle_expanded_shards_to_seismic(tmp_path: Path) -> None:
    documents = tmp_path / "vectors"
    first = documents / "part-r00-00000.jsonl" / "part-r00-00000.jsonl.gz.tmp"
    second = documents / "part-r00-00001.jsonl" / "part-r00-00001.jsonl.gz.tmp"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")

    with seismic_document_input(documents, tmp_path, shard_input_mode="merge") as merged:
        assert merged.read_bytes() == io.BytesIO(b"first\nsecond\n").getvalue()
