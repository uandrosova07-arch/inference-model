from __future__ import annotations

from src.datasets.msmarco import load_qrels, read_documents


def test_file_dataset_readers(tmp_path) -> None:
    documents = tmp_path / "documents.tsv"
    documents.write_text("d1\tfirst document\nd2\tsecond document\n", encoding="utf-8")
    assert list(read_documents(documents)) == [
        ("d1", "first document"),
        ("d2", "second document"),
    ]
    assert list(read_documents(documents, rank=1, world_size=2)) == [("d2", "second document")]

    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("q1\t0\td1\t1\nq1\t0\td2\t0\n", encoding="utf-8")
    assert load_qrels("file", qrels, "unused") == {"q1": {"d1": 1, "d2": 0}}
