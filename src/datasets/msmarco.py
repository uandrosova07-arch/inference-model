from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def _read_tsv(path: Path, limit: int | None = None) -> Iterator[tuple[str, str]]:
    with path.open(encoding="utf-8") as source:
        for row_number, line in enumerate(source, 1):
            if limit is not None and row_number > limit:
                break
            try:
                identifier, text = line.rstrip("\n").split("\t", 1)
            except ValueError as error:
                raise ValueError(f"Invalid two-column TSV row {row_number} in {path}") from error
            yield identifier, text


def read_documents(
    path: Path,
    limit: int | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[tuple[str, str]]:
    for index, row in enumerate(_read_tsv(path, limit)):
        if index % world_size == rank:
            yield row


def read_queries(
    source: str,
    path: Path,
    topics_name: str,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    if source == "file":
        return list(_read_tsv(path, limit))
    if source != "pyserini":
        raise ValueError(f"Unsupported query_source={source!r}")

    from pyserini.search import get_topics

    result = []
    for query_id, value in get_topics(topics_name).items():
        text = value.get("title", "") if isinstance(value, dict) else str(value)
        result.append((str(query_id), text))
        if limit is not None and len(result) >= limit:
            break
    return result


def load_qrels(source: str, path: Path, qrels_name: str) -> dict[str, dict[str, int]]:
    if source == "pyserini":
        from pyserini.search import get_qrels

        return {
            str(query_id): {str(doc_id): int(relevance) for doc_id, relevance in docs.items()}
            for query_id, docs in get_qrels(qrels_name).items()
        }
    if source != "file":
        raise ValueError(f"Unsupported qrels_source={source!r}")

    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as input_file:
        for row_number, line in enumerate(input_file, 1):
            fields = line.split()
            if len(fields) == 4:
                query_id, _, document_id, relevance = fields
            elif len(fields) == 3:
                query_id, document_id, relevance = fields
            else:
                raise ValueError(f"Invalid qrels row {row_number} in {path}")
            qrels.setdefault(str(query_id), {})[str(document_id)] = int(relevance)
    return qrels
