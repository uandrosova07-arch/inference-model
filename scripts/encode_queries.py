"""Encode the standard MS MARCO dev-subset queries for impact retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import SparseEncoder

from config import baseline_model_path
from scripts.index_baseline import encode_window

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "outputs" / "baseline" / "queries.dev.splade.tsv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics-name", default="msmarco-passage-dev-subset")
    parser.add_argument("--model", default=baseline_model_path())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    return parser.parse_args()


def read_builtin_topics(name: str) -> list[tuple[str, str]]:
    from pyserini.search import get_topics

    topics = get_topics(name)
    queries = []
    for query_id, value in topics.items():
        text = value.get("title", "") if isinstance(value, dict) else str(value)
        queries.append((str(query_id), text))
    return queries


def main() -> None:
    args = parse_args()
    queries = read_builtin_topics(args.topics_name)

    model = SparseEncoder(args.model, device=args.device)
    model.max_seq_length = args.max_length
    model.eval()
    encoded, stats = encode_window(
        model,
        queries,
        args.batch_size,
        args.max_length,
        args.precision,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for query_id, vector in encoded:
            weighted_tokens = " ".join(
                token for token, weight in vector.items() for _ in range(weight)
            )
            output.write(f"{query_id}\t{weighted_tokens}\n")
    temporary.replace(args.output)
    print(f"Encoded {len(encoded)} queries to {args.output}")
    print(stats.report())


if __name__ == "__main__":
    main()
