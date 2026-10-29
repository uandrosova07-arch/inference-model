"""Kaggle/local launcher for the unified Hydra pipeline.

It installs only the lightweight Kaggle requirements when they are missing,
then forwards all unknown arguments as Hydra overrides.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
KAGGLE_INPUT_DIR = Path("/kaggle/input")
MSMARCO_COMPANION_FILES = ("queries.dev.tsv", "qrels.dev.tsv")
DOCUMENT_STAGES = ("prepare", "encode_documents")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--config-name", default="kaggle" if Path("/kaggle").exists() else "config")
    return parser.parse_known_args()


def ensure_requirements(skip: bool, wheelhouse: Path | None) -> None:
    if skip:
        return
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
    ]
    if wheelhouse:
        command.extend(("--no-index", "--find-links", str(wheelhouse.resolve())))
    command.extend(("-r", str(PROJECT_DIR / "requirements-kaggle.txt")))
    subprocess.check_call(command)


def has_documents_override(overrides: Sequence[str]) -> bool:
    """Return whether Hydra arguments already specify the collection path."""
    return any(
        override.lstrip("+").partition("=")[0] == "dataset.documents" for override in overrides
    )


def should_discover_msmarco_collection(
    overrides: Sequence[str], input_dir: Path = KAGGLE_INPUT_DIR
) -> bool:
    """Return whether a Kaggle run needs automatic collection discovery."""
    start_at = override_value(overrides, "pipeline.start_at", "prepare")
    return (
        input_dir.is_dir() and start_at in DOCUMENT_STAGES and not has_documents_override(overrides)
    )


def override_value(overrides: Sequence[str], name: str, default: str) -> str:
    for override in reversed(overrides):
        key, separator, value = override.lstrip("+").partition("=")
        if separator and key == name:
            return value
    return default


def discover_msmarco_collection(
    input_dir: Path = KAGGLE_INPUT_DIR,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Find an unambiguous MS MARCO collection in Kaggle's mounted datasets."""
    environment = os.environ if environment is None else environment
    configured = environment.get("MSMARCO_COLLECTION")
    if configured:
        collection = Path(configured)
        if not collection.is_file():
            raise FileNotFoundError(f"MSMARCO_COLLECTION points to a missing file: {collection}")
        return collection.resolve()

    conventional = input_dir / "msmarco" / "collection.tsv"
    if conventional.is_file():
        return conventional.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Kaggle input directory does not exist: {input_dir}. "
            "Attach the MS MARCO dataset or pass dataset.documents=/path/to/collection.tsv."
        )

    candidates = sorted(path for path in input_dir.rglob("collection.tsv") if path.is_file())
    complete_candidates = [
        path
        for path in candidates
        if all((path.parent / name).is_file() for name in MSMARCO_COMPANION_FILES)
    ]
    choices = complete_candidates or candidates
    if len(choices) == 1:
        return choices[0].resolve()
    if not choices:
        raise FileNotFoundError(
            f"No collection.tsv found below {input_dir}. "
            "Attach the MS MARCO dataset or pass dataset.documents=/path/to/collection.tsv."
        )

    formatted = "\n".join(f"  - {path}" for path in choices)
    raise RuntimeError(
        "Multiple possible MS MARCO collections found. Pass an explicit Hydra override "
        f"dataset.documents=/path/to/collection.tsv:\n{formatted}"
    )


def main() -> None:
    args, overrides = parse_args()
    environment = os.environ.copy()
    if should_discover_msmarco_collection(overrides):
        collection = discover_msmarco_collection(environment=environment)
        environment["MSMARCO_COLLECTION"] = str(collection)
        print(f"MS MARCO collection: {collection}", flush=True)
    ensure_requirements(args.skip_install, args.wheelhouse)
    environment.setdefault("PYTHONPATH", str(PROJECT_DIR))
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("PYTHONHASHSEED", override_value(overrides, "seed", "42"))
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if Path("/kaggle/working").exists():
        environment.setdefault("HF_HOME", "/kaggle/working/.cache/huggingface")
    subprocess.check_call(
        [
            sys.executable,
            str(PROJECT_DIR / "pipeline.py"),
            f"--config-name={args.config_name}",
            *overrides,
        ],
        cwd=PROJECT_DIR,
        env=environment,
    )


if __name__ == "__main__":
    main()
