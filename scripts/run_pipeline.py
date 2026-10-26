"""Kaggle/local launcher for the unified Hydra pipeline.

It installs only the lightweight Kaggle requirements when they are missing,
then forwards all unknown arguments as Hydra overrides.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


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


def main() -> None:
    args, overrides = parse_args()
    ensure_requirements(args.skip_install, args.wheelhouse)
    environment = os.environ.copy()
    environment.setdefault("PYTHONPATH", str(PROJECT_DIR))
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
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
