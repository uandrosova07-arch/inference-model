from __future__ import annotations

import logging
import sys
from pathlib import Path

from omegaconf import DictConfig


def configure_logging(config: DictConfig, project_root: Path) -> Path:
    """Log to the notebook output and a persistent experiment file."""
    log_path = Path(str(config.paths.pipeline_log)).expanduser()
    if not log_path.is_absolute():
        log_path = project_root / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, str(config.runtime.log_level).upper())
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(processName)s | %(name)s | %(message)s"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)
    return log_path
