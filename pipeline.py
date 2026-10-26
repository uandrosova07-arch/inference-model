"""Unified configurable entry point for all retrieval experiments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# Hydra 1.3 uses a lazily rendered help object. Python 3.14's argparse started
# validating help text with string containment during parser construction.
if not getattr(argparse.HelpFormatter._expand_help, "_hydra_py314_compatible", False):
    _original_expand_help = argparse.HelpFormatter._expand_help

    def _expand_help_compatible(self, action):
        if not isinstance(action.help, str):
            return repr(action.help)
        return _original_expand_help(self, action)

    _expand_help_compatible._hydra_py314_compatible = True
    argparse.HelpFormatter._expand_help = _expand_help_compatible

import hydra
from omegaconf import DictConfig

from src.pipeline.runner import PipelineRunner
from src.utils.reproducibility import seed_everything


@hydra.main(version_base=None, config_path="src/configs", config_name="config")
def main(config: DictConfig) -> None:
    seed_everything(int(config.seed))
    logging.basicConfig(
        level=getattr(logging, str(config.runtime.log_level).upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    PipelineRunner(config, project_root=Path(__file__).resolve().parent).run()


if __name__ == "__main__":
    main()
