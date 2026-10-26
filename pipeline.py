"""Unified configurable entry point for all retrieval experiments."""

from __future__ import annotations

import argparse
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
from src.utils.logging import configure_logging
from src.utils.reproducibility import seed_everything


@hydra.main(version_base=None, config_path="src/configs", config_name="config")
def main(config: DictConfig) -> None:
    project_root = Path(__file__).resolve().parent
    configure_logging(config, project_root)
    seed_everything(int(config.seed))
    PipelineRunner(config, project_root=project_root).run()


if __name__ == "__main__":
    main()
