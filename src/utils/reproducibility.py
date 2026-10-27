from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def installed_packages() -> dict[str, str]:
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = distribution.version
    return dict(sorted(packages.items()))


def _git(root: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def git_metadata(root: Path) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    remote = _git(root, "remote", "get-url", "origin")
    if remote and "://" in remote:
        parsed = urlsplit(remote)
        if parsed.username or parsed.password:
            host = parsed.hostname or ""
            if parsed.port:
                host = f"{host}:{parsed.port}"
            remote = urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "remote": remote,
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "packages": installed_packages(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "PYTHONHASHSEED",
                "TOKENIZERS_PARALLELISM",
            )
        },
    }
    try:
        import torch

        devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                        "capability": list(torch.cuda.get_device_capability(index)),
                    }
                )
        snapshot["torch_runtime"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_devices": devices,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
    except ImportError:
        snapshot["torch_runtime"] = None
    return snapshot
