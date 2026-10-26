"""Launch independent index workers without a distributed rendezvous service."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, required=True)
    args, worker_args = parser.parse_known_args()
    if worker_args and worker_args[0] == "--":
        worker_args = worker_args[1:]
    return args, worker_args


def main() -> None:
    args, worker_args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    processes: list[subprocess.Popen] = []
    try:
        for rank in range(args.workers):
            environment = os.environ.copy()
            environment.update(
                RANK=str(rank),
                LOCAL_RANK=str(rank),
                WORLD_SIZE=str(args.workers),
                PYTHONPATH=str(PROJECT_DIR),
            )
            command = [sys.executable, "scripts/index_baseline.py", *worker_args]
            print(f"Starting rank {rank}: {' '.join(command)}", flush=True)
            processes.append(subprocess.Popen(command, cwd=PROJECT_DIR, env=environment))

        while True:
            statuses = [process.poll() for process in processes]
            failed = next(
                ((rank, code) for rank, code in enumerate(statuses) if code not in (None, 0)),
                None,
            )
            if failed:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    process.wait()
                rank, return_code = failed
                raise subprocess.CalledProcessError(return_code, f"index worker rank {rank}")
            if all(code == 0 for code in statuses):
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise


if __name__ == "__main__":
    main()
