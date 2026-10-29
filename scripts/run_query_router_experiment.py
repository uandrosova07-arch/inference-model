"""Run the complete query-adaptive query_cut router experiment on Kaggle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_pipeline import ensure_requirements

from src.utils.io import atomic_write_json

DEFAULT_BASELINE_ROOT = Path(
    "/kaggle/input/datasets/iskulovdamir/msmarco-splade-seismic"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--experiment-prefix", default="query_router_v1")
    parser.add_argument("--query-cuts", default="6,8,10")
    parser.add_argument("--baseline-query-cut", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=64)
    parser.add_argument("--leaf-safety-quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument(
        "--skip-fingerprints",
        action="store_true",
        help="Use only for iteration; final experiment manifests should retain SHA256 fingerprints.",
    )
    return parser.parse_args()


def _required_baseline_paths(root: Path) -> dict[str, Path]:
    return {
        "manifest": root / "manifest.json",
        "index": root / "seismic" / "msmarco.index.seismic",
        "queries": root / "vectors" / "queries.dev.jsonl",
        "run": root / "seismic" / "run.dev.trec",
        "report": root / "report.json",
    }


def _run(command: list[str], environment: dict[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.check_call(command, cwd=PROJECT_DIR, env=environment)


def _search_seconds(report_path: Path) -> float | None:
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    value = report.get("stages", {}).get("search", {}).get("search_seconds")
    return float(value) if value is not None else None


def main() -> None:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    baseline = _required_baseline_paths(baseline_root)
    missing = [path for path in baseline.values() if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Baseline dataset is incomplete:\n{formatted}")

    query_cuts = sorted({int(value) for value in args.query_cuts.split(",")})
    if any(value <= 0 for value in query_cuts):
        raise ValueError("query cuts must be positive")
    if args.baseline_query_cut != 10:
        raise ValueError("query_router_v1 expects the published baseline query_cut=10")
    if args.baseline_query_cut not in query_cuts:
        raise ValueError("baseline-query-cut must be included in query-cuts")
    if query_cuts[-1] != args.baseline_query_cut:
        raise ValueError("baseline-query-cut must be the safest and largest candidate")

    ensure_requirements(args.skip_install, args.wheelhouse)
    environment = os.environ.copy()
    environment.setdefault("PYTHONPATH", str(PROJECT_DIR))
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("PYTHONHASHSEED", str(args.seed))

    output_root = PROJECT_DIR / "outputs"
    model_dir = output_root / f"{args.experiment_prefix}_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    policies: list[dict[str, Any]] = []
    for query_cut in query_cuts:
        name = f"qc{query_cut}"
        parameters = {
            "query_cut": query_cut,
            "heap_factor": 0.8,
            "n_knn": 0,
            "first_sorted": False,
        }
        if query_cut == args.baseline_query_cut:
            policies.append(
                {
                    "name": name,
                    "parameters": parameters,
                    "run": str(baseline["run"]),
                    "report": str(baseline["report"]),
                    "search_seconds": _search_seconds(baseline["report"]),
                    "source": "published_baseline",
                }
            )
            continue

        experiment_name = f"{args.experiment_prefix}_candidate_qc{query_cut}"
        experiment_dir = output_root / experiment_name
        change_note = (
            f"query_cut_{args.baseline_query_cut}_to_{query_cut}_for_router_training"
        )
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "run_pipeline.py"),
            "--skip-install",
            "--config-name=kaggle",
            f"experiment.name={experiment_name}",
            "experiment.kind=query_router_candidate",
            f"experiment.parent_manifest={baseline['manifest']}",
            f"experiment.change_note={change_note}",
            "pipeline.start_at=search",
            f"paths.index_file={baseline['index']}",
            f"paths.query_vectors={baseline['queries']}",
            f"index.threads={args.threads}",
            f"index.search.query_cut={query_cut}",
            "index.search.heap_factor=0.8",
            "index.search.n_knn=0",
            "index.search.first_sorted=false",
            "runtime.require_cuda=false",
        ]
        if args.skip_fingerprints:
            command.append("runtime.fingerprint_artifacts=false")
        _run(command, environment)
        report_path = experiment_dir / "report.json"
        policies.append(
            {
                "name": name,
                "parameters": parameters,
                "run": str(experiment_dir / "seismic" / "run.dev.trec"),
                "report": str(report_path),
                "search_seconds": _search_seconds(report_path),
                "source": "candidate_search",
            }
        )

    plan_path = model_dir / "training_plan.json"
    atomic_write_json(
        plan_path,
        {
            "schema_version": 1,
            "experiment_prefix": args.experiment_prefix,
            "baseline_root": str(baseline_root),
            "policies": policies,
        },
    )
    model_path = model_dir / "router.json"
    training_report = model_dir / "training_report.json"
    training_assignments = model_dir / "training_assignments.jsonl"
    test_qrels = model_dir / "test.qrels.tsv"
    train_command = [
        sys.executable,
        str(SCRIPTS_DIR / "train_query_router.py"),
        "--plan",
        str(plan_path),
        "--query-vectors",
        str(baseline["queries"]),
        "--output-model",
        str(model_path),
        "--output-report",
        str(training_report),
        "--output-assignments",
        str(training_assignments),
        "--output-test-qrels",
        str(test_qrels),
        "--seed",
        str(args.seed),
        "--max-depth",
        str(args.max_depth),
        "--min-samples-leaf",
        str(args.min_samples_leaf),
        "--leaf-safety-quantile",
        str(args.leaf_safety_quantile),
    ]
    _run(train_command, environment)

    adaptive_name = f"{args.experiment_prefix}_adaptive"
    adaptive_command = [
        sys.executable,
        str(SCRIPTS_DIR / "run_pipeline.py"),
        "--skip-install",
        "--config-name=kaggle",
        f"experiment.name={adaptive_name}",
        "experiment.kind=query_router",
        f"experiment.parent_manifest={baseline['manifest']}",
        f"experiment.change_note=depth{args.max_depth}_query_router_selects_query_cut_"
        + "_".join(str(value) for value in query_cuts),
        "pipeline.start_at=search",
        f"paths.index_file={baseline['index']}",
        f"paths.query_vectors={baseline['queries']}",
        f"index.threads={args.threads}",
        "index.search.query_cut=10",
        "index.search.heap_factor=0.8",
        "index.search.n_knn=0",
        "index.search.first_sorted=false",
        "index.search.router.enabled=true",
        f"index.search.router.model_path={model_path}",
        "runtime.require_cuda=false",
    ]
    if args.skip_fingerprints:
        adaptive_command.append("runtime.fingerprint_artifacts=false")
    _run(adaptive_command, environment)

    adaptive_dir = output_root / adaptive_name
    summary = {
        "schema_version": 1,
        "baseline_root": str(baseline_root),
        "query_cuts": query_cuts,
        "training_plan": str(plan_path),
        "router_model": str(model_path),
        "training_report": str(training_report),
        "training_assignments": str(training_assignments),
        "test_qrels": str(test_qrels),
        "adaptive_output": str(adaptive_dir),
        "adaptive_report": str(adaptive_dir / "report.json"),
        "adaptive_manifest": str(adaptive_dir / "manifest.json"),
    }
    summary_path = model_dir / "experiment_summary.json"
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
