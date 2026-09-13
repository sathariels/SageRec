#!/usr/bin/env python3
"""Thin CLI: time native graph_sampler vs the Python reference sampler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_sampler_benchmark as bench


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify native/reference sample_neighbors parity, then time both "
            "on the same ADR-005 without-replacement workload. Default graph "
            "is synthetic (no MovieLens download)."
        )
    )
    parser.add_argument(
        "--source",
        choices=("synthetic", "movielens-100k"),
        default="synthetic",
        help="synthetic is the default CI/published path; 100K needs processed/",
    )
    parser.add_argument("--num-users", type=int, default=512)
    parser.add_argument("--num-movies", type=int, default=1024)
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=4000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "processed",
        help="Used only with --source movielens-100k. Never downloads.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=_REPO_ROOT / "build",
        help="Release build dir (CMakeCache.txt supplies compiler/build type).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=_REPO_ROOT / "results" / "sampler_timing.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=_REPO_ROOT / "results" / "sampler_timing.md",
    )
    parser.add_argument(
        "--chart-output",
        type=Path,
        default=_REPO_ROOT / "results" / "sampler_timing.svg",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = bench.SamplerBenchmarkConfig(
        source=args.source,
        num_users=args.num_users,
        num_movies=args.num_movies,
        degree=args.degree,
        k=args.k,
        n_queries=args.n_queries,
        warmup_repetitions=args.warmup,
        measured_repetitions=args.repetitions,
        seed=args.seed,
        processed_dir=args.processed_dir if args.source == "movielens-100k" else None,
    )
    print(
        "config="
        + json.dumps(
            {
                "source": config.source,
                "num_users": config.num_users,
                "num_movies": config.num_movies,
                "degree": config.degree,
                "k": config.k,
                "n_queries": config.n_queries,
                "warmup_repetitions": config.warmup_repetitions,
                "measured_repetitions": config.measured_repetitions,
                "seed": config.seed,
                "processed_dir": str(config.processed_dir)
                if config.processed_dir
                else None,
            }
        ),
        flush=True,
    )
    payload = bench.run_benchmark(
        config,
        repo_root=_REPO_ROOT,
        build_dir=args.build_dir,
    )
    artifacts = bench.write_timing_artifacts(
        payload,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
        chart_path=args.chart_output,
        repo_root=_REPO_ROOT,
    )
    summary = {
        "parity_verified": artifacts.payload["parity_verified"],
        "timing_statistic": artifacts.payload["timing_statistic"]["name"],
        "native_median_seconds": artifacts.payload["native"]["seconds"]["median"],
        "reference_median_seconds": artifacts.payload["reference"]["seconds"]["median"],
        "speedup_native_over_reference": artifacts.payload["speedup"][
            "native_over_reference"
        ],
        "build_type": artifacts.payload["build_type"],
        "python": artifacts.payload["python"],
        "cpu": artifacts.payload["cpu"],
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {artifacts.json_path}")
    print(f"wrote {artifacts.markdown_path}")
    print(f"wrote {artifacts.chart_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except bench.BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
