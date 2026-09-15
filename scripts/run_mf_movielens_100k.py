#!/usr/bin/env python3
"""Thin CLI: seeded implicit MF on prepared MovieLens 100K (ADR-003/004)."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_baseline as baseline
import sagerec_dataset as dataset
import sagerec_metrics as metrics


def _git_commit() -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def _parse_args() -> argparse.Namespace:
    defaults = baseline.movielens_100k_config()
    parser = argparse.ArgumentParser(
        description="Train implicit MF on prepared MovieLens 100K and write results."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "results" / "mf_movielens_100k.json",
    )
    parser.add_argument("--n-factors", type=int, default=defaults.n_factors)
    parser.add_argument("--n-epochs", type=int, default=defaults.n_epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--n-negatives", type=int, default=defaults.n_negatives)
    parser.add_argument("--l2", type=float, default=defaults.l2)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    split = dataset.load_split_from_processed(args.processed_dir)
    manifest = dataset.load_manifest(args.processed_dir)
    config = baseline.MFConfig(
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        n_negatives=args.n_negatives,
        l2=args.l2,
        seed=args.seed,
    )
    print(
        "config="
        + json.dumps({"seed": config.seed, **config.as_dict()}),
        flush=True,
    )
    print("training implicit MF on train-only pairs...", flush=True)
    model = baseline.train_implicit_mf(split, config)
    print("evaluating ranking...", flush=True)
    report = metrics.evaluate_ranking(
        model, split, target_split=args.split, k=args.k
    )
    payload = baseline.mf_result_payload(
        manifest=manifest,
        config=config,
        report=report,
        git_commit=_git_commit(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        platform_info={
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (dataset.MovieLensPrepError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
