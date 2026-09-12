#!/usr/bin/env python3
"""Thin CLI: seeded GraphSAGE on prepared MovieLens 100K (ADR-002/003/005)."""

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

import sagerec_dataset as dataset
import sagerec_graphsage as graphsage
import sagerec_metrics as metrics
import sagerec_minibatch as minibatch


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
    parser = argparse.ArgumentParser(
        description=(
            "Train native-backed GraphSAGE on prepared MovieLens 100K and "
            "write a provenance JSON. Neighborhoods come from "
            "sagerec_minibatch / graph_sampler, not a PyG NeighborLoader."
        )
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "results" / "graphsage_movielens_100k.json",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail if processed artifacts and u.data are both missing.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print training loss every N steps (0 disables).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    split, manifest = dataset.ensure_movielens_100k_processed(
        args.processed_dir,
        args.raw_dir,
        download_if_missing=not args.no_download,
    )
    config = graphsage.movielens_100k_config(seed=args.seed)
    print(
        "config="
        + json.dumps(
            {
                **config.as_dict(),
                "training_seeds": graphsage.graphsage_training_seed_notes(config.seed),
                "sampler": "sagerec_minibatch.NativeMinibatchSampler",
            }
        ),
        flush=True,
    )
    print(
        "training GraphSAGE on train-only pairs via native sample_neighbors...",
        flush=True,
    )

    def _progress(epoch: int, step: int, loss: float) -> None:
        if args.log_every > 0 and step % args.log_every == 0:
            print(f"epoch={epoch} step={step} loss={loss:.6f}", flush=True)

    model = graphsage.train_graphsage(split, config, progress=_progress)
    if not isinstance(model.sampler, minibatch.NativeMinibatchSampler):
        raise TypeError(
            "GraphSAGE 100K run requires NativeMinibatchSampler; "
            "refusing a non-native sampler."
        )
    print("materializing eval embeddings via native multi-hop sampling...", flush=True)
    model.materialize_eval_embeddings()
    print("evaluating ranking...", flush=True)
    report = metrics.evaluate_ranking(
        model, split, target_split=args.split, k=args.k
    )
    payload = graphsage.graphsage_result_payload(
        manifest=manifest,
        config=config,
        report=report,
        git_commit=_git_commit(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        torch_version=graphsage.torch.__version__,
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
    except (dataset.MovieLensPrepError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
