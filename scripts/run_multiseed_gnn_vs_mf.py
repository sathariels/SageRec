#!/usr/bin/env python3
"""Thin CLI: multi-seed MovieLens 100K MF vs GraphSAGE leaderboard (ADR-007).

Reuses the Phase 5 train/eval path (shared PairScorer metrics, native
``sagerec_minibatch`` neighborhoods, PyG SAGEConv). Does not overwrite
historical single-seed Phase 5 files. Default CI must not run this 100K job.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_baseline as baseline
import sagerec_dataset as dataset
import sagerec_graphsage as graphsage
import sagerec_multiseed as multiseed


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


def _parse_seeds(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    seeds: list[int] = []
    for part in parts:
        try:
            seeds.append(int(part, 10))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid seed {part!r}; expected comma-separated integers"
            ) from exc
    try:
        return multiseed.normalize_seeds(seeds)
    except multiseed.MultiseedError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_args() -> argparse.Namespace:
    default_seeds = ",".join(str(seed) for seed in multiseed.DEFAULT_SEEDS)
    parser = argparse.ArgumentParser(
        description=(
            "Train implicit MF and native-backed GraphSAGE across a seed list "
            "on prepared MovieLens 100K, then write mean±std leaderboard "
            "artifacts. Does not overwrite Phase 5 single-seed JSON files. "
            "Neighborhoods come from sagerec_minibatch / graph_sampler, not a "
            "PyG NeighborLoader."
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
        "--json-output",
        type=Path,
        default=_REPO_ROOT / "results" / "multiseed_gnn_vs_mf_movielens_100k.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=_REPO_ROOT / "results" / "multiseed_gnn_vs_mf_movielens_100k.md",
    )
    parser.add_argument(
        "--chart-output",
        type=Path,
        default=_REPO_ROOT / "results" / "multiseed_gnn_vs_mf_movielens_100k.svg",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "runs" / "multiseed",
        help="Gitignored per-seed JSON directory used to resume a long run.",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=multiseed.DEFAULT_SEEDS,
        help=(
            "Comma-separated seeds (must include 7). "
            f"Default: {default_seeds}"
        ),
    )
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
        "--no-resume",
        action="store_true",
        help="Retrain every seed even if work-dir JSON already exists.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print GraphSAGE training loss every N steps (0 disables).",
    )
    return parser.parse_args()


def _provenance() -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": graphsage.torch.__version__,
        "torch_geometric": graphsage.TORCH_GEOMETRIC_VERSION,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }


def main() -> int:
    args = _parse_args()
    seeds = multiseed.normalize_seeds(args.seeds)
    historical = multiseed.historical_single_seed_notes(_REPO_ROOT)
    split, manifest = dataset.ensure_movielens_100k_processed(
        args.processed_dir,
        args.raw_dir,
        download_if_missing=not args.no_download,
    )
    provenance = _provenance()
    mf_config = baseline.movielens_100k_config(seed=seeds[0])
    gnn_config = graphsage.movielens_100k_config(seed=seeds[0])
    print(
        "config="
        + json.dumps(
            {
                "seeds": list(seeds),
                "continuity_seed": multiseed.CONTINUITY_SEED,
                "k": args.k,
                "split": args.split,
                "mf": {
                    **mf_config.as_dict(),
                    "optimizer": "sgd",
                    "seed": "per-seed from --seeds",
                },
                "graphsage": {
                    **gnn_config.as_dict(),
                    "seed": "per-seed from --seeds",
                    "sampler": "sagerec_minibatch.NativeMinibatchSampler",
                    "conv_stack": "torch_geometric.nn.SAGEConv",
                    "training_seeds": graphsage.graphsage_training_seed_notes(
                        gnn_config.seed
                    ),
                },
                "historical_single_seed": historical,
            }
        ),
        flush=True,
    )

    pairs: list[tuple[dict, dict]] = []
    for seed in seeds:
        resumed = None if args.no_resume else multiseed.maybe_load_completed_pair(
            args.work_dir, seed
        )
        if resumed is not None:
            print(f"seed={seed} resuming from {args.work_dir}", flush=True)
            pairs.append(resumed)
            continue

        print(f"seed={seed} training implicit MF...", flush=True)
        mf_payload = multiseed.evaluate_mf_seed(
            split,
            manifest,
            seed=seed,
            k=args.k,
            target_split=args.split,
            provenance=provenance,
        )
        print(
            f"seed={seed} mf metrics="
            + json.dumps(mf_payload["metrics"]),
            flush=True,
        )

        def _progress(epoch: int, step: int, loss: float, *, _seed: int = seed) -> None:
            if args.log_every > 0 and step % args.log_every == 0:
                print(
                    f"seed={_seed} epoch={epoch} step={step} loss={loss:.6f}",
                    flush=True,
                )

        print(
            f"seed={seed} training GraphSAGE via native sample_neighbors...",
            flush=True,
        )
        gnn_payload = multiseed.evaluate_graphsage_seed(
            split,
            manifest,
            seed=seed,
            k=args.k,
            target_split=args.split,
            provenance=provenance,
            progress=_progress,
        )
        print(
            f"seed={seed} graphsage metrics="
            + json.dumps(gnn_payload["metrics"]),
            flush=True,
        )
        multiseed.write_seed_pair(args.work_dir, seed, mf_payload, gnn_payload)
        pairs.append((mf_payload, gnn_payload))

    artifacts = multiseed.write_multiseed_artifacts(
        pairs,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
        chart_path=args.chart_output,
        seeds=seeds,
        git_commit=str(provenance.get("git_commit")) if provenance.get("git_commit") else None,
        generated_at=str(provenance["generated_at"]),
        python_version=str(provenance["python"]),
        numpy_version=str(provenance["numpy"]),
        torch_version=str(provenance["torch"]),
        torch_geometric_version=str(provenance["torch_geometric"]),
        platform_info=dict(provenance["platform"]),
    )
    print(json.dumps(artifacts.payload["models"], indent=2))
    print(f"wrote {artifacts.json_path}")
    print(f"wrote {artifacts.markdown_path}")
    print(f"wrote {artifacts.chart_path}")
    print("historical single-seed files were not overwritten")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        dataset.MovieLensPrepError,
        TypeError,
        ValueError,
        multiseed.MultiseedError,
        argparse.ArgumentTypeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
