#!/usr/bin/env python3
"""Thin CLI: tiny synthetic GraphSAGE protocol smoke (not MovieLens 100K)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_graphsage as graphsage
import sagerec_metrics as metrics
import sagerec_prep as prep


def _row(
    user_id: int, movie_id: int, timestamp: int, rating: int = 1
) -> prep.NormalizedInteraction:
    return prep.NormalizedInteraction(user_id, movie_id, timestamp, rating)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train GraphSAGE on a tiny synthetic ADR-003 split and print "
            "protocol-smoke Recall@10 / NDCG@10. This is not a MovieLens 100K "
            "result and must not be stored as one."
        )
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    interactions = [
        _row(0, 0, 10),
        _row(0, 1, 20),
        _row(0, 2, 30),
        _row(0, 3, 40),
        _row(0, 4, 50),
        _row(1, 5, 11),
        _row(1, 6, 21),
        _row(1, 7, 31),
        _row(1, 8, 41),
        _row(2, 0, 1),
        _row(2, 9, 2),
    ]
    split = prep.split_interactions(interactions, num_users=3, num_movies=20)
    config = graphsage.GraphSAGEConfig(
        embedding_dim=4,
        hidden_dim=4,
        n_layers=2,
        fanouts=(3, 2),
        n_epochs=2,
        batch_size=4,
        learning_rate=0.1,
        n_negatives=2,
        l2=1e-4,
        seed=args.seed,
    )
    print(
        "GraphSAGE synthetic protocol smoke "
        f"(not MovieLens 100K) seed={config.seed} fanouts={config.fanouts}",
        flush=True,
    )
    model = graphsage.train_graphsage(split, config)
    report = metrics.evaluate_ranking(model, split, target_split="test", k=args.k)
    payload = {
        "label": "protocol_smoke_synthetic",
        "not_movielens_100k": True,
        "seed": config.seed,
        "k": report.k,
        "split": report.split,
        "n_users": report.n_users,
        "recall_at_k": report.recall_at_k,
        "ndcg_at_k": report.ndcg_at_k,
        "sampler": "graph_sampler via sagerec_minibatch",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
