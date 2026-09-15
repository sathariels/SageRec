"""ADR-007 multi-seed aggregation and wiring tests.

Aggregation tests use fixture JSON only (no invented 100K quality numbers).
The tiny synthetic smoke trains MF + GraphSAGE on in-memory ADR-003 data.
Default CI must not download MovieLens or run the 100K leaderboard job.
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

_PYTHON_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_baseline as baseline
import sagerec_compare as compare
import sagerec_graphsage as graphsage
import sagerec_multiseed as multiseed
import sagerec_prep as prep


def _row(
    user_id: int, movie_id: int, timestamp: int, rating: int = 1
) -> prep.NormalizedInteraction:
    return prep.NormalizedInteraction(user_id, movie_id, timestamp, rating)


def _synthetic_split() -> prep.SplitResult:
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
    return prep.split_interactions(interactions, num_users=3, num_movies=20)


def _tiny_graphsage_config(seed: int) -> graphsage.GraphSAGEConfig:
    return graphsage.GraphSAGEConfig(
        embedding_dim=4,
        hidden_dim=4,
        n_layers=2,
        fanouts=(3, 2),
        n_epochs=1,
        batch_size=4,
        learning_rate=0.1,
        n_negatives=2,
        l2=1e-4,
        seed=seed,
    )


def _tiny_mf_config(seed: int) -> baseline.MFConfig:
    return baseline.MFConfig(
        n_factors=4,
        n_epochs=1,
        learning_rate=0.05,
        n_negatives=1,
        l2=0.01,
        seed=seed,
    )


def _synthetic_manifest(split: prep.SplitResult) -> dict:
    return {
        "dataset_edition": "100k",
        "source_url": "https://example.test/ml-100k.zip",
        "checksum": "md5:deadbeef",
        "license": "test",
        "split_policy_id": "per_user_chronological_leave_one_out",
        "split_policy_version": "adr-003-v1",
        "min_interactions_for_eval": 3,
        "cold_start_policy": "all_to_train_exclude_from_eval",
        "counts": {
            "users": split.num_users,
            "movies": split.num_movies,
            "interactions": {"total": 11, "train": 6, "validation": 2, "test": 2},
            "eligible_users": len(split.eligible_user_ids),
            "cold_start_users": len(split.cold_start_user_ids),
        },
    }


def _fixture_result(
    *,
    model: str,
    recall: float,
    ndcg: float,
    n_epochs: int,
    n_users: int = 943,
    k: int = 10,
    split: str = "test",
    seed: int = 7,
    optimizer: str | None = None,
) -> dict:
    hyperparams: dict = {"n_epochs": n_epochs, "seed": seed, "learning_rate": 0.05}
    if optimizer is not None:
        hyperparams["optimizer"] = optimizer
    return {
        "model": model,
        "dataset_edition": "100k",
        "source_url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "checksum": "md5:0e33842e24a9c977be4e0107933c0723",
        "license": "test-license",
        "split_policy_id": "per_user_chronological_leave_one_out",
        "split_policy_version": "adr-003-v1",
        "min_interactions_for_eval": 3,
        "cold_start_policy": "all_to_train_exclude_from_eval",
        "seed": seed,
        "hyperparams": hyperparams,
        "metrics": {
            "k": k,
            "split": split,
            "recall_at_k": recall,
            "ndcg_at_k": ndcg,
            "n_evaluated_users": n_users,
        },
        "eligibility": {
            "users": n_users,
            "movies": 1682,
            "interactions": {
                "total": 100000,
                "train": 98114,
                "validation": n_users,
                "test": n_users,
            },
            "eligible_users": n_users,
            "cold_start_users": 0,
            "evaluated_users": n_users,
        },
        "note": "fixture",
        "sampler": (
            "graph_sampler via sagerec_minibatch" if model == "graphsage" else None
        ),
        "conv_stack": (
            "torch_geometric.nn.SAGEConv on native samples"
            if model == "graphsage"
            else None
        ),
    }


def _fixture_pairs() -> list[tuple[dict, dict]]:
    seeds = (7, 11, 13)
    mf_recalls = (0.10, 0.20, 0.30)
    mf_ndcgs = (0.04, 0.05, 0.06)
    gnn_recalls = (0.12, 0.18, 0.24)
    gnn_ndcgs = (0.05, 0.07, 0.09)
    pairs = []
    for seed, mf_r, mf_n, gnn_r, gnn_n in zip(
        seeds, mf_recalls, mf_ndcgs, gnn_recalls, gnn_ndcgs
    ):
        pairs.append(
            (
                _fixture_result(
                    model="implicit_mf",
                    recall=mf_r,
                    ndcg=mf_n,
                    n_epochs=1,
                    seed=seed,
                ),
                _fixture_result(
                    model="graphsage",
                    recall=gnn_r,
                    ndcg=gnn_n,
                    n_epochs=3,
                    seed=seed,
                    optimizer="adam",
                ),
            )
        )
    return pairs


class SeedListTests(unittest.TestCase):
    def test_default_seeds_include_seven_and_are_modest(self) -> None:
        seeds = multiseed.normalize_seeds()
        self.assertEqual(seeds, (7, 11, 13, 17, 19))
        self.assertEqual(len(seeds), 5)
        self.assertIn(multiseed.CONTINUITY_SEED, seeds)

    def test_rejects_missing_continuity_seed_or_duplicates(self) -> None:
        with self.assertRaises(multiseed.MultiseedError) as missing:
            multiseed.normalize_seeds((11, 13, 17, 19, 23))
        self.assertIn("7", str(missing.exception))
        with self.assertRaises(multiseed.MultiseedError):
            multiseed.normalize_seeds((7,))
        with self.assertRaises(multiseed.MultiseedError) as dup:
            multiseed.normalize_seeds((7, 11, 7))
        self.assertIn("unique", str(dup.exception))


class AggregationMathTests(unittest.TestCase):
    def test_metric_moments_match_statistics_module(self) -> None:
        values = [0.10, 0.20, 0.30, 0.40, 0.50]
        moments = multiseed.metric_moments(values)
        self.assertEqual(moments["n"], 5)
        self.assertAlmostEqual(moments["mean"], statistics.fmean(values))
        self.assertAlmostEqual(moments["std"], statistics.stdev(values))
        self.assertAlmostEqual(moments["median"], statistics.median(values))
        self.assertEqual(moments["min"], 0.10)
        self.assertEqual(moments["max"], 0.50)
        self.assertEqual(moments["values"], values)

    def test_build_comparison_aggregates_per_seed_rows(self) -> None:
        pairs = _fixture_pairs()
        payload = multiseed.build_multiseed_comparison(
            pairs, seeds=(7, 11, 13)
        )
        self.assertEqual(payload["comparison"], "multiseed_gnn_vs_mf")
        self.assertEqual(payload["protocol_version"], "adr-007-v1")
        self.assertEqual(payload["seeds"], [7, 11, 13])
        self.assertEqual(payload["continuity_seed"], 7)
        self.assertEqual(payload["n_evaluated_users"], 943)
        self.assertEqual(payload["k"], 10)
        mf_recall = payload["models"]["implicit_mf"]["recall_at_k"]
        self.assertAlmostEqual(mf_recall["mean"], 0.20)
        self.assertAlmostEqual(mf_recall["std"], statistics.stdev([0.10, 0.20, 0.30]))
        self.assertAlmostEqual(mf_recall["median"], 0.20)
        gnn_recall = payload["models"]["graphsage"]["recall_at_k"]
        self.assertAlmostEqual(gnn_recall["mean"], 0.18)
        self.assertEqual(len(payload["per_seed"]), 3)
        self.assertEqual(payload["per_seed"][0]["seed"], 7)
        self.assertEqual(payload["per_seed"][0]["implicit_mf"]["recall_at_k"], 0.10)
        self.assertEqual(payload["per_seed"][1]["graphsage"]["recall_at_k"], 0.18)
        self.assertIn("shared eval protocol", payload["protocol_note"].lower())
        self.assertIn("not identical wall-clock", payload["protocol_note"].lower())
        self.assertIn("SAGEConv", payload["protocol_note"])
        self.assertIn("NeighborLoader", payload["protocol_note"])
        self.assertEqual(
            payload["historical_single_seed"]["graphsage"],
            "results/graphsage_movielens_100k.json",
        )
        self.assertNotEqual(
            gnn_recall["mean"],
            0.04772004241781548,
        )

    def test_write_artifacts_include_error_bars_and_mean_std(self) -> None:
        pairs = _fixture_pairs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = multiseed.write_multiseed_artifacts(
                pairs,
                json_path=root / "board.json",
                markdown_path=root / "board.md",
                chart_path=root / "board.svg",
                seeds=(7, 11, 13),
            )
            loaded = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["n_seeds"], 3)
            markdown = artifacts.markdown_path.read_text(encoding="utf-8")
            self.assertIn("mean±std", markdown)
            self.assertIn("0.200000 ±", markdown)
            self.assertIn("Per-seed rows", markdown)
            self.assertIn("not overwritten", markdown.lower())
            svg = artifacts.chart_path.read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertIn("sample std", svg.lower())
            self.assertIn("±", svg)
            self.assertIn("implicit MF", svg)
            self.assertIn("GraphSAGE", svg)
            self.assertIn('stroke="#333"', svg)

    def test_rejects_protocol_or_hyperparam_mismatch_across_seeds(self) -> None:
        pairs = _fixture_pairs()
        pairs[1][0]["metrics"]["n_evaluated_users"] = 900
        pairs[1][0]["eligibility"]["evaluated_users"] = 900
        pairs[1][0]["eligibility"]["eligible_users"] = 900
        with self.assertRaises(multiseed.MultiseedError):
            multiseed.build_multiseed_comparison(pairs, seeds=(7, 11, 13))

        pairs = _fixture_pairs()
        pairs[2][1]["hyperparams"]["n_epochs"] = 9
        with self.assertRaises(multiseed.MultiseedError) as ctx:
            multiseed.build_multiseed_comparison(pairs, seeds=(7, 11, 13))
        self.assertIn("hyperparameters", str(ctx.exception))


class HistoricalSingleSeedTests(unittest.TestCase):
    def test_phase5_files_remain_labeled_single_seed(self) -> None:
        notes = multiseed.historical_single_seed_notes(_REPO_ROOT)
        mf_path = _REPO_ROOT / "results" / "mf_movielens_100k.json"
        gnn_path = _REPO_ROOT / "results" / "graphsage_movielens_100k.json"
        comparison_path = _REPO_ROOT / "results" / "gnn_vs_mf_movielens_100k.json"
        mf = compare.load_ranking_result(mf_path)
        gnn = compare.load_ranking_result(gnn_path)
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        self.assertEqual(mf["seed"], 7)
        self.assertEqual(gnn["seed"], 7)
        self.assertIn("single-seed", mf["note"].lower())
        self.assertIn("not a multi-seed leaderboard", mf["note"].lower())
        self.assertIn("single-seed", gnn["note"].lower())
        self.assertIn("not a multi-seed leaderboard", gnn["note"].lower())
        self.assertIn("single-seed", comparison["protocol_note"].lower())
        self.assertIn("multi-seed leaderboard", comparison["note"].lower())
        self.assertAlmostEqual(mf["metrics"]["recall_at_k"], 0.03817603393425239)
        self.assertAlmostEqual(mf["metrics"]["ndcg_at_k"], 0.020303175177886473)
        self.assertAlmostEqual(gnn["metrics"]["recall_at_k"], 0.04772004241781548)
        self.assertAlmostEqual(gnn["metrics"]["ndcg_at_k"], 0.020968588363147352)
        self.assertEqual(notes["implicit_mf"]["path"], "results/mf_movielens_100k.json")
        self.assertEqual(
            notes["graphsage"]["path"], "results/graphsage_movielens_100k.json"
        )

    def test_committed_multiseed_file_matches_per_seed_math(self) -> None:
        path = _REPO_ROOT / "results" / "multiseed_gnn_vs_mf_movielens_100k.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["comparison"], "multiseed_gnn_vs_mf")
        self.assertEqual(payload["protocol_version"], "adr-007-v1")
        self.assertEqual(payload["seeds"], [7, 11, 13, 17, 19])
        self.assertEqual(payload["n_evaluated_users"], 943)
        mf_recalls = [
            row["implicit_mf"]["recall_at_k"] for row in payload["per_seed"]
        ]
        gnn_recalls = [
            row["graphsage"]["recall_at_k"] for row in payload["per_seed"]
        ]
        self.assertAlmostEqual(
            payload["models"]["implicit_mf"]["recall_at_k"]["mean"],
            statistics.fmean(mf_recalls),
        )
        self.assertAlmostEqual(
            payload["models"]["implicit_mf"]["recall_at_k"]["std"],
            statistics.stdev(mf_recalls),
        )
        self.assertAlmostEqual(
            payload["models"]["graphsage"]["recall_at_k"]["mean"],
            statistics.fmean(gnn_recalls),
        )
        self.assertAlmostEqual(
            payload["models"]["graphsage"]["recall_at_k"]["std"],
            statistics.stdev(gnn_recalls),
        )
        hist_mf = compare.load_ranking_result(
            _REPO_ROOT / "results" / "mf_movielens_100k.json"
        )
        hist_gnn = compare.load_ranking_result(
            _REPO_ROOT / "results" / "graphsage_movielens_100k.json"
        )
        self.assertAlmostEqual(
            payload["per_seed"][0]["implicit_mf"]["recall_at_k"],
            hist_mf["metrics"]["recall_at_k"],
        )
        self.assertNotAlmostEqual(
            payload["per_seed"][0]["graphsage"]["recall_at_k"],
            hist_gnn["metrics"]["recall_at_k"],
        )
        markdown = (
            _REPO_ROOT / "results" / "multiseed_gnn_vs_mf_movielens_100k.md"
        ).read_text(encoding="utf-8")
        svg = (
            _REPO_ROOT / "results" / "multiseed_gnn_vs_mf_movielens_100k.svg"
        ).read_text(encoding="utf-8")
        mean_s = f"{payload['models']['implicit_mf']['recall_at_k']['mean']:.6f}"
        self.assertIn(mean_s, markdown)
        self.assertIn("sample std", svg.lower())
        self.assertIn("SAGEConv", payload["protocol_note"])
        self.assertIn("NeighborLoader", payload["protocol_note"])

    def test_phase5_comparison_writer_still_labels_single_seed(self) -> None:
        mf = _fixture_result(model="implicit_mf", recall=0.1, ndcg=0.05, n_epochs=1)
        gnn = _fixture_result(
            model="graphsage", recall=0.2, ndcg=0.08, n_epochs=2, optimizer="adam"
        )
        payload = compare.build_comparison(
            mf, gnn, mf_path="results/mf.json", gnn_path="results/gs.json"
        )
        self.assertEqual(payload["comparison"], "gnn_vs_mf")
        self.assertIn("single-seed", payload["protocol_note"].lower())
        self.assertNotEqual(payload["comparison"], "multiseed_gnn_vs_mf")


class RunnerWiringTests(unittest.TestCase):
    def test_script_uses_native_minibatch_not_neighborloader(self) -> None:
        script = (_SCRIPTS_DIR / "run_multiseed_gnn_vs_mf.py").read_text(
            encoding="utf-8"
        )
        source = Path(multiseed.__file__).read_text(encoding="utf-8")
        self.assertIn("evaluate_mf_seed", script)
        self.assertIn("evaluate_graphsage_seed", script)
        self.assertIn("write_multiseed_artifacts", script)
        self.assertIn("historical_single_seed_notes", script)
        self.assertIn("NativeMinibatchSampler", source)
        self.assertNotIn("NeighborLoader(", script)
        self.assertNotIn("ClusterLoader(", script)
        self.assertNotIn("torch_geometric.loader", script)
        self.assertNotIn("NeighborLoader(", source)
        self.assertNotIn("ClusterLoader(", source)
        self.assertIn("multiseed_gnn_vs_mf_movielens_100k.json", script)
        self.assertNotIn(
            'default=_REPO_ROOT / "results" / "mf_movielens_100k.json"',
            script,
        )
        self.assertNotIn(
            'default=_REPO_ROOT / "results" / "graphsage_movielens_100k.json"',
            script,
        )
        self.assertNotIn(
            'default=_REPO_ROOT / "results" / "gnn_vs_mf_movielens_100k.json"',
            script,
        )

    def test_mf_100k_config_matches_phase5_and_includes_seed_7(self) -> None:
        config = baseline.movielens_100k_config()
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.n_epochs, 1)
        self.assertEqual(config.n_factors, 16)
        self.assertEqual(config.n_negatives, 2)


class SyntheticTwoSeedSmokeTests(unittest.TestCase):
    def test_mf_and_graphsage_seed_pair_aggregates(self) -> None:
        split = _synthetic_split()
        manifest = _synthetic_manifest(split)
        provenance = {
            "git_commit": "test",
            "generated_at": "2026-09-15T00:00:00+00:00",
            "python": "3.12.3",
            "numpy": "2.0.0",
            "torch": "2.6.0",
            "platform": {"system": "Linux"},
        }
        pairs = []
        for seed in (7, 11):
            mf = multiseed.evaluate_mf_seed(
                split,
                manifest,
                seed=seed,
                k=10,
                target_split="test",
                provenance=provenance,
                config=_tiny_mf_config(seed),
            )
            gnn = multiseed.evaluate_graphsage_seed(
                split,
                manifest,
                seed=seed,
                k=10,
                target_split="test",
                provenance=provenance,
                config=_tiny_graphsage_config(seed),
            )
            self.assertEqual(mf["seed"], seed)
            self.assertEqual(gnn["seed"], seed)
            self.assertIsInstance(gnn["metrics"]["recall_at_k"], float)
            self.assertGreaterEqual(gnn["metrics"]["recall_at_k"], 0.0)
            self.assertLessEqual(gnn["metrics"]["recall_at_k"], 1.0)
            pairs.append((mf, gnn))
        payload = multiseed.build_multiseed_comparison(pairs, seeds=(7, 11))
        self.assertEqual(payload["seeds"], [7, 11])
        self.assertEqual(payload["n_seeds"], 2)
        mean = payload["models"]["graphsage"]["recall_at_k"]["mean"]
        expected = statistics.fmean(
            [
                pairs[0][1]["metrics"]["recall_at_k"],
                pairs[1][1]["metrics"]["recall_at_k"],
            ]
        )
        self.assertAlmostEqual(mean, expected)
        self.assertIn("SAGEConv", str(payload["models"]["graphsage"].get("conv_stack")))


if __name__ == "__main__":
    unittest.main()
