"""Phase 5 comparison-writer and GraphSAGE 100K wiring tests.

Comparison tests use tiny fixture JSON only (no invented 100K quality
numbers). GraphSAGE wiring tests stay on the tiny synthetic ADR-003
split and spy on native ``sample_neighbors``.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_PYTHON_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_compare as compare
import sagerec_dataset as dataset
import sagerec_graphsage as graphsage
import sagerec_metrics as metrics
import sagerec_minibatch as minibatch
import sagerec_prep as prep
from sagerec_scoring import PairScorer


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


def _tiny_config(seed: int = 7) -> graphsage.GraphSAGEConfig:
    return graphsage.GraphSAGEConfig(
        embedding_dim=4,
        hidden_dim=4,
        n_layers=2,
        fanouts=(3, 2),
        n_epochs=2,
        batch_size=4,
        learning_rate=0.1,
        n_negatives=2,
        l2=1e-4,
        seed=seed,
    )


def _fixture_result(
    *,
    model: str,
    recall: float,
    ndcg: float,
    n_epochs: int,
    n_users: int = 4,
    k: int = 10,
    split: str = "test",
    seed: int = 7,
) -> dict:
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
        "hyperparams": {"n_epochs": n_epochs, "seed": seed},
        "metrics": {
            "k": k,
            "split": split,
            "recall_at_k": recall,
            "ndcg_at_k": ndcg,
            "n_evaluated_users": n_users,
        },
        "eligibility": {
            "users": n_users,
            "movies": 10,
            "interactions": {"total": 20, "train": 12, "validation": 4, "test": 4},
            "eligible_users": n_users,
            "cold_start_users": 0,
            "evaluated_users": n_users,
        },
        "note": "fixture",
        "sampler": "graph_sampler via sagerec_minibatch" if model == "graphsage" else None,
    }


class ComparisonWriterTests(unittest.TestCase):
    def test_build_comparison_copies_stored_metrics(self) -> None:
        mf = _fixture_result(model="implicit_mf", recall=0.038176, ndcg=0.020303, n_epochs=1)
        gnn = _fixture_result(model="graphsage", recall=0.041123, ndcg=0.022456, n_epochs=2)
        payload = compare.build_comparison(
            mf, gnn, mf_path="results/mf.json", gnn_path="results/gs.json"
        )
        self.assertEqual(payload["comparison"], "gnn_vs_mf")
        self.assertEqual(payload["dataset_edition"], "100k")
        self.assertEqual(payload["split_policy_id"], "per_user_chronological_leave_one_out")
        self.assertEqual(payload["k"], 10)
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["n_evaluated_users"], 4)
        self.assertEqual(payload["models"]["implicit_mf"]["recall_at_k"], 0.038176)
        self.assertEqual(payload["models"]["graphsage"]["ndcg_at_k"], 0.022456)
        self.assertEqual(payload["models"]["implicit_mf"]["result_path"], "results/mf.json")
        self.assertEqual(payload["models"]["graphsage"]["result_path"], "results/gs.json")
        self.assertIn("single-seed", payload["protocol_note"].lower())
        self.assertIn("multi-seed leaderboard", payload["note"].lower())

    def test_write_artifacts_round_trip_and_chart(self) -> None:
        mf = _fixture_result(model="implicit_mf", recall=0.1, ndcg=0.05, n_epochs=1)
        gnn = _fixture_result(model="graphsage", recall=0.2, ndcg=0.08, n_epochs=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mf_path = root / "mf.json"
            gnn_path = root / "gs.json"
            mf_path.write_text(json.dumps(mf), encoding="utf-8")
            gnn_path.write_text(json.dumps(gnn), encoding="utf-8")
            artifacts = compare.write_comparison_artifacts(
                mf_path,
                gnn_path,
                json_path=root / "cmp.json",
                markdown_path=root / "cmp.md",
                chart_path=root / "cmp.svg",
            )
            loaded = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["models"]["implicit_mf"]["recall_at_k"], 0.1)
            self.assertEqual(loaded["models"]["graphsage"]["recall_at_k"], 0.2)
            markdown = artifacts.markdown_path.read_text(encoding="utf-8")
            self.assertIn("0.100000", markdown)
            self.assertIn("0.200000", markdown)
            self.assertIn("GraphSAGE", markdown)
            self.assertIn("implicit MF", markdown)
            svg = artifacts.chart_path.read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertIn("0.1000", svg)
            self.assertIn("0.2000", svg)
            self.assertIn("implicit MF", svg)
            self.assertIn("GraphSAGE", svg)

    def test_rejects_protocol_mismatch(self) -> None:
        mf = _fixture_result(model="implicit_mf", recall=0.1, ndcg=0.05, n_epochs=1)
        gnn = _fixture_result(
            model="graphsage", recall=0.2, ndcg=0.08, n_epochs=2, n_users=5
        )
        with self.assertRaises(compare.ComparisonError) as ctx:
            compare.assert_comparable(mf, gnn, mf_path="mf", gnn_path="gs")
        self.assertIn("n_evaluated_users", str(ctx.exception))

        gnn_ok_users = _fixture_result(
            model="graphsage", recall=0.2, ndcg=0.08, n_epochs=2, k=5
        )
        with self.assertRaises(compare.ComparisonError):
            compare.assert_comparable(mf, gnn_ok_users, mf_path="mf", gnn_path="gs")

        gnn_wrong_model = _fixture_result(
            model="implicit_mf", recall=0.2, ndcg=0.08, n_epochs=2
        )
        with self.assertRaises(compare.ComparisonError) as model_ctx:
            compare.assert_comparable(mf, gnn_wrong_model, mf_path="mf", gnn_path="gs")
        self.assertIn("graphsage", str(model_ctx.exception))

    def test_rejects_missing_or_invented_out_of_range_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            with self.assertRaises(compare.ComparisonError) as ctx:
                compare.load_ranking_result(missing)
            self.assertIn("not found", str(ctx.exception))
            bad = _fixture_result(model="implicit_mf", recall=1.5, ndcg=0.1, n_epochs=1)
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(compare.ComparisonError) as range_ctx:
                compare.load_ranking_result(path)
            self.assertIn("[0, 1]", str(range_ctx.exception))

    def test_stored_result_ref_is_repo_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "results" / "mf.json"
            path.parent.mkdir()
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(
                compare.stored_result_ref(path, repo_root=root),
                "results/mf.json",
            )

    def test_existing_mf_result_matches_comparison_schema(self) -> None:
        mf_path = Path(__file__).resolve().parents[2] / "results" / "mf_movielens_100k.json"
        payload = compare.load_ranking_result(mf_path)
        self.assertEqual(payload["model"], "implicit_mf")
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["metrics"]["k"], 10)
        self.assertEqual(payload["metrics"]["n_evaluated_users"], 943)

    def test_committed_100k_results_are_comparable(self) -> None:
        root = Path(__file__).resolve().parents[2]
        mf_path = root / "results" / "mf_movielens_100k.json"
        gnn_path = root / "results" / "graphsage_movielens_100k.json"
        if not gnn_path.is_file():
            self.skipTest("GraphSAGE 100K result file is not present")
        mf = compare.load_ranking_result(mf_path)
        gnn = compare.load_ranking_result(gnn_path)
        compare.assert_comparable(mf, gnn, mf_path=mf_path, gnn_path=gnn_path)
        payload = compare.build_comparison(
            mf, gnn, mf_path=mf_path, gnn_path=gnn_path, repo_root=root
        )
        self.assertEqual(payload["models"]["implicit_mf"]["result_path"], "results/mf_movielens_100k.json")
        self.assertEqual(
            payload["models"]["graphsage"]["result_path"],
            "results/graphsage_movielens_100k.json",
        )
        self.assertEqual(
            payload["models"]["implicit_mf"]["recall_at_k"],
            mf["metrics"]["recall_at_k"],
        )
        self.assertEqual(
            payload["models"]["graphsage"]["recall_at_k"],
            gnn["metrics"]["recall_at_k"],
        )


class GraphSAGE100kWiringTests(unittest.TestCase):
    def test_movielens_100k_config_is_seed_7_and_modest(self) -> None:
        config = graphsage.movielens_100k_config()
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.n_layers, 2)
        self.assertEqual(config.fanouts, (8, 8))
        self.assertEqual(config.embedding_dim, 16)
        self.assertGreaterEqual(config.n_epochs, 1)
        self.assertLessEqual(config.n_epochs, 4)
        self.assertGreaterEqual(config.batch_size, 32)
        notes = graphsage.graphsage_training_seed_notes(config.seed)
        self.assertEqual(notes["experiment_seed"], 7)
        self.assertIn("derived_sample_seed", str(notes["minibatch_neighborhood_seed"]))

    def test_100k_script_requires_native_minibatch_not_pyg(self) -> None:
        script = (_SCRIPTS_DIR / "run_graphsage_movielens_100k.py").read_text(
            encoding="utf-8"
        )
        source = Path(graphsage.__file__).read_text(encoding="utf-8")
        self.assertIn("movielens_100k_config", script)
        self.assertIn("train_graphsage", script)
        self.assertIn("NativeMinibatchSampler", script)
        self.assertIn("evaluate_ranking", script)
        self.assertIn("graphsage_result_payload", script)
        self.assertNotIn("import torch_geometric", script)
        self.assertNotIn("from torch_geometric", script)
        self.assertNotIn("NeighborLoader(", script)
        self.assertNotIn("SAGEConv(", script)
        self.assertIn("NativeMinibatchSampler", source)
        self.assertNotIn("import torch_geometric", source)
        self.assertNotIn("from torch_geometric", source)

    def test_result_payload_schema_mirrors_mf_and_labels_single_seed(self) -> None:
        split = _synthetic_split()
        config = _tiny_config(seed=7)
        model = graphsage.train_graphsage(split, config)
        report = metrics.evaluate_ranking(model, split, target_split="test", k=10)
        manifest = {
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
        payload = graphsage.graphsage_result_payload(
            manifest=manifest,
            config=config,
            report=report,
            git_commit="abc123",
            generated_at="2026-09-12T00:00:00+00:00",
            python_version="3.12.3",
            numpy_version="2.0.0",
            torch_version="2.6.0",
            platform_info={"system": "Linux"},
        )
        self.assertEqual(payload["model"], "graphsage")
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["metrics"]["k"], 10)
        self.assertEqual(payload["metrics"]["n_evaluated_users"], report.n_users)
        self.assertEqual(payload["eligibility"]["evaluated_users"], report.n_users)
        self.assertIn("NativeMinibatchSampler", payload["sampler"])
        self.assertIn("single-seed", payload["note"].lower())
        self.assertIn("not a multi-seed leaderboard", payload["note"].lower())
        self.assertNotIn("torch_geometric", payload["sampler"])
        self.assertIn("no PyG", payload["sampler"])
        for key in compare.REQUIRED_RESULT_KEYS:
            self.assertIn(key, payload)

    def test_eval_materialize_and_score_pairs_use_native_sampler(self) -> None:
        split = _synthetic_split()
        config = _tiny_config(seed=3)
        model = graphsage.train_graphsage(split, config)
        self.assertIsInstance(model, PairScorer)
        self.assertIsInstance(model.sampler, minibatch.NativeMinibatchSampler)
        model.clear_eval_embeddings()
        with mock.patch.object(
            minibatch,
            "call_native_sample_neighbors",
            wraps=minibatch.call_native_sample_neighbors,
        ) as native:
            model.materialize_eval_embeddings()
            scores = model.score_pairs([0, 1], [3, 8])
        self.assertGreaterEqual(native.call_count, 1)
        for call in native.call_args_list:
            self.assertIs(call.args[0], model.sampler.graph)
            node_id, k, seed = call.args[1:]
            self.assertGreaterEqual(node_id, 0)
            self.assertLess(node_id, model.sampler.num_nodes)
            self.assertIn(k, config.fanouts)
            self.assertGreaterEqual(seed, 0)
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(math.isfinite(float(value)) for value in scores))

    def test_ensure_processed_reuses_artifacts_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "u.data"
            raw.write_text(
                "9\t8\t5\t10\n9\t3\t4\t20\n9\t4\t3\t30\n5\t3\t1\t40\n",
                encoding="utf-8",
            )
            processed = Path(tmp) / "processed"
            dataset.prepare_movielens_100k(raw, processed)
            with mock.patch.object(dataset.download, "download_movielens_100k") as dl:
                split, manifest = dataset.ensure_movielens_100k_processed(
                    processed,
                    Path(tmp) / "raw",
                    download_if_missing=True,
                )
            dl.assert_not_called()
            self.assertEqual(manifest["dataset_edition"], "100k")
            self.assertTrue(split.eligible_user_ids)
            missing = Path(tmp) / "missing-processed"
            with self.assertRaises(dataset.MovieLensPrepError) as ctx:
                dataset.ensure_movielens_100k_processed(
                    missing,
                    Path(tmp) / "raw-missing",
                    download_if_missing=False,
                )
            self.assertIn("download_if_missing", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
