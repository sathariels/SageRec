"""Phase 4 GraphSAGE training smoke on the native mini-batch harness.

Tiny synthetic ADR-003 data only. Metrics here are protocol verification,
not MovieLens 100K quality results. Neighborhood expansion must call
compiled graph_sampler via sagerec_minibatch; these tests fail if the
extension is missing or bypassed.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import graph_sampler
import sagerec_graphsage as graphsage
import sagerec_metrics as metrics
import sagerec_minibatch as minibatch
import sagerec_prep as prep
import sagerec_reference_sampler as reference
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


class GraphSAGEModuleContractTests(unittest.TestCase):
    def test_module_uses_native_minibatch_not_pyg_loader(self) -> None:
        source = Path(graphsage.__file__).read_text(encoding="utf-8")
        self.assertIn("sagerec_minibatch", source)
        self.assertIn("NativeMinibatchSampler", source)
        self.assertTrue(callable(minibatch.call_native_sample_neighbors))
        self.assertNotIn("import torch_geometric", source)
        self.assertNotIn("from torch_geometric", source)
        self.assertNotIn("torch_geometric.loader", source)
        self.assertNotIn("import sagerec_reference_sampler", source)
        self.assertNotIn("from sagerec_reference_sampler", source)

    def test_invalid_config_fails(self) -> None:
        with self.assertRaises(ValueError):
            graphsage.GraphSAGEConfig(embedding_dim=0)
        with self.assertRaises(ValueError):
            graphsage.GraphSAGEConfig(n_layers=2, fanouts=(4,))
        with self.assertRaises(ValueError):
            graphsage.GraphSAGEConfig(learning_rate=0.0)
        with self.assertRaises(ValueError):
            graphsage.fit_graphsage([], num_users=1, num_items=1)


class GraphSAGETrainingTests(unittest.TestCase):
    def test_training_calls_native_sample_neighbors_not_reference(self) -> None:
        split = _synthetic_split()
        config = _tiny_config(seed=5)
        with mock.patch.object(
            minibatch,
            "call_native_sample_neighbors",
            wraps=minibatch.call_native_sample_neighbors,
        ) as native:
            with mock.patch.object(
                reference, "sample_neighbors", wraps=reference.sample_neighbors
            ) as ref:
                model = graphsage.train_graphsage(split, config)
        ref.assert_not_called()
        self.assertGreaterEqual(native.call_count, 1)
        for call in native.call_args_list:
            self.assertIs(call.args[0], model.sampler.graph)
            self.assertIsInstance(call.args[0], graph_sampler.BipartiteCSR)
            node_id, k, seed = call.args[1:]
            self.assertGreaterEqual(node_id, 0)
            self.assertLess(node_id, model.sampler.num_nodes)
            self.assertIn(k, config.fanouts)
            self.assertGreaterEqual(seed, 0)
        self.assertIsInstance(model, PairScorer)
        self.assertIsInstance(model.sampler, minibatch.NativeMinibatchSampler)

    def test_score_pairs_calls_native_sampler(self) -> None:
        split = _synthetic_split()
        model = graphsage.train_graphsage(split, _tiny_config(seed=3))
        with mock.patch.object(
            minibatch,
            "call_native_sample_neighbors",
            wraps=minibatch.call_native_sample_neighbors,
        ) as native:
            with mock.patch.object(
                reference, "sample_neighbors", wraps=reference.sample_neighbors
            ) as ref:
                scores = model.score_pairs([0, 1], [3, 8])
        ref.assert_not_called()
        self.assertGreaterEqual(native.call_count, 1)
        self.assertEqual(len(scores), 2)
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_held_out_positives_never_enter_train_graph_or_negatives(self) -> None:
        split = _synthetic_split()
        train_pairs = split.train_positive_pairs()
        held_out = split.positive_pair_set("validation") | split.positive_pair_set("test")
        self.assertTrue(held_out)
        self.assertTrue(set(train_pairs).isdisjoint(held_out))

        captured: dict[str, list[tuple[int, int]]] = {}
        original = graphsage.fit_graphsage

        def wrapped(train_pairs_arg, **kwargs):  # type: ignore[no-untyped-def]
            captured["pairs"] = list(train_pairs_arg)
            return original(train_pairs_arg, **kwargs)

        graphsage.fit_graphsage = wrapped  # type: ignore[method-assign]
        try:
            model = graphsage.train_graphsage(split, _tiny_config())
        finally:
            graphsage.fit_graphsage = original  # type: ignore[method-assign]

        self.assertEqual(set(captured["pairs"]), set(train_pairs))
        self.assertTrue(set(captured["pairs"]).isdisjoint(held_out))

        offsets = model.sampler.offsets()
        neighbors = model.sampler.neighbors()

        def neighborhood(node_id: int) -> list[int]:
            return neighbors[offsets[node_id] : offsets[node_id + 1]]

        for user_id, movie_id in held_out:
            global_movie = split.num_users + movie_id
            self.assertNotIn(global_movie, neighborhood(user_id))
            self.assertNotIn(user_id, neighborhood(global_movie))

    def test_seeded_smoke_metrics_are_finite_bounded_and_reproducible(self) -> None:
        split = _synthetic_split()
        config = _tiny_config(seed=11)

        first_model = graphsage.train_graphsage(split, config)
        first = metrics.evaluate_ranking(first_model, split, target_split="test", k=10)
        second_model = graphsage.train_graphsage(split, config)
        second = metrics.evaluate_ranking(second_model, split, target_split="test", k=10)

        for report in (first, second):
            self.assertEqual(report.k, 10)
            self.assertEqual(report.split, "test")
            self.assertEqual(report.evaluated_user_ids, split.eligible_user_ids)
            self.assertNotIn(2, report.evaluated_user_ids)
            self.assertTrue(math.isfinite(report.recall_at_k))
            self.assertTrue(math.isfinite(report.ndcg_at_k))
            self.assertGreaterEqual(report.recall_at_k, 0.0)
            self.assertLessEqual(report.recall_at_k, 1.0)
            self.assertGreaterEqual(report.ndcg_at_k, 0.0)
            self.assertLessEqual(report.ndcg_at_k, 1.0)

        self.assertEqual(first.recall_at_k, second.recall_at_k)
        self.assertEqual(first.ndcg_at_k, second.ndcg_at_k)
        self.assertEqual(first.per_user_recall, second.per_user_recall)
        self.assertEqual(first.per_user_ndcg, second.per_user_ndcg)
        np.testing.assert_allclose(
            first_model.embedding.weight.detach().cpu().numpy(),
            second_model.embedding.weight.detach().cpu().numpy(),
        )

    def test_rejects_non_native_sampler(self) -> None:
        class FakeSampler:
            num_nodes = 2
            num_users = 1
            num_movies = 1

        with self.assertRaises(TypeError) as ctx:
            graphsage.GraphSAGERecommender(FakeSampler(), _tiny_config())  # type: ignore[arg-type]
        self.assertIn("NativeMinibatchSampler", str(ctx.exception))
        self.assertIn("PyG", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
