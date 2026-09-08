"""Phase 3 MF baseline and shared ranking-metric tests.

Tiny in-memory fixtures only. Metrics here verify the protocol; they are not
MovieLens 100K quality results.

Requires PYTHONPATH to include the CMake build directory (graph_sampler) and
``python/``.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import graph_sampler
import sagerec_baseline as baseline
import sagerec_metrics as metrics
import sagerec_negatives as negatives
import sagerec_prep as prep
from sagerec_scoring import PairScorer


def _row(
    user_id: int, movie_id: int, timestamp: int, rating: int = 1
) -> prep.NormalizedInteraction:
    return prep.NormalizedInteraction(user_id, movie_id, timestamp, rating)


class DictScorer:
    """Deterministic PairScorer for metric tests."""

    def __init__(
        self,
        scores: dict[tuple[int, int], float],
        default: float = 0.0,
    ) -> None:
        self.scores = scores
        self.default = default

    def score_pairs(
        self, user_ids: list[int] | tuple[int, ...], item_ids: list[int] | tuple[int, ...]
    ) -> list[float]:
        return [
            self.scores.get((int(user_id), int(item_id)), self.default)
            for user_id, item_id in zip(user_ids, item_ids)
        ]


class RankingMetricTests(unittest.TestCase):
    def test_recall_and_ndcg_known_ranks(self) -> None:
        ranked = [4, 7, 1, 0, 2, 3, 5, 6, 8, 9, 10]
        relevant = {7}
        self.assertEqual(metrics.recall_at_k(ranked, relevant, k=10), 1.0)
        self.assertAlmostEqual(
            metrics.ndcg_at_k(ranked, relevant, k=10),
            1.0 / math.log2(3),
        )
        self.assertEqual(metrics.recall_at_k(ranked, {10}, k=10), 0.0)
        self.assertEqual(metrics.ndcg_at_k(ranked, {10}, k=10), 0.0)

    def test_rank_items_breaks_ties_on_item_id(self) -> None:
        scorer = DictScorer({(0, 2): 1.0, (0, 5): 1.0, (0, 1): 0.5})
        ranked = metrics.rank_items(scorer, 0, [5, 1, 2])
        self.assertEqual(ranked, [2, 5, 1])

    def test_evaluate_filters_train_keeps_target_and_skips_cold_start(self) -> None:
        interactions = [
            _row(0, 0, 10),
            _row(0, 1, 20),
            _row(0, 2, 30),
            _row(0, 3, 40),
            _row(1, 0, 1),
            _row(1, 1, 2),
            _row(1, 2, 3),
            _row(2, 4, 5),
        ]
        split = prep.split_interactions(interactions, num_users=3, num_movies=12)
        self.assertEqual(split.eligible_user_ids, (0, 1))
        self.assertEqual(split.cold_start_user_ids, (2,))
        # User 0: train {0,1}, val 2, test 3.
        # Huge train score must not occupy a ranking slot after filtering.
        scorer = DictScorer(
            {
                (0, 0): 100.0,
                (0, 1): 100.0,
                (0, 2): 50.0,
                (0, 3): 9.0,
                (1, 2): 8.0,
            },
            default=0.0,
        )
        report = metrics.evaluate_ranking(scorer, split, target_split="test", k=10)
        self.assertEqual(report.evaluated_user_ids, (0, 1))
        self.assertNotIn(2, report.evaluated_user_ids)
        self.assertEqual(report.n_users, 2)
        self.assertTrue(0.0 <= report.recall_at_k <= 1.0)
        self.assertTrue(0.0 <= report.ndcg_at_k <= 1.0)
        self.assertEqual(report.per_user_recall[0], 1.0)

        ranked = metrics.rank_items(
            scorer,
            0,
            metrics.filtered_candidates(split.num_movies, exclude={0, 1, 2}),
        )
        self.assertNotIn(0, ranked)
        self.assertNotIn(1, ranked)
        self.assertNotIn(2, ranked)
        self.assertEqual(ranked[0], 3)

    def test_macro_average_two_users(self) -> None:
        interactions = [
            _row(0, 0, 1),
            _row(0, 1, 2),
            _row(0, 2, 3),
            _row(1, 0, 1),
            _row(1, 1, 2),
            _row(1, 2, 3),
        ]
        split = prep.split_interactions(interactions, num_users=2, num_movies=15)
        # User 0: test item 2 ranks first. User 1: test item 2 ranks 11th.
        scores: dict[tuple[int, int], float] = {}
        for item in range(15):
            scores[(0, item)] = 0.0
            scores[(1, item)] = 0.0
        scores[(0, 2)] = 10.0
        for item in range(15):
            if item not in {0, 1, 2}:
                scores[(1, item)] = float(20 - item)
        scores[(1, 2)] = -1.0
        report = metrics.evaluate_ranking(
            DictScorer(scores), split, target_split="test", k=10
        )
        self.assertEqual(report.per_user_recall, (1.0, 0.0))
        self.assertAlmostEqual(report.recall_at_k, 0.5)
        self.assertGreater(report.ndcg_at_k, 0.0)
        self.assertLess(report.ndcg_at_k, 1.0)

    def test_rejects_no_eligible_users(self) -> None:
        split = prep.split_interactions([_row(0, 0, 1), _row(0, 1, 2)], num_movies=3)
        with self.assertRaises(ValueError) as ctx:
            metrics.evaluate_ranking(DictScorer({}), split)
        self.assertIn("eligible", str(ctx.exception))


class NegativeSamplingTests(unittest.TestCase):
    def test_negatives_never_overlap_train_positives(self) -> None:
        train = {(0, 0), (0, 1), (0, 4), (1, 2)}
        by_user = negatives.items_by_user(train)
        rng = np.random.default_rng(42)
        drawn: list[tuple[int, int]] = []
        for _ in range(200):
            item = negatives.sample_negative_item(
                0, num_items=8, forbidden_items=by_user[0], rng=rng
            )
            drawn.append((0, item))
        negatives.assert_no_positive_overlap(drawn, train)
        self.assertTrue(all(item not in {0, 1, 4} for _, item in drawn))

    def test_full_catalog_forbidden_fails(self) -> None:
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError) as ctx:
            negatives.sample_negative_item(
                0, num_items=3, forbidden_items={0, 1, 2}, rng=rng
            )
        self.assertIn("no eligible negative", str(ctx.exception))


class ImplicitMfTests(unittest.TestCase):
    def _synthetic_split(self) -> prep.SplitResult:
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

    def _tiny_config(self, seed: int = 7) -> baseline.MFConfig:
        return baseline.MFConfig(
            n_factors=4,
            n_epochs=3,
            learning_rate=0.1,
            n_negatives=2,
            l2=0.01,
            seed=seed,
        )

    def test_held_out_positives_never_enter_mf_train_or_csr(self) -> None:
        split = self._synthetic_split()
        train_pairs = split.train_positive_pairs()
        held_out = split.positive_pair_set("validation") | split.positive_pair_set("test")
        self.assertTrue(held_out)
        self.assertTrue(set(train_pairs).isdisjoint(held_out))

        captured: dict[str, list[tuple[int, int]]] = {}
        original = baseline.fit_implicit_mf

        def wrapped(train_pairs_arg, **kwargs):  # type: ignore[no-untyped-def]
            captured["pairs"] = list(train_pairs_arg)
            return original(train_pairs_arg, **kwargs)

        baseline.fit_implicit_mf = wrapped  # type: ignore[method-assign]
        try:
            model = baseline.train_implicit_mf(split, self._tiny_config())
        finally:
            baseline.fit_implicit_mf = original  # type: ignore[method-assign]

        self.assertEqual(set(captured["pairs"]), set(train_pairs))
        self.assertTrue(set(captured["pairs"]).isdisjoint(held_out))
        self.assertIsInstance(model, PairScorer)

        graph = graph_sampler.BipartiteCSR(
            split.num_users, split.num_movies, train_pairs
        )
        offsets = graph.offsets()
        neighbors = graph.neighbors()

        def neighborhood(node_id: int) -> list[int]:
            return neighbors[offsets[node_id] : offsets[node_id + 1]]

        for user_id, movie_id in held_out:
            global_movie = split.num_users + movie_id
            self.assertNotIn(global_movie, neighborhood(user_id))

    def test_seeded_smoke_metrics_are_finite_bounded_and_reproducible(self) -> None:
        split = self._synthetic_split()
        config = self._tiny_config(seed=11)

        first_model = baseline.train_implicit_mf(split, config)
        first = metrics.evaluate_ranking(first_model, split, target_split="test", k=10)
        second_model = baseline.train_implicit_mf(split, config)
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
        np.testing.assert_allclose(
            first_model.user_factors, second_model.user_factors
        )
        np.testing.assert_allclose(
            first_model.item_factors, second_model.item_factors
        )

    def test_invalid_config_fails(self) -> None:
        with self.assertRaises(ValueError):
            baseline.MFConfig(n_factors=0)
        with self.assertRaises(ValueError):
            baseline.MFConfig(learning_rate=0.0)
        with self.assertRaises(ValueError):
            baseline.fit_implicit_mf([], num_users=1, num_items=1)


if __name__ == "__main__":
    unittest.main()
