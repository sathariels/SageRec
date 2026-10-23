"""Phase 10 ADR-010 feature, re-ranker, and offline-eval tests.

Tiny in-memory fixtures only. Metrics here verify the protocol; they are not
MovieLens 100K quality results. The committed ``results/rerank_eval.json``
test checks the measured file's schema and does not retrain 100K.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_PYTHON_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PYTHON_DIR.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_metrics as metrics
import sagerec_prep as prep
import sagerec_rerank as rerank

_RERANK_JSON = _REPO_ROOT / "results" / "rerank_eval.json"
_RERANK_MD = _REPO_ROOT / "results" / "rerank_eval.md"


def _row(
    user_id: int,
    movie_id: int,
    timestamp: int,
    rating: int = 1,
) -> prep.NormalizedInteraction:
    return prep.NormalizedInteraction(user_id, movie_id, timestamp, rating)


def _uitem_line(source_id: int, flags: list[int]) -> str:
    prefix = f"{source_id}|Movie {source_id}|01-Jan-1995||http://example"
    return prefix + "|" + "|".join(str(flag) for flag in flags)


class MapMetricTests(unittest.TestCase):
    def test_average_precision_at_10_for_one_relevant_item(self) -> None:
        relevant = {4}
        self.assertEqual(metrics.average_precision_at_k([4, 1, 2], relevant, 10), 1.0)
        self.assertEqual(metrics.average_precision_at_k([1, 4, 2], relevant, 10), 0.5)
        self.assertEqual(metrics.average_precision_at_k([1, 2, 3], relevant, 10), 0.0)
        self.assertEqual(metrics.average_precision_at_k([1, 2, 4], relevant, 2), 0.0)
        with self.assertRaises(ValueError):
            metrics.average_precision_at_k([1], set(), 10)


class FeatureTests(unittest.TestCase):
    def _split(self) -> prep.SplitResult:
        interactions = [
            _row(0, 0, 10),
            _row(0, 1, 30),
            _row(0, 2, 40),
            _row(0, 3, 1000),
            _row(1, 0, 10),
            _row(1, 4, 20),
            _row(1, 2, 30),
        ]
        return prep.split_interactions(interactions, num_users=2, num_movies=5)

    def test_train_only_popularity_recency_and_genre_affinity(self) -> None:
        split = self._split()
        genres = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
        tables = rerank.build_feature_tables(split, genres)
        self.assertTrue(tables.genres_available)
        np.testing.assert_allclose(tables.item_popularity, np.log1p([2.0, 1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(tables.item_recency, [0.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(tables.user_genre_profile[0], [0.5, 0.5])
        features = rerank.candidate_features(
            tables,
            0,
            [0, 1, 3],
            [0.2, 0.4, 0.6],
        )
        self.assertEqual(features.shape, (3, 4))
        np.testing.assert_allclose(features[:, 0], [0.2, 0.4, 0.6])
        np.testing.assert_allclose(features[:, 1], tables.item_popularity[[0, 1, 3]])
        np.testing.assert_allclose(features[:, 2], [0.5, 0.5, 1.0])
        np.testing.assert_allclose(features[:, 3], tables.item_recency[[0, 1, 3]])
        self.assertEqual(list(rerank.FEATURE_NAMES), [
            "retrieval_score",
            "item_popularity",
            "genre_affinity",
            "item_recency",
        ])

    def test_held_out_rows_do_not_change_feature_tables(self) -> None:
        full = self._split()
        genres = np.ones((full.num_movies, 1), dtype=np.float64)
        tables = rerank.build_feature_tables(full, genres)
        train_counts = np.zeros(full.num_movies, dtype=np.float64)
        for row in full.interactions_for("train"):
            train_counts[row.movie_id] += 1.0
        np.testing.assert_allclose(tables.item_popularity, np.log1p(train_counts))
        self.assertEqual(float(tables.item_popularity[3]), 0.0)
        self.assertEqual(float(tables.item_recency[3]), 0.0)
        self.assertGreater(float(train_counts[0]), 0.0)

    def test_training_pool_drops_the_test_item_and_test_pool_drops_validation(self) -> None:
        split = self._split()

        def score_user(_user_id: int) -> np.ndarray:
            return np.arange(split.num_movies, dtype=np.float64)

        train_pools = rerank.build_candidate_pools(
            split,
            score_user,
            label_split="validation",
            extra_exclude_splits=("test",),
            stage1_n=5,
        )
        user0_train = next(pool for pool in train_pools if pool.user_id == 0)
        self.assertNotIn(3, user0_train.item_ids.tolist())
        self.assertIn(2, user0_train.item_ids.tolist())
        self.assertGreaterEqual(user0_train.positive_index, 0)

        test_pools = rerank.build_candidate_pools(
            split,
            score_user,
            label_split="test",
            extra_exclude_splits=("validation",),
            stage1_n=5,
        )
        user0_test = next(pool for pool in test_pools if pool.user_id == 0)
        self.assertNotIn(2, user0_test.item_ids.tolist())
        self.assertNotIn(0, user0_test.item_ids.tolist())
        self.assertIn(3, user0_test.item_ids.tolist())


class GenreParseTests(unittest.TestCase):
    def test_uitem_genres_align_to_local_source_ids(self) -> None:
        flags_a = [1] + [0] * 18
        flags_b = [0, 1] + [0] * 17
        text = "\n".join(
            [
                _uitem_line(5, flags_a),
                _uitem_line(1, flags_b),
            ]
        )
        genres = rerank.parse_uitem_genres(text, 2, [5, 1])
        self.assertEqual(genres.shape, (2, 19))
        self.assertEqual(float(genres[0, 0]), 1.0)
        self.assertEqual(float(genres[1, 1]), 1.0)
        self.assertEqual(len(rerank.GENRE_NAMES), 19)

    def test_uitem_rejects_a_short_row(self) -> None:
        with self.assertRaises(rerank.RerankError) as ctx:
            rerank.parse_uitem_genres("1|Movie|date||url|0\n", 1, [1])
        self.assertIn("fields", str(ctx.exception))


class MlpTests(unittest.TestCase):
    def test_listwise_mlp_prefers_the_popularity_feature_and_is_deterministic(self) -> None:
        positive = np.asarray([0.0, 4.0, 0.0, 0.0], dtype=np.float64)
        negative = np.asarray([4.0, 0.0, 0.0, 0.0], dtype=np.float64)
        matrix = np.vstack([positive, negative, negative, negative])
        matrices = [matrix.copy() for _ in range(24)]
        labels = [0] * len(matrices)
        config = rerank.RerankConfig(
            stage1_n=4,
            hidden_dim=8,
            n_epochs=40,
            learning_rate=0.2,
            l2=0.0,
            seed=7,
            k=2,
            init_std=0.1,
        )
        first, first_loss = rerank.train_listwise_mlp(matrices, labels, config)
        second, second_loss = rerank.train_listwise_mlp(matrices, labels, config)
        np.testing.assert_allclose(first["w_linear"], second["w_linear"])
        np.testing.assert_allclose(first["w1"], second["w1"])
        self.assertLess(first_loss[-1], first_loss[0])
        _pre, _hidden, logits = rerank._forward(matrix, first)
        self.assertGreater(float(logits[0]), float(np.max(logits[1:])))

    def test_stage1_n_below_k_is_rejected(self) -> None:
        with self.assertRaises(rerank.RerankError):
            rerank.RerankConfig(stage1_n=5, k=10)

    def test_artifact_roundtrip_preserves_order(self) -> None:
        tables = rerank.FeatureTables(
            item_popularity=np.asarray([0.0, 3.0, 1.0], dtype=np.float64),
            item_recency=np.zeros(3, dtype=np.float64),
            item_genres=np.zeros((3, 1), dtype=np.float64),
            user_genre_profile=np.zeros((1, 1), dtype=np.float64),
            genres_available=False,
        )
        bundle = rerank.zero_linear_bundle(
            tables,
            stage1_model="fixture",
            popularity_weight=1.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rerank.npz"
            rerank.save_rerank_artifact(path, bundle)
            loaded = rerank.load_rerank_artifact(path)
        ordered = rerank.order_candidates(loaded, 0, [0, 1, 2], [5.0, 0.0, 4.0])
        self.assertEqual([item_id for item_id, _score in ordered], [1, 2, 0])
        tied = rerank.zero_linear_bundle(
            rerank.FeatureTables(
                item_popularity=np.zeros(3, dtype=np.float64),
                item_recency=np.zeros(3, dtype=np.float64),
                item_genres=np.zeros((3, 1), dtype=np.float64),
                user_genre_profile=np.zeros((1, 1), dtype=np.float64),
                genres_available=False,
            ),
            stage1_model="fixture",
            popularity_weight=1.0,
        )
        tie_order = rerank.order_candidates(tied, 0, [2, 0, 1], [1.0, 1.0, 1.0])
        self.assertEqual([item_id for item_id, _score in tie_order], [0, 1, 2])
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.npz"
            with self.assertRaises(rerank.RerankError) as ctx:
                rerank.load_rerank_artifact(missing)
        self.assertIn("does not exist", str(ctx.exception))


class TwoStageEvalTests(unittest.TestCase):
    def test_tiny_eval_is_finite_and_repeatable(self) -> None:
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
        ]
        split = prep.split_interactions(interactions, num_users=2, num_movies=12)
        genres = np.zeros((split.num_movies, 2), dtype=np.float64)
        genres[:, 0] = 1.0
        tables = rerank.build_feature_tables(split, genres)

        def score_user(user_id: int) -> np.ndarray:
            scores = np.linspace(0.0, 1.0, split.num_movies)
            scores[user_id] += 0.01
            return scores

        config = rerank.RerankConfig(
            stage1_n=12,
            k=2,
            n_epochs=3,
            hidden_dim=4,
            learning_rate=0.05,
            l2=1e-4,
            seed=7,
        )
        first = rerank.evaluate_two_stage(
            split, score_user, tables, config, stage1_model="fixture"
        )
        second = rerank.evaluate_two_stage(
            split, score_user, tables, config, stage1_model="fixture"
        )
        self.assertEqual(first["n_users"], second["n_users"])
        self.assertGreater(first["n_users"], 0)
        self.assertEqual(first["retrieval_only"], second["retrieval_only"])
        self.assertEqual(first["retrieval_plus_rerank"], second["retrieval_plus_rerank"])
        np.testing.assert_allclose(first["bundle"].w1, second["bundle"].w1)
        for block in (first["retrieval_only"], first["retrieval_plus_rerank"]):
            for key in ("recall_at_k", "ndcg_at_k", "map_at_k"):
                self.assertGreaterEqual(block[key], 0.0)
                self.assertLessEqual(block[key], 1.0)
        self.assertGreater(first["rerank_train"]["n_train_users"], 0)
        self.assertIn("test", first["rerank_train"]["excluded_from_training_pools"])


class CommittedRerankEvalTests(unittest.TestCase):
    def test_committed_rerank_eval_matches_markdown(self) -> None:
        import sagerec_rerank_eval as rerank_eval

        self.assertTrue(_RERANK_JSON.is_file(), "results/rerank_eval.json is missing")
        self.assertTrue(_RERANK_MD.is_file(), "results/rerank_eval.md is missing")
        payload = json.loads(_RERANK_JSON.read_text(encoding="utf-8"))
        for key in rerank_eval.REQUIRED_RERANK_EVAL_KEYS:
            self.assertIn(key, payload)
        self.assertEqual(payload["protocol_version"], rerank_eval.PROTOCOL_VERSION)
        self.assertEqual(payload["adr"], "ADR-010")
        self.assertEqual(payload["dataset_edition"], "100k")
        self.assertEqual(payload["split_policy_id"], prep.SPLIT_POLICY_ID)
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["stage1"]["model"], "implicit_mf")
        self.assertEqual(payload["stage1"]["stage1_n"], 200)
        self.assertEqual(payload["stage2"]["model"], "numpy_mlp_listwise")
        self.assertEqual(payload["stage2"]["feature_names"], list(rerank.FEATURE_NAMES))
        self.assertTrue(payload["stage2"]["genres_available"])
        self.assertEqual(payload["n_users"], 943)
        self.assertGreaterEqual(payload["stage1_hit_rate_at_n"], 0.0)
        self.assertLessEqual(payload["stage1_hit_rate_at_n"], 1.0)
        for name in ("retrieval_only", "retrieval_plus_rerank"):
            block = payload[name]
            for key in ("recall_at_10", "ndcg_at_10", "map"):
                self.assertGreaterEqual(block[key], 0.0)
                self.assertLessEqual(block[key], 1.0)
        delta = payload["delta_retrieval_plus_rerank_minus_retrieval_only"]
        retrieval = payload["retrieval_only"]
        reranked = payload["retrieval_plus_rerank"]
        for key in ("recall_at_10", "ndcg_at_10", "map"):
            self.assertAlmostEqual(delta[key], reranked[key] - retrieval[key])
        note = payload["note"].lower()
        self.assertIn("not a graphsage", note)
        self.assertIn("map@10", note)
        rendered = rerank_eval.render_rerank_eval_markdown(payload)
        self.assertEqual(_RERANK_MD.read_text(encoding="utf-8"), rendered)
        self.assertIn("retrieval + re-rank", rendered)


if __name__ == "__main__":
    unittest.main()
