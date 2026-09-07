"""Leakage-safe ADR-003 leave-one-out tests on tiny synthetic interactions.

Requires PYTHONPATH to include the CMake build directory (graph_sampler) and
``python/`` (sagerec_prep). Fixtures are in-memory only; no MovieLens files.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import graph_sampler
import sagerec_prep as prep


def _row(
    user_id: int, movie_id: int, timestamp: int, rating: int = 1
) -> prep.NormalizedInteraction:
    return prep.NormalizedInteraction(user_id, movie_id, timestamp, rating)


class LeaveOneOutSplitTests(unittest.TestCase):
    def test_eligible_user_holds_out_latest_and_second_latest(self) -> None:
        # Input order is not chronological: latest is movie 1 at t=40.
        interactions = [
            _row(0, 2, 10),
            _row(0, 0, 20),
            _row(0, 1, 40),
            _row(0, 3, 30),
        ]
        result = prep.split_interactions(interactions, num_users=1, num_movies=4)
        by_movie = {item.movie_id: item.split for item in result.assignments}
        self.assertEqual(by_movie, {2: "train", 0: "train", 3: "validation", 1: "test"})
        self.assertEqual(result.eligible_user_ids, (0,))
        self.assertEqual(result.cold_start_user_ids, ())
        self.assertEqual(result.train_positive_pairs(), [(0, 2), (0, 0)])

    def test_timestamp_tie_uses_local_user_and_movie_ids(self) -> None:
        # Same timestamp for user 0; local movie_id breaks the tie.
        # Ranked (t, user, movie): (100,0,0), (100,0,1), (100,0,2)
        # -> train movie 0, validation movie 1, test movie 2.
        interactions = [
            _row(0, 2, 100),
            _row(0, 0, 100),
            _row(0, 1, 100),
        ]
        result = prep.split_interactions(interactions)
        by_movie = {item.movie_id: item.split for item in result.assignments}
        self.assertEqual(by_movie[0], "train")
        self.assertEqual(by_movie[1], "validation")
        self.assertEqual(by_movie[2], "test")

        # Shuffling input order must not change the assignment.
        shuffled = [_row(0, 1, 100), _row(0, 2, 100), _row(0, 0, 100)]
        again = prep.split_interactions(shuffled)
        self.assertEqual(
            {(item.movie_id, item.split) for item in again.assignments},
            {(item.movie_id, item.split) for item in result.assignments},
        )

    def test_cold_start_users_stay_in_train_and_are_ineligible(self) -> None:
        interactions = [
            _row(0, 0, 1),
            _row(1, 0, 2),
            _row(1, 1, 3),
        ]
        result = prep.split_interactions(interactions, num_users=2, num_movies=2)
        self.assertEqual(result.eligible_user_ids, ())
        self.assertEqual(result.cold_start_user_ids, (0, 1))
        self.assertEqual({item.split for item in result.assignments}, {"train"})
        self.assertEqual(result.train_positive_pairs(), [(0, 0), (1, 0), (1, 1)])
        self.assertEqual(result.interactions_for("validation"), ())
        self.assertEqual(result.interactions_for("test"), ())

    def test_mixed_eligible_and_cold_start(self) -> None:
        interactions = [
            _row(0, 0, 10),
            _row(0, 1, 20),
            _row(0, 2, 30),
            _row(1, 0, 5),
            _row(1, 2, 15),
        ]
        result = prep.split_interactions(interactions, num_users=2, num_movies=3)
        self.assertEqual(result.eligible_user_ids, (0,))
        self.assertEqual(result.cold_start_user_ids, (1,))
        self.assertEqual(
            {(item.user_id, item.movie_id, item.split) for item in result.assignments},
            {
                (0, 0, "train"),
                (0, 1, "validation"),
                (0, 2, "test"),
                (1, 0, "train"),
                (1, 2, "train"),
            },
        )

    def test_held_out_positives_never_enter_train_pair_set(self) -> None:
        interactions = [
            _row(0, 0, 1),
            _row(0, 1, 2),
            _row(0, 2, 3),
            _row(0, 3, 4),
            _row(1, 1, 8),
        ]
        result = prep.split_interactions(interactions, num_users=2, num_movies=4)
        train = result.positive_pair_set("train")
        held_out = result.positive_pair_set("validation") | result.positive_pair_set("test")
        self.assertTrue(held_out)
        self.assertTrue(train.isdisjoint(held_out))
        self.assertEqual(held_out, {(0, 2), (0, 3)})
        self.assertEqual(train, {(0, 0), (0, 1), (1, 1)})

    def test_train_pairs_csr_excludes_validation_and_test_edges(self) -> None:
        interactions = [
            _row(0, 0, 10),
            _row(0, 1, 20),
            _row(0, 2, 30),
            _row(1, 0, 1),
        ]
        result = prep.split_interactions(interactions, num_users=2, num_movies=3)
        pairs = prep.train_positive_pairs(result)
        self.assertEqual(pairs, [(0, 0), (1, 0)])

        graph = graph_sampler.BipartiteCSR(result.num_users, result.num_movies, pairs)
        offsets = graph.offsets()
        neighbors = graph.neighbors()

        def neighborhood(node_id: int) -> list[int]:
            return neighbors[offsets[node_id] : offsets[node_id + 1]]

        # Movie global IDs are num_users + local movie_id.
        self.assertEqual(neighborhood(0), [2])  # user 0 -> movie 0 only
        self.assertNotIn(2 + 1, neighborhood(0))  # held-out validation movie 1
        self.assertNotIn(2 + 2, neighborhood(0))  # held-out test movie 2
        self.assertEqual(graph.degree(3), 0)  # movie 1 never in train
        self.assertEqual(graph.degree(4), 0)  # movie 2 never in train

    def test_split_is_deterministic(self) -> None:
        interactions = [
            _row(0, 2, 9),
            _row(0, 0, 9),
            _row(0, 1, 8),
            _row(2, 0, 1),
        ]
        first = prep.split_interactions(interactions, num_users=3, num_movies=3)
        second = prep.split_interactions(list(interactions), num_users=3, num_movies=3)
        self.assertEqual(first, second)
        self.assertEqual(prep.build_manifest(first), prep.build_manifest(second))

    def test_accepts_parser_structs_and_preserves_counts(self) -> None:
        # Users {5, 9} -> local {0, 1}; movies {3, 4, 8} -> local {0, 1, 2}.
        text = (
            "9\t8\t5\t10\n"
            "9\t3\t4\t20\n"
            "9\t4\t3\t30\n"
            "5\t3\t1\t40\n"
        )
        parsed = graph_sampler.parse_movielens_100k(text)
        result = prep.split_interactions(
            parsed.interactions,
            num_users=parsed.num_users,
            num_movies=parsed.num_movies,
        )
        self.assertEqual(result.num_users, 2)
        self.assertEqual(result.num_movies, 3)
        # Local user 1 is source 9 (three rows) -> eligible.
        # Local user 0 is source 5 (one row) -> cold-start.
        self.assertEqual(result.eligible_user_ids, (1,))
        self.assertEqual(result.cold_start_user_ids, (0,))
        self.assertEqual(
            {(item.user_id, item.movie_id, item.split) for item in result.assignments},
            {
                (1, 2, "train"),
                (1, 0, "validation"),
                (1, 1, "test"),
                (0, 0, "train"),
            },
        )

    def test_manifest_records_policy_counts_and_placeholders(self) -> None:
        interactions = [_row(0, 0, 1), _row(0, 1, 2), _row(0, 2, 3)]
        result = prep.split_interactions(interactions)
        manifest = prep.build_manifest(result)
        self.assertEqual(manifest["dataset_edition"], "100k")
        self.assertEqual(manifest["split_policy_id"], prep.SPLIT_POLICY_ID)
        self.assertEqual(manifest["split_policy_version"], prep.SPLIT_POLICY_VERSION)
        self.assertEqual(manifest["min_interactions_for_eval"], 3)
        self.assertEqual(manifest["cold_start_policy"], prep.COLD_START_POLICY)
        self.assertEqual(manifest["tie_break"], prep.TIE_BREAK)
        self.assertIsNone(manifest["seed"])
        self.assertIsNone(manifest["source_url"])
        self.assertIsNone(manifest["checksum"])
        self.assertIsNone(manifest["license"])
        self.assertEqual(
            manifest["counts"],
            {
                "users": 1,
                "movies": 3,
                "interactions": {
                    "total": 3,
                    "train": 1,
                    "validation": 1,
                    "test": 1,
                },
                "eligible_users": 1,
                "cold_start_users": 0,
            },
        )
        filled = prep.build_manifest(
            result,
            source_url="https://example.invalid/ml-100k.zip",
            checksum="sha256:placeholder",
            license="GroupLens MovieLens",
        )
        self.assertEqual(filled["source_url"], "https://example.invalid/ml-100k.zip")
        self.assertEqual(filled["checksum"], "sha256:placeholder")

    def test_rejects_empty_duplicates_negative_ids_and_non_100k(self) -> None:
        with self.assertRaises(ValueError) as empty:
            prep.split_interactions([])
        self.assertIn("empty", str(empty.exception))

        with self.assertRaises(ValueError) as dup:
            prep.split_interactions([_row(0, 0, 1), _row(0, 0, 2)])
        self.assertIn("duplicate", str(dup.exception))

        with self.assertRaises(ValueError) as negative:
            prep.split_interactions([_row(-1, 0, 1)])
        self.assertIn("negative", str(negative.exception))

        with self.assertRaises(ValueError) as edition:
            prep.split_interactions([_row(0, 0, 1)], dataset_edition="1m")
        self.assertIn("100K", str(edition.exception))

        with self.assertRaises(ValueError) as too_small:
            prep.split_interactions([_row(0, 2, 1)], num_movies=2)
        self.assertIn("num_movies", str(too_small.exception))


if __name__ == "__main__":
    unittest.main()
