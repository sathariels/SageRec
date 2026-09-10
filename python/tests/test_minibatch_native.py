"""Phase 4 native-backed mini-batch neighborhood tests.

Tiny synthetic graphs only. Sampling must call compiled graph_sampler;
these tests fail if the extension is missing or bypassed. No MovieLens
files, no GraphSAGE training, no quality metrics.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import graph_sampler
import sagerec_minibatch as minibatch
import sagerec_prep as prep
import sagerec_reference_sampler as reference


def _synthetic_train_pairs() -> tuple[int, int, list[tuple[int, int]]]:
    # users {0,1,2}, movies {0..4} -> global movies {3..7}
    # user 0: all five movies (degree 5)
    # user 1: movies 0,1 (degree 2)
    # user 2: isolated
    return (
        3,
        5,
        [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 1),
        ],
    )


def _row(
    user_id: int, movie_id: int, timestamp: int, rating: int = 1
) -> prep.NormalizedInteraction:
    return prep.NormalizedInteraction(user_id, movie_id, timestamp, rating)


class LoadGraphSamplerTests(unittest.TestCase):
    def test_loads_compiled_extension(self) -> None:
        module = minibatch.load_graph_sampler()
        self.assertIs(module, graph_sampler)
        path = Path(module.__file__)
        self.assertTrue(path.name.startswith("graph_sampler"))
        self.assertTrue(path.name.endswith(".so") or ".so." in path.name)
        self.assertTrue(callable(module.BipartiteCSR.sample_neighbors))

    def test_helper_does_not_import_reference_sampler(self) -> None:
        source = Path(minibatch.__file__).read_text(encoding="utf-8")
        self.assertNotIn("sagerec_reference_sampler", source)
        self.assertIn("call_native_sample_neighbors", source)

    def test_missing_graph_sampler_fails_actionably(self) -> None:
        real = sys.modules.get("graph_sampler")
        sys.modules["graph_sampler"] = None
        try:
            with self.assertRaises(ImportError) as ctx:
                minibatch.load_graph_sampler()
            message = str(ctx.exception)
            self.assertIn("graph_sampler", message)
            self.assertIn("PYTHONPATH", message)
            self.assertIn("CMake", message)
        finally:
            if real is not None:
                sys.modules["graph_sampler"] = real
            else:
                sys.modules.pop("graph_sampler", None)

    def test_non_compiled_module_is_rejected(self) -> None:
        fake = mock.Mock()
        fake.__file__ = str(_PYTHON_DIR / "graph_sampler.py")
        fake.BipartiteCSR = mock.Mock()
        fake.BipartiteCSR.sample_neighbors = mock.Mock()
        with mock.patch.dict(sys.modules, {"graph_sampler": fake}):
            with self.assertRaises(ImportError) as ctx:
                minibatch.load_graph_sampler()
            self.assertIn("compiled", str(ctx.exception).lower())
            self.assertIn("graph_sampler", str(ctx.exception))


class NativeMinibatchSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        num_users, num_movies, pairs = _synthetic_train_pairs()
        self.num_users = num_users
        self.num_movies = num_movies
        self.pairs = pairs
        self.sampler = minibatch.NativeMinibatchSampler.from_train_pairs(
            num_users, num_movies, pairs
        )
        self.graph = self.sampler.graph

    def test_from_train_pairs_is_native_csr(self) -> None:
        self.assertIsInstance(self.graph, graph_sampler.BipartiteCSR)
        self.assertEqual(self.sampler.num_users, 3)
        self.assertEqual(self.sampler.num_movies, 5)
        self.assertEqual(self.sampler.num_nodes, 8)
        self.assertEqual(self.sampler.offsets()[0], 0)
        self.assertEqual(self.sampler.offsets()[-1], len(self.sampler.neighbors()))

    def test_rejects_non_native_graph(self) -> None:
        class FakeGraph:
            def sample_neighbors(self, node_id: int, k: int, seed: int) -> list[int]:
                return [node_id]

        with self.assertRaises(TypeError) as ctx:
            minibatch.NativeMinibatchSampler(FakeGraph())
        self.assertIn("graph_sampler.BipartiteCSR", str(ctx.exception))

    def test_from_graph_wraps_compiled_csr_offsets_neighbors(self) -> None:
        wrapped = minibatch.NativeMinibatchSampler.from_graph(self.graph)
        self.assertIs(wrapped.graph, self.graph)
        self.assertEqual(wrapped.offsets(), self.graph.offsets())
        self.assertEqual(wrapped.neighbors(), self.graph.neighbors())

    def test_sample_neighbors_calls_native_not_reference(self) -> None:
        with mock.patch.object(
            minibatch,
            "call_native_sample_neighbors",
            wraps=minibatch.call_native_sample_neighbors,
        ) as native:
            with mock.patch.object(
                reference, "sample_neighbors", wraps=reference.sample_neighbors
            ) as ref:
                sampled = self.sampler.sample_neighbors(0, 3, 42)
        ref.assert_not_called()
        native.assert_called_once()
        args, kwargs = native.call_args
        self.assertIs(args[0], self.graph)
        self.assertEqual(args[1:], (0, 3, 42))
        self.assertEqual(kwargs, {})
        self.assertEqual(sampled, self.graph.sample_neighbors(0, 3, 42))
        self.assertEqual(len(sampled), 3)
        self.assertEqual(len(set(sampled)), 3)

    def test_k_zero_isolated_and_full_neighborhood(self) -> None:
        self.assertEqual(self.sampler.sample_neighbors(0, 0, 11), [])
        self.assertEqual(self.sampler.sample_neighbors(2, 4, 9), [])
        self.assertEqual(self.sampler.sample_neighbors(2, 0, 1), [])

        expected_user0 = [3, 4, 5, 6, 7]
        self.assertEqual(self.sampler.degree(0), 5)
        self.assertEqual(self.sampler.sample_neighbors(0, 5, 99), expected_user0)
        self.assertEqual(self.sampler.sample_neighbors(0, 100, 0), expected_user0)

        expected_user1 = [3, 4]
        self.assertEqual(self.sampler.sample_neighbors(1, 2, 1), expected_user1)
        self.assertEqual(self.sampler.sample_neighbors(1, 8, 7), expected_user1)

        begin = self.sampler.offsets()[0]
        end = self.sampler.offsets()[1]
        self.assertEqual(self.sampler.neighbors()[begin:end], expected_user0)

    def test_seed_reproducibility(self) -> None:
        first = self.sampler.sample_neighbors(0, 3, 42)
        second = self.sampler.sample_neighbors(0, 3, 42)
        native = self.graph.sample_neighbors(0, 3, 42)
        self.assertEqual(first, second)
        self.assertEqual(first, native)
        _ = self.sampler.sample_neighbors(0, 3, 99)
        self.assertEqual(self.sampler.sample_neighbors(0, 3, 42), first)

    def test_multihop_calls_native_with_derived_seeds(self) -> None:
        with mock.patch.object(
            minibatch,
            "call_native_sample_neighbors",
            wraps=minibatch.call_native_sample_neighbors,
        ) as native:
            with mock.patch.object(
                reference, "sample_neighbors", wraps=reference.sample_neighbors
            ) as ref:
                batch = self.sampler.sample_multihop([0, 1], [2, 2], seed=7)
        ref.assert_not_called()
        self.assertGreaterEqual(native.call_count, 2)

        self.assertEqual(batch.seed_nodes, (0, 1))
        self.assertEqual(batch.fanouts, (2, 2))
        self.assertEqual(batch.seed, 7)
        self.assertEqual(batch.sources[0], (0, 1))
        self.assertEqual(len(batch.hops), 2)
        self.assertEqual(len(batch.hops[0]), 2)

        first_calls = [call.args[1:] for call in native.call_args_list[:2]]
        expected_first = [
            (0, 2, minibatch.derived_sample_seed(7, 0, 0)),
            (1, 2, minibatch.derived_sample_seed(7, 0, 1)),
        ]
        self.assertEqual(first_calls, expected_first)
        for call in native.call_args_list:
            self.assertIs(call.args[0], self.graph)
            node_id, k, _seed = call.args[1:]
            self.assertEqual(k, 2)
            self.assertGreaterEqual(node_id, 0)
            self.assertLess(node_id, self.sampler.num_nodes)

        hop0_frontier = []
        for neighbors in batch.hops[0]:
            self.assertEqual(len(neighbors), len(set(neighbors)))
            hop0_frontier.extend(neighbors)
        self.assertEqual(batch.sources[1], tuple(hop0_frontier))
        self.assertEqual(len(batch.hops[1]), len(hop0_frontier))

    def test_multihop_empty_k_zero_and_full_neighborhood(self) -> None:
        empty_seeds = self.sampler.sample_multihop([], [2], seed=1)
        self.assertEqual(empty_seeds.sources, ((),))
        self.assertEqual(empty_seeds.hops, ((),))

        isolated = self.sampler.sample_multihop([2], [3, 3], seed=4)
        self.assertEqual(isolated.sources[0], (2,))
        self.assertEqual(isolated.hops[0], ((),))
        self.assertEqual(isolated.sources[1], ())
        self.assertEqual(isolated.hops[1], ())

        k_zero = self.sampler.sample_multihop([0], [0], seed=8)
        self.assertEqual(k_zero.hops[0], ((),))

        full = self.sampler.sample_multihop([0], [5], seed=0)
        self.assertEqual(full.hops[0][0], (3, 4, 5, 6, 7))

        no_hops = self.sampler.sample_multihop([0], [], seed=3)
        self.assertEqual(no_hops.hops, ())
        self.assertEqual(no_hops.sources, ())

    def test_multihop_is_reproducible(self) -> None:
        first = self.sampler.sample_multihop([0], [3, 2], seed=42)
        second = self.sampler.sample_multihop([0], [3, 2], seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first.hops[0][0]), 3)

    def test_invalid_fanout_and_seed_raise(self) -> None:
        with self.assertRaises(ValueError) as fanout_ctx:
            self.sampler.sample_multihop([0], [-1], seed=1)
        self.assertIn("fanout", str(fanout_ctx.exception))

        with self.assertRaises(ValueError) as seed_ctx:
            self.sampler.sample_multihop([0], [1], seed=-5)
        self.assertIn("seed", str(seed_ctx.exception))

        with self.assertRaises(graph_sampler.GraphError) as node_ctx:
            self.sampler.sample_neighbors(99, 1, 0)
        self.assertIn("node_id", str(node_ctx.exception))

    def test_train_pairs_exclude_held_out_positives(self) -> None:
        interactions = [
            _row(0, 0, 10),
            _row(0, 1, 20),
            _row(0, 2, 30),
            _row(0, 3, 40),
        ]
        split = prep.split_interactions(interactions, num_users=1, num_movies=4)
        held_out = {
            (item.user_id, item.movie_id)
            for item in split.assignments
            if item.split != "train"
        }
        self.assertEqual(held_out, {(0, 2), (0, 3)})

        sampler = minibatch.NativeMinibatchSampler.from_train_pairs(
            split.num_users, split.num_movies, split.train_positive_pairs()
        )
        user_neighbors = set(
            sampler.neighbors()[sampler.offsets()[0] : sampler.offsets()[1]]
        )
        held_out_global = {split.num_users + movie_id for _, movie_id in held_out}
        self.assertTrue(user_neighbors.isdisjoint(held_out_global))
        self.assertEqual(user_neighbors, {1, 2})  # movies 0 and 1 -> global 1, 2


if __name__ == "__main__":
    unittest.main()
