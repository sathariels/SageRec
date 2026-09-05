"""Correctness-parity tests: Python reference sampler vs native graph_sampler.

Requires PYTHONPATH to include the CMake build directory that contains
graph_sampler*.so. Uses synthetic bipartite graphs only; no MovieLens files.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import graph_sampler
import sagerec_reference_sampler as reference


def _synthetic_graph() -> graph_sampler.BipartiteCSR:
    # users {0,1,2}, movies {0..4} -> global movies {3..7}
    # user 0: all five movies (degree 5)
    # user 1: movies 0,1 (degree 2)
    # user 2: isolated
    return graph_sampler.BipartiteCSR(
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


class ReferenceSamplerParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = _synthetic_graph()
        self.offsets = self.graph.offsets()
        self.neighbors = self.graph.neighbors()

    def _native(self, node_id: int, k: int, seed: int) -> list[int]:
        return self.graph.sample_neighbors(node_id, k, seed)

    def _reference(self, node_id: int, k: int, seed: int) -> list[int]:
        return reference.sample_neighbors(
            self.offsets, self.neighbors, node_id, k, seed
        )

    def test_k_zero_and_isolated_are_empty(self) -> None:
        self.assertEqual(self._native(0, 0, 11), [])
        self.assertEqual(self._reference(0, 0, 11), [])
        self.assertEqual(self._native(2, 4, 9), [])
        self.assertEqual(self._reference(2, 4, 9), [])
        self.assertEqual(self._native(2, 0, 1), [])
        self.assertEqual(self._reference(2, 0, 1), [])

    def test_full_neighborhood_matches_csr_order(self) -> None:
        expected_user0 = [3, 4, 5, 6, 7]
        self.assertEqual(self.graph.degree(0), 5)
        self.assertEqual(self._native(0, 5, 99), expected_user0)
        self.assertEqual(self._reference(0, 5, 99), expected_user0)
        self.assertEqual(self._native(0, 100, 0), expected_user0)
        self.assertEqual(self._reference(0, 100, 0), expected_user0)

        expected_user1 = [3, 4]
        self.assertEqual(self._native(1, 2, 1), expected_user1)
        self.assertEqual(self._reference(1, 2, 1), expected_user1)
        self.assertEqual(self._native(1, 8, 7), expected_user1)
        self.assertEqual(self._reference(1, 8, 7), expected_user1)

    def test_subset_and_edge_cases_match_native(self) -> None:
        queries: list[tuple[int, int, int]] = []
        for node_id in range(self.graph.num_nodes):
            for k in (0, 1, 2, 3, 4, 5, 10):
                for seed in (0, 1, 7, 42, 99, 12345, 2**31 - 1):
                    queries.append((node_id, k, seed))

        self.assertGreaterEqual(len(queries), 200)
        for node_id, k, seed in queries:
            native = self._native(node_id, k, seed)
            ref = self._reference(node_id, k, seed)
            self.assertEqual(
                native,
                ref,
                f"mismatch for node_id={node_id} k={k} seed={seed}",
            )
            self.assertEqual(len(native), len(set(native)))

    def test_from_graph_helper_matches_native(self) -> None:
        native = self.graph.sample_neighbors(0, 3, 42)
        via_helper = reference.sample_neighbors_from_graph(self.graph, 0, 3, 42)
        self.assertEqual(native, via_helper)
        self.assertEqual(len(native), 3)

    def test_same_tuple_is_reproducible_and_independent(self) -> None:
        first = self._reference(0, 3, 42)
        second = self._reference(0, 3, 42)
        native_first = self._native(0, 3, 42)
        native_second = self._native(0, 3, 42)
        self.assertEqual(first, second)
        self.assertEqual(native_first, native_second)
        self.assertEqual(first, native_first)

        # A later call with the same seed must not see leftover RNG state.
        _ = self._reference(0, 3, 99)
        again = self._reference(0, 3, 42)
        self.assertEqual(again, first)

    def test_invalid_inputs_raise_actionable_errors(self) -> None:
        with self.assertRaises(ValueError) as native_node:
            self._native(99, 1, 0)
        with self.assertRaises(ValueError) as ref_node:
            self._reference(99, 1, 0)
        self.assertIn("node_id", str(native_node.exception))
        self.assertIn("[0, 8)", str(native_node.exception))
        self.assertIn("node_id", str(ref_node.exception))
        self.assertIn("[0, 8)", str(ref_node.exception))

        with self.assertRaises(ValueError) as native_k:
            self._native(0, -1, 0)
        with self.assertRaises(ValueError) as ref_k:
            self._reference(0, -1, 0)
        self.assertIn("sample size", str(native_k.exception))
        self.assertIn("sample size", str(ref_k.exception))

        with self.assertRaises(ValueError) as native_seed:
            self._native(0, 1, -5)
        with self.assertRaises(ValueError) as ref_seed:
            self._reference(0, 1, -5)
        self.assertIn("seed", str(native_seed.exception))
        self.assertIn("seed", str(ref_seed.exception))

        with self.assertRaises(ValueError):
            reference.sample_neighbors(self.offsets, self.neighbors, 0, 1, True)


if __name__ == "__main__":
    unittest.main()
