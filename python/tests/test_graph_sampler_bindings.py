"""Binding smoke tests for the compiled graph_sampler module.

Requires PYTHONPATH to include the CMake build directory that contains
graph_sampler*.so. Uses only a tiny synthetic graph; no MovieLens files.
"""

from __future__ import annotations

import unittest

import graph_sampler


class GraphSamplerBindingTests(unittest.TestCase):
    def _synthetic_graph(self) -> graph_sampler.BipartiteCSR:
        return graph_sampler.BipartiteCSR(2, 3, [(0, 0), (0, 1), (1, 1)])

    def test_import_construct_and_metadata(self) -> None:
        graph = self._synthetic_graph()
        self.assertEqual(graph.num_users, 2)
        self.assertEqual(graph.num_movies, 3)
        self.assertEqual(graph.num_nodes, 5)
        self.assertEqual(graph.degree(0), 2)
        self.assertEqual(graph.degree(4), 0)
        self.assertEqual(graph.offsets()[0], 0)
        self.assertEqual(graph.offsets()[-1], len(graph.neighbors()))

    def test_seeded_sample_is_deterministic(self) -> None:
        graph = self._synthetic_graph()
        first = graph.sample_neighbors(0, 1, 7)
        second = graph.sample_neighbors(0, 1, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertIn(first[0], (2, 3))

    def test_full_neighborhood_and_empty_cases(self) -> None:
        graph = self._synthetic_graph()
        self.assertEqual(graph.sample_neighbors(0, 2, 1), [2, 3])
        self.assertEqual(graph.sample_neighbors(0, 10, 1), [2, 3])
        self.assertEqual(graph.sample_neighbors(0, 0, 1), [])
        self.assertEqual(graph.sample_neighbors(4, 3, 1), [])

    def test_invalid_ids_raise_graph_error(self) -> None:
        graph = self._synthetic_graph()
        with self.assertRaises(graph_sampler.GraphError) as ctx:
            graph.sample_neighbors(99, 1, 0)
        self.assertIn("node_id", str(ctx.exception))
        self.assertIn("[0, 5)", str(ctx.exception))

        with self.assertRaises(graph_sampler.GraphError):
            graph.sample_neighbors(0, -1, 0)

        with self.assertRaises(graph_sampler.GraphError):
            graph.sample_neighbors(0, 1, -5)

        with self.assertRaises(graph_sampler.GraphError):
            graph_sampler.BipartiteCSR(1, 1, [(0, 4)])


if __name__ == "__main__":
    unittest.main()
