"""Binding smoke tests for the compiled graph_sampler module.

Requires PYTHONPATH to include the CMake build directory that contains
graph_sampler*.so. Uses a tiny synthetic graph and in-memory u.data strings;
no MovieLens files.
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


class MovieLens100kBindingTests(unittest.TestCase):
    # Same fixture as cpp/tests/test_movielens_100k.cpp. Source IDs are not
    # contiguous: users {7, 10} -> local {0, 1}, movies {8, 20} -> local {0, 1}.
    _TINY_UDATA = "10\t20\t5\t100\n7\t20\t4\t200\n10\t8\t1\t50\n"

    def test_parse_mappings_and_fields(self) -> None:
        parsed = graph_sampler.parse_movielens_100k(self._TINY_UDATA)
        self.assertEqual(parsed.num_users, 2)
        self.assertEqual(parsed.num_movies, 2)
        self.assertEqual(parsed.user_source_ids, [7, 10])
        self.assertEqual(parsed.movie_source_ids, [8, 20])
        self.assertEqual(len(parsed.interactions), 3)

        first = parsed.interactions[0]
        self.assertEqual(first.user_id, 1)
        self.assertEqual(first.movie_id, 1)
        self.assertEqual(first.rating, 5)
        self.assertEqual(first.timestamp, 100)

        second = parsed.interactions[1]
        self.assertEqual(second.user_id, 0)
        self.assertEqual(second.movie_id, 1)
        self.assertEqual(second.rating, 4)
        self.assertEqual(second.timestamp, 200)

        third = parsed.interactions[2]
        self.assertEqual(third.user_id, 1)
        self.assertEqual(third.movie_id, 0)
        self.assertEqual(third.rating, 1)
        self.assertEqual(third.timestamp, 50)

        self.assertEqual(parsed.local_pairs(), [(1, 1), (0, 1), (1, 0)])

    def test_local_pairs_build_csr(self) -> None:
        parsed = graph_sampler.parse_movielens_100k(self._TINY_UDATA)
        graph = graph_sampler.BipartiteCSR(
            parsed.num_users, parsed.num_movies, parsed.local_pairs()
        )
        self.assertEqual(graph.num_users, 2)
        self.assertEqual(graph.num_movies, 2)
        self.assertEqual(graph.num_nodes, 4)
        self.assertEqual(graph.offsets(), [0, 1, 3, 4, 6])
        self.assertEqual(graph.neighbors(), [3, 2, 3, 1, 0, 1])

    def test_malformed_input_raises_graph_error(self) -> None:
        with self.assertRaises(graph_sampler.GraphError) as ctx:
            graph_sampler.parse_movielens_100k("1\t2\t3\n")
        self.assertIn("expected 4 tab-separated fields", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
