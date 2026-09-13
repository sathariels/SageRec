"""Phase 2 native-vs-reference sampler timing tests.

Default cases use a tiny deterministic synthetic graph. They verify
parity-before-timing and the timing JSON schema. They do not download
MovieLens and they do not assert machine-specific speedups.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_PYTHON_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_reference_sampler as reference
import sagerec_sampler_benchmark as bench


def _fixture_environment() -> dict:
    return {
        "git_commit": "abc123",
        "generated_at": "2026-09-13T00:00:00+00:00",
        "python": "3.12.3",
        "python_implementation": "CPython",
        "build_type": "Release",
        "compiler": {
            "id": "GNU",
            "path": "/usr/bin/g++",
            "version": "13.2.0",
            "version_line": "g++ (Debian) 13.2.0",
        },
        "cpu": {"model": "test-cpu", "logical_cpus": 4, "machine": "x86_64"},
        "platform": {
            "system": "Linux",
            "release": "6.12",
            "machine": "x86_64",
            "processor": "x86_64",
        },
    }


def _fixture_graph_meta() -> dict:
    return {
        "kind": "synthetic_bipartite",
        "num_users": 8,
        "num_movies": 16,
        "num_nodes": 24,
        "num_undirected_edges": 32,
        "num_directed_csr_entries": 64,
        "requested_degree": 4,
    }


def _fixture_workload() -> dict:
    return {
        "k": 2,
        "n_queries": 16,
        "warmup_repetitions": 1,
        "measured_repetitions": 3,
        "replacement_policy": bench.REPLACEMENT_POLICY,
    }


class SyntheticGraphTests(unittest.TestCase):
    def test_pairs_are_deterministic_and_regular(self) -> None:
        first = bench.synthetic_train_pairs(4, 8, 3)
        second = bench.synthetic_train_pairs(4, 8, 3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4 * 3)
        self.assertEqual(first[0], (0, 0))
        self.assertTrue(all(0 <= movie < 8 for _user, movie in first))

    def test_rejects_invalid_degree(self) -> None:
        with self.assertRaises(bench.BenchmarkError) as ctx:
            bench.synthetic_train_pairs(4, 3, 5)
        self.assertIn("degree", str(ctx.exception))

    def test_ci_smoke_config_is_synthetic_and_tiny(self) -> None:
        config = bench.ci_smoke_config()
        self.assertEqual(config.source, "synthetic")
        self.assertLessEqual(config.n_queries, 64)
        self.assertLessEqual(config.num_users, 16)
        self.assertGreaterEqual(config.warmup_repetitions, 1)
        self.assertGreaterEqual(config.measured_repetitions, 2)
        self.assertIsNone(config.processed_dir)


class ParityBeforeTimingTests(unittest.TestCase):
    def test_parity_mismatch_skips_timing(self) -> None:
        config = bench.ci_smoke_config()
        measure = mock.Mock(side_effect=AssertionError("timed before parity"))

        def broken_reference(offsets, neighbors, node_id, k, seed):
            return [999]

        with self.assertRaises(bench.BenchmarkError) as ctx:
            bench.run_benchmark(
                config,
                reference_sample=broken_reference,
                measure=measure,
            )
        self.assertIn("parity failed before timing", str(ctx.exception))
        measure.assert_not_called()

    def test_native_and_reference_match_on_smoke_graph(self) -> None:
        config = bench.ci_smoke_config()
        graph, _meta = bench.load_benchmark_graph(config)
        offsets = list(graph.offsets())
        neighbors = list(graph.neighbors())
        queries = bench.build_queries(
            graph, k=config.k, n_queries=config.n_queries, seed=config.seed
        )
        bench.verify_parity(graph, queries, offsets, neighbors)
        self.assertGreaterEqual(len(queries), 1)
        native = graph.sample_neighbors(*queries[0])
        ref = reference.sample_neighbors(offsets, neighbors, *queries[0])
        self.assertEqual(native, ref)
        self.assertEqual(len(native), len(set(native)))


class TimingSchemaTests(unittest.TestCase):
    def test_payload_schema_and_computed_speedup(self) -> None:
        native_times = [0.20, 0.10, 0.30]
        reference_times = [1.00, 0.80, 1.20]
        payload = bench.build_timing_payload(
            graph_meta=_fixture_graph_meta(),
            workload=_fixture_workload(),
            native_times=native_times,
            reference_times=reference_times,
            environment=_fixture_environment(),
            seed=7,
        )
        bench.validate_timing_payload(payload)
        for key in bench.REQUIRED_TIMING_KEYS:
            self.assertIn(key, payload)
        self.assertEqual(payload["parity_verified"], True)
        self.assertEqual(payload["replacement_policy"], "uniform_without_replacement")
        self.assertEqual(payload["adr"], "ADR-005")
        self.assertEqual(payload["timing_statistic"]["name"], "median")
        self.assertEqual(payload["native"]["seconds"]["median"], 0.20)
        self.assertEqual(payload["reference"]["seconds"]["median"], 1.00)
        self.assertAlmostEqual(payload["speedup"]["native_over_reference"], 5.0)
        self.assertEqual(
            payload["speedup"]["formula"],
            "reference_median_seconds / native_median_seconds",
        )
        self.assertAlmostEqual(
            payload["native"]["throughput_queries_per_s"]["median"],
            16 / 0.20,
        )
        self.assertEqual(payload["build_type"], "Release")
        self.assertEqual(payload["compiler"]["id"], "GNU")
        self.assertEqual(payload["python"], "3.12.3")
        self.assertEqual(payload["cpu"]["model"], "test-cpu")
        self.assertEqual(payload["seed"], 7)
        self.assertIn("not a production", payload["note"].lower())

    def test_speedup_rejects_non_positive_times(self) -> None:
        with self.assertRaises(bench.BenchmarkError):
            bench.compute_speedup(1.0, 0.0)
        with self.assertRaises(bench.BenchmarkError):
            bench.compute_speedup(-1.0, 0.1)

    def test_write_artifacts_round_trip(self) -> None:
        payload = bench.build_timing_payload(
            graph_meta=_fixture_graph_meta(),
            workload=_fixture_workload(),
            native_times=[0.04, 0.05, 0.06],
            reference_times=[0.40, 0.50, 0.60],
            environment=_fixture_environment(),
            seed=7,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = bench.write_timing_artifacts(
                payload,
                json_path=root / "results" / "sampler_timing.json",
                markdown_path=root / "results" / "sampler_timing.md",
                chart_path=root / "results" / "sampler_timing.svg",
                repo_root=root,
            )
            loaded = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
            bench.validate_timing_payload(loaded)
            self.assertEqual(loaded["chart_path"], "results/sampler_timing.svg")
            self.assertEqual(loaded["markdown_path"], "results/sampler_timing.md")
            markdown = artifacts.markdown_path.read_text(encoding="utf-8")
            self.assertIn("median", markdown.lower())
            self.assertIn("graph_sampler", markdown)
            self.assertIn("Python reference", markdown)
            svg = artifacts.chart_path.read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertIn("0.0500s", svg)
            self.assertIn("0.5000s", svg)
            self.assertIn("native graph_sampler", svg)
            self.assertIn("Python reference", svg)

    def test_validate_rejects_missing_fields(self) -> None:
        with self.assertRaises(bench.BenchmarkError) as ctx:
            bench.validate_timing_payload({"benchmark": bench.BENCHMARK_NAME})
        self.assertIn("missing required keys", str(ctx.exception))


class EndToEndSmokeTests(unittest.TestCase):
    def test_run_benchmark_writes_required_schema(self) -> None:
        payload = bench.run_benchmark(
            bench.ci_smoke_config(),
            generated_at="2026-09-13T00:00:00+00:00",
        )
        bench.validate_timing_payload(payload)
        self.assertEqual(payload["graph"]["kind"], "synthetic_bipartite")
        self.assertEqual(payload["workload"]["k"], 2)
        self.assertEqual(payload["workload"]["n_queries"], 16)
        self.assertEqual(len(payload["native"]["seconds"]["repetitions"]), 2)
        self.assertEqual(len(payload["reference"]["seconds"]["repetitions"]), 2)
        self.assertGreater(payload["speedup"]["native_over_reference"], 0.0)
        self.assertTrue(payload["parity_verified"])
        self.assertNotEqual(payload["graph"]["kind"], "movielens_100k_train_only")

    def test_default_path_does_not_import_dataset_or_download(self) -> None:
        source = Path(bench.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import sagerec_download", source)
        self.assertIn("source='synthetic'", source)
        script = (_SCRIPTS_DIR / "run_sampler_timing.py").read_text(encoding="utf-8")
        self.assertIn("sagerec_sampler_benchmark", script)
        self.assertIn("--source", script)
        self.assertIn("synthetic", script)

    def test_movielens_source_requires_processed_dir_and_does_not_download(self) -> None:
        with self.assertRaises(bench.BenchmarkError) as ctx:
            bench.load_benchmark_graph(
                bench.SamplerBenchmarkConfig(source="movielens-100k")
            )
        self.assertIn("processed_dir", str(ctx.exception))
        self.assertIn("synthetic", str(ctx.exception))
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "processed"
            with self.assertRaises(Exception) as load_ctx:
                bench.load_benchmark_graph(
                    bench.SamplerBenchmarkConfig(
                        source="movielens-100k",
                        processed_dir=missing,
                    )
                )
            self.assertNotIn("grouplens.org", str(load_ctx.exception).lower())

    def test_committed_results_match_schema_when_present(self) -> None:
        path = (
            Path(__file__).resolve().parents[2] / "results" / "sampler_timing.json"
        )
        if not path.is_file():
            self.skipTest("committed sampler timing JSON is not present yet")
        payload = json.loads(path.read_text(encoding="utf-8"))
        bench.validate_timing_payload(payload)
        self.assertEqual(payload["graph"]["kind"], "synthetic_bipartite")
        self.assertTrue(payload["parity_verified"])
        self.assertEqual(payload["replacement_policy"], "uniform_without_replacement")
        self.assertNotIn("recall_at_k", payload)
        self.assertNotIn("ndcg_at_k", payload)


if __name__ == "__main__":
    unittest.main()
