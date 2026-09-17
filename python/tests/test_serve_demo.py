"""Phase 8 ADR-008 demo-CLI and checkpoint harness.

Synthetic graphs and temp-dir checkpoints only. These tests must not
download MovieLens or read/write stored 100K quality result files.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

_PYTHON_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PYTHON_DIR.parent
_SCRIPT = _REPO_ROOT / "scripts" / "demo_recommend.py"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import graph_sampler
import sagerec_baseline as baseline
import sagerec_graphsage as graphsage
import sagerec_minibatch as minibatch
import sagerec_prep as prep
import sagerec_reference_sampler as reference
import sagerec_serve as serve
from sagerec_scoring import PairScorer

_PHASE5_RESULT = _REPO_ROOT / "results" / "graphsage_movielens_100k.json"
_PHASE7_RESULT = _REPO_ROOT / "results" / "multiseed_gnn_vs_mf_movielens_100k.json"


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
        n_epochs=1,
        batch_size=4,
        learning_rate=0.1,
        n_negatives=1,
        l2=1e-4,
        seed=seed,
    )


class DictScorer:
    """Deterministic PairScorer for ranking-policy tests."""

    def __init__(
        self,
        scores: dict[tuple[int, int], float],
        default: float = 0.0,
    ) -> None:
        self.scores = scores
        self.default = default

    def score_pairs(
        self,
        user_ids: list[int] | tuple[int, ...],
        item_ids: list[int] | tuple[int, ...],
    ) -> list[float]:
        return [
            self.scores.get((int(user_id), int(item_id)), self.default)
            for user_id, item_id in zip(user_ids, item_ids)
        ]


def _parse_tsv(text: str) -> list[tuple[int, int, float]]:
    lines = [line for line in text.strip().splitlines() if line]
    self_header = lines[0].split("\t")
    if self_header != ["rank", "movie_id", "score"]:
        raise AssertionError(f"unexpected header {lines[0]!r}")
    rows: list[tuple[int, int, float]] = []
    for line in lines[1:]:
        rank_s, movie_s, score_s = line.split("\t")
        rows.append((int(rank_s), int(movie_s), float(score_s)))
    return rows


def _run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    code = serve.main(argv, stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


class ServeContractTests(unittest.TestCase):
    def test_module_uses_pairscorer_and_native_minibatch_not_pyg_or_download(self) -> None:
        source = Path(serve.__file__).read_text(encoding="utf-8")
        self.assertIn("sagerec_minibatch", source)
        self.assertIn("NativeMinibatchSampler", source)
        self.assertIn("PairScorer", source)
        self.assertIn("TIE_BREAK_POLICY", source)
        self.assertIn("score descending", serve.TIE_BREAK_POLICY)
        self.assertIn("movie_id ascending", serve.TIE_BREAK_POLICY)
        self.assertNotIn("torch_geometric.loader", source)
        self.assertNotIn("NeighborLoader(", source)
        self.assertNotIn("ClusterLoader(", source)
        self.assertNotIn("from torch_geometric.loader", source)
        self.assertNotIn("ensure_movielens_100k_processed", source)
        self.assertNotIn("download_movielens_100k", source)
        self.assertNotIn("SAGEREC_LIVE_MOVIELENS", source)
        script = _SCRIPT.read_text(encoding="utf-8")
        self.assertIn("serve.main", script)
        self.assertNotIn("ensure_movielens_100k_processed", script)

    def test_does_not_touch_stored_100k_quality_files(self) -> None:
        source = Path(serve.__file__).read_text(encoding="utf-8")
        self.assertNotIn("graphsage_movielens_100k.json", source)
        self.assertNotIn("multiseed_gnn_vs_mf_movielens_100k", source)
        self.assertNotIn("gnn_vs_mf_movielens_100k", source)
        self.assertTrue(_PHASE5_RESULT.is_file())
        self.assertTrue(_PHASE7_RESULT.is_file())


class RankingPolicyTests(unittest.TestCase):
    def test_rank_is_score_desc_then_movie_id_asc(self) -> None:
        scorer = DictScorer(
            {
                (0, 5): 1.0,
                (0, 2): 1.0,
                (0, 9): 0.5,
                (0, 1): 3.0,
            }
        )
        rows = serve.rank_candidates(scorer, 0, [5, 9, 1, 2], k=10)
        self.assertEqual([row.movie_id for row in rows], [1, 2, 5, 9])
        self.assertEqual([row.rank for row in rows], [1, 2, 3, 4])
        self.assertEqual(rows[1].score, 1.0)
        self.assertEqual(rows[2].score, 1.0)
        self.assertLess(rows[1].movie_id, rows[2].movie_id)

    def test_top_k_truncates_and_scores_are_finite(self) -> None:
        scorer = DictScorer({(0, i): float(10 - i) for i in range(6)})
        rows = serve.rank_candidates(scorer, 0, [0, 1, 2, 3, 4, 5], k=3)
        self.assertEqual([row.movie_id for row in rows], [0, 1, 2])
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertTrue(math.isfinite(row.score))

    def test_nonfinite_score_fails(self) -> None:
        scorer = DictScorer({(0, 1): math.nan, (0, 2): 1.0})
        with self.assertRaises(serve.ServeError) as ctx:
            serve.rank_candidates(scorer, 0, [1, 2], k=2)
        self.assertIn("non-finite", str(ctx.exception))

    def test_empty_or_duplicate_candidates_fail(self) -> None:
        scorer = DictScorer({})
        with self.assertRaises(serve.ServeError):
            serve.rank_candidates(scorer, 0, [], k=1)
        with self.assertRaises(serve.ServeError) as ctx:
            serve.rank_candidates(scorer, 0, [1, 1], k=1)
        self.assertIn("duplicate", str(ctx.exception).lower())
        with self.assertRaises(serve.ServeError):
            serve.rank_candidates(scorer, 0, [1], k=0)

    def test_parse_candidates_comma_and_whitespace(self) -> None:
        self.assertEqual(serve.parse_candidate_ids("1, 2,3"), [1, 2, 3])
        self.assertEqual(serve.parse_candidate_ids("4 5"), [4, 5])
        with self.assertRaises(serve.ServeError):
            serve.parse_candidate_ids("")
        with self.assertRaises(serve.ServeError):
            serve.parse_candidate_ids("1,1")
        with self.assertRaises(serve.ServeError):
            serve.parse_candidate_ids("1,x")


class CheckpointErrorTests(unittest.TestCase):
    def test_missing_checkpoint_is_actionable_nonzero(self) -> None:
        missing = Path("/tmp/sagerec-missing-checkpoint-does-not-exist.pt")
        self.assertFalse(missing.exists())
        code, stdout, stderr = _run_main(
            [
                "--checkpoint",
                str(missing),
                "--user",
                "0",
                "--candidates",
                "1,2",
            ]
        )
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("does not exist", stderr.lower())
        self.assertIn("error:", stderr.lower())
        self.assertIn("does not download", stderr.lower())

    def test_corrupt_bytes_and_wrong_schema_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            garbage = dest / "garbage.pt"
            garbage.write_bytes(b"not a torch checkpoint")
            code, stdout, stderr = _run_main(
                [
                    "--checkpoint",
                    str(garbage),
                    "--user",
                    "0",
                    "--candidates",
                    "1",
                ]
            )
            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertIn("corrupt", stderr.lower())
            self.assertIn(serve.CHECKPOINT_SCHEMA, stderr)

            empty = dest / "empty.pt"
            empty.write_bytes(b"")
            code, _, stderr = _run_main(
                ["--checkpoint", str(empty), "--user", "0", "--candidates", "1"]
            )
            self.assertEqual(code, 1)
            self.assertIn("corrupt", stderr.lower())

            wrong = dest / "wrong.pt"
            graphsage.torch.save({"hello": 1}, wrong)
            with self.assertRaises(serve.ServeError) as ctx:
                serve.load_checkpoint(wrong)
            self.assertIn("schema", str(ctx.exception).lower())

            bad_version = dest / "version.pt"
            graphsage.torch.save(
                {
                    "schema": serve.CHECKPOINT_SCHEMA,
                    "schema_version": 99,
                    "model": serve.MODEL_GRAPHSAGE,
                },
                bad_version,
            )
            with self.assertRaises(serve.ServeError) as ctx:
                serve.load_checkpoint(bad_version)
            self.assertIn("schema_version", str(ctx.exception))


class GraphSAGEServeTests(unittest.TestCase):
    def test_roundtrip_is_deterministic_with_finite_scores_and_native_sampler(self) -> None:
        split = _synthetic_split()
        config = _tiny_config(seed=7)
        train_pairs = split.train_positive_pairs()
        model = graphsage.train_graphsage(split, config)
        self.assertIsInstance(model, PairScorer)
        candidates = [0, 3, 5, 8, 9, 11, 14]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graphsage.pt"
            serve.save_checkpoint(path, model, train_pairs=train_pairs)
            payload = graphsage.torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["schema"], serve.CHECKPOINT_SCHEMA)
            self.assertEqual(payload["schema_version"], serve.CHECKPOINT_SCHEMA_VERSION)
            self.assertEqual(payload["model"], serve.MODEL_GRAPHSAGE)
            self.assertIn("hyperparams", payload)
            self.assertIn("state_dict", payload)
            self.assertIn("train_pairs", payload)

            with mock.patch.object(
                minibatch,
                "call_native_sample_neighbors",
                wraps=minibatch.call_native_sample_neighbors,
            ) as native:
                with mock.patch.object(
                    reference, "sample_neighbors", wraps=reference.sample_neighbors
                ) as ref:
                    first = serve.recommend_from_checkpoint(
                        path, 0, candidates, k=5
                    )
                    second = serve.recommend_from_checkpoint(
                        path, 0, candidates, k=5
                    )
            ref.assert_not_called()
            self.assertGreaterEqual(native.call_count, 1)
            self.assertIsInstance(native.call_args_list[0].args[0], graph_sampler.BipartiteCSR)

        self.assertEqual(len(first), 5)
        self.assertEqual(
            [(row.rank, row.movie_id, row.score) for row in first],
            [(row.rank, row.movie_id, row.score) for row in second],
        )
        scores = [row.score for row in first]
        self.assertTrue(all(math.isfinite(score) for score in scores))
        self.assertEqual(scores, sorted(scores, reverse=True))
        for left, right in zip(first, first[1:]):
            if left.score == right.score:
                self.assertLess(left.movie_id, right.movie_id)
        self.assertEqual([row.rank for row in first], [1, 2, 3, 4, 5])

        live = serve.rank_candidates(model, 0, candidates, k=5)
        self.assertEqual(
            [row.movie_id for row in first],
            [row.movie_id for row in live],
        )
        np.testing.assert_allclose(
            [row.score for row in first],
            [row.score for row in live],
        )

    def test_processed_dir_rebuilds_csr_when_pairs_omitted(self) -> None:
        split = _synthetic_split()
        model = graphsage.train_graphsage(split, _tiny_config(seed=3))
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            ckpt = dest / "model.pt"
            serve.save_checkpoint(ckpt, model, train_pairs=split.train_positive_pairs())
            payload = graphsage.torch.load(ckpt, map_location="cpu", weights_only=False)
            del payload["train_pairs"]
            stripped = dest / "stripped.pt"
            graphsage.torch.save(payload, stripped)
            processed = dest / "processed"
            processed.mkdir()
            (processed / "train_pairs.json").write_text(
                json.dumps(split.train_positive_pairs()) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(serve.ServeError) as ctx:
                serve.load_checkpoint(stripped)
            self.assertIn("train_pairs", str(ctx.exception))
            self.assertIn("does not download", str(ctx.exception).lower())

            loaded = serve.load_checkpoint(stripped, processed_dir=processed)
            self.assertIsInstance(loaded, graphsage.GraphSAGERecommender)
            self.assertIsInstance(loaded.sampler, minibatch.NativeMinibatchSampler)
            rows = serve.rank_candidates(loaded, 1, [5, 6, 7, 8], k=2)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(math.isfinite(row.score) for row in rows))

    def test_cli_prints_ranked_tsv(self) -> None:
        split = _synthetic_split()
        model = graphsage.train_graphsage(split, _tiny_config(seed=11))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            serve.save_checkpoint(path, model)
            code, stdout, stderr = _run_main(
                [
                    "--checkpoint",
                    str(path),
                    "--user",
                    "0",
                    "--candidates",
                    "0,3,8,11",
                    "--k",
                    "3",
                ]
            )
        self.assertEqual(code, 0, msg=stderr)
        self.assertEqual(stderr, "")
        parsed = _parse_tsv(stdout)
        self.assertEqual(len(parsed), 3)
        self.assertEqual([rank for rank, _, __ in parsed], [1, 2, 3])
        scores = [score for _, __, score in parsed]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(math.isfinite(score) for score in scores))

    def test_thin_script_subprocess_matches_module(self) -> None:
        split = _synthetic_split()
        model = graphsage.train_graphsage(split, _tiny_config(seed=5))
        sampler_dir = Path(graph_sampler.__file__).resolve().parent
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(sampler_dir), str(_PYTHON_DIR), env.get("PYTHONPATH", "")]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            serve.save_checkpoint(path, model, train_pairs=split.train_positive_pairs())
            argv = [
                "--checkpoint",
                str(path),
                "--user",
                "1",
                "--candidates",
                "5,6,7,8",
                "--k",
                "2",
            ]
            module_code, module_out, module_err = _run_main(argv)
            completed = subprocess.run(
                [sys.executable, str(_SCRIPT), *argv],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
        self.assertEqual(module_code, 0, msg=module_err)
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertEqual(completed.stdout, module_out)
        self.assertEqual(_parse_tsv(completed.stdout)[0][0], 1)

    def test_out_of_range_user_fails(self) -> None:
        split = _synthetic_split()
        model = graphsage.train_graphsage(split, _tiny_config(seed=2))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            serve.save_checkpoint(path, model)
            code, stdout, stderr = _run_main(
                [
                    "--checkpoint",
                    str(path),
                    "--user",
                    "99",
                    "--candidates",
                    "0",
                ]
            )
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("user_ids", stderr)


class MFServeTests(unittest.TestCase):
    def test_pairscorer_mf_checkpoint_roundtrip(self) -> None:
        split = _synthetic_split()
        model = baseline.train_implicit_mf(
            split,
            baseline.MFConfig(n_factors=4, n_epochs=1, seed=7, n_negatives=1),
        )
        self.assertIsInstance(model, PairScorer)
        candidates = [0, 1, 2, 3, 4]
        expected = serve.rank_candidates(model, 0, candidates, k=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mf.pt"
            serve.save_checkpoint(path, model)
            loaded = serve.load_checkpoint(path)
            ranked = serve.rank_candidates(loaded, 0, candidates, k=3)
        self.assertEqual(
            [row.movie_id for row in ranked],
            [row.movie_id for row in expected],
        )
        np.testing.assert_allclose(
            [row.score for row in ranked],
            [row.score for row in expected],
        )


if __name__ == "__main__":
    unittest.main()
