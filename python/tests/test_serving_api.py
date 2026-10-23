"""Phase 9 ADR-009 HTTP serving tests (pytest + TestClient).

Importing this module must stay free of FastAPI, PyTorch, and
``graph_sampler`` so ``unittest discover`` can load it before those
packages are required. Each test imports what it needs.

Synthetic checkpoints and temp-dir artifacts only. These tests must not
download MovieLens or rewrite stored 100K quality files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PYTHON_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PYTHON_DIR.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_serving as serving

_LATENCY_JSON = _REPO_ROOT / "results" / "serving_latency.json"
_LATENCY_MD = _REPO_ROOT / "results" / "serving_latency.md"


def _fixture_index() -> serving.ServingIndex:
    """Hand-built catalog with a known tie and one seen item."""
    user_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    item_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.0],
            [2.0, 4.0],
        ],
        dtype=np.float64,
    )
    return serving.make_serving_index(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        seen_pairs=[(0, 3)],
        item_titles=("Alpha", "Beta", "Gamma", "Delta"),
        model="fixture",
        extra_metadata={"catalog_kind": "fixture", "seed": 0},
    )


def _client_for(index: serving.ServingIndex, tmp_path: Path):
    from fastapi.testclient import TestClient

    import sagerec_api

    artifact = tmp_path / "embeddings.npz"
    serving.save_serving_artifact(artifact, index)
    return TestClient(sagerec_api.create_app(artifact))


def test_topk_excludes_seen_items_and_breaks_score_ties() -> None:
    index = _fixture_index()
    rows = serving.recommend_topk(index, 0, k=10)
    assert [row.item_id for row in rows] == [0, 1, 2]
    assert [row.title for row in rows] == ["Alpha", "Beta", "Gamma"]
    assert rows[0].score == 1.0
    assert rows[1].score == 1.0
    assert rows[0].item_id < rows[1].item_id
    assert 3 not in [row.item_id for row in rows]
    top2 = serving.recommend_topk(index, 0, k=2)
    assert [row.item_id for row in top2] == [0, 1]
    user1 = serving.recommend_topk(index, 1, k=1)
    assert user1[0].item_id == 3
    assert user1[0].title == "Delta"
    assert user1[0].score == 4.0


def test_unknown_user_and_invalid_k_fail_clearly() -> None:
    index = _fixture_index()
    try:
        serving.recommend_topk(index, 99, k=1)
    except serving.UnknownUserError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected UnknownUserError")
    assert "unknown user_id 99" in message
    assert "[0, 2)" in message
    try:
        serving.recommend_topk(index, -1, k=1)
    except serving.UnknownUserError:
        pass
    else:
        raise AssertionError("expected UnknownUserError for a negative id")
    try:
        serving.recommend_topk(index, 0, k=0)
    except serving.ServingError as exc:
        assert "k must be >= 1" in str(exc)
    else:
        raise AssertionError("expected ServingError for k=0")


def test_artifact_roundtrip_preserves_ranking(tmp_path: Path) -> None:
    index = _fixture_index()
    path = tmp_path / "embeddings.npz"
    serving.save_serving_artifact(path, index)
    loaded = serving.load_serving_artifact(path)
    assert loaded.metadata["schema"] == serving.SERVING_SCHEMA
    assert loaded.metadata["schema_version"] == serving.SERVING_SCHEMA_VERSION
    assert loaded.num_users == 2
    assert loaded.num_items == 4
    original = [(row.item_id, row.score) for row in serving.recommend_topk(index, 0, k=3)]
    restored = [(row.item_id, row.score) for row in serving.recommend_topk(loaded, 0, k=3)]
    assert original == restored


def test_missing_artifact_is_actionable(tmp_path: Path) -> None:
    missing = tmp_path / "nope.npz"
    try:
        serving.load_serving_artifact(missing)
    except serving.ServingError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected ServingError")
    assert "does not exist" in message
    assert "export_embeddings" in message


def test_healthz_and_recommend_contract(tmp_path: Path) -> None:
    client = _client_for(_fixture_index(), tmp_path)
    health = client.get("/healthz")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["model"] == "fixture"
    assert body["num_users"] == 2
    assert body["num_items"] == 4
    assert body["embedding_dim"] == 2
    assert body["schema"] == serving.SERVING_SCHEMA
    assert body["rerank_loaded"] is False

    recommend = client.get("/recommend", params={"user_id": 0, "k": 2})
    assert recommend.status_code == 200
    payload = recommend.json()
    assert payload["user_id"] == 0
    assert payload["k"] == 2
    assert payload["rerank"] is False
    assert "stage1_n" not in payload
    assert [row["item_id"] for row in payload["recommendations"]] == [0, 1]
    assert "retrieval_score" not in payload["recommendations"][0]
    assert [row["title"] for row in payload["recommendations"]] == ["Alpha", "Beta"]
    assert payload["recommendations"][0]["score"] == 1.0
    assert payload["recommendations"][1]["score"] == 1.0
    assert all("item_id" in row and "title" in row and "score" in row for row in payload["recommendations"])

    default_k = client.get("/recommend", params={"user_id": 0})
    assert default_k.status_code == 200
    assert default_k.json()["k"] == serving.DEFAULT_K
    assert len(default_k.json()["recommendations"]) == 3


def test_unknown_user_is_404_and_bad_queries_are_422(tmp_path: Path) -> None:
    client = _client_for(_fixture_index(), tmp_path)
    missing_user = client.get("/recommend", params={"user_id": 99, "k": 1})
    assert missing_user.status_code == 404
    detail = missing_user.json()["detail"]
    assert isinstance(detail, str)
    assert "unknown user_id 99" in detail
    assert "[0, 2)" in detail

    negative = client.get("/recommend", params={"user_id": -1, "k": 1})
    assert negative.status_code == 404
    assert "unknown user_id -1" in negative.json()["detail"]

    omitted = client.get("/recommend", params={"k": 1})
    assert omitted.status_code == 422

    not_an_int = client.get("/recommend", params={"user_id": "abc", "k": 1})
    assert not_an_int.status_code == 422

    zero_k = client.get("/recommend", params={"user_id": 0, "k": 0})
    assert zero_k.status_code == 422

    negative_k = client.get("/recommend", params={"user_id": 0, "k": -3})
    assert negative_k.status_code == 422


def test_http_surface_is_healthz_and_recommend(tmp_path: Path) -> None:
    client = _client_for(_fixture_index(), tmp_path)
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_api_source_does_not_sample_or_download() -> None:
    api_source = (_PYTHON_DIR / "sagerec_api.py").read_text(encoding="utf-8")
    serving_source = (_PYTHON_DIR / "sagerec_serving.py").read_text(encoding="utf-8")
    rerank_source = (_PYTHON_DIR / "sagerec_rerank.py").read_text(encoding="utf-8")
    for source in (api_source, serving_source, rerank_source):
        assert "NeighborLoader" not in source
        assert "ClusterLoader" not in source
        assert "torch_geometric" not in source
        assert "ensure_movielens_100k_processed" not in source
        assert "download_movielens_100k" not in source
        assert "import torch" not in source
        assert "import graph_sampler" not in source
    assert "recommend_topk" in api_source
    assert "order_candidates" in api_source
    assert "SAGEREC_RERANK_ARTIFACT" in api_source
    assert "numpy" in serving_source
    assert "FAISS is not required" in serving_source


def test_uitem_and_json_titles(tmp_path: Path) -> None:
    import sagerec_export as export

    uitem = tmp_path / "u.item"
    uitem.write_text(
        "1|Toy Story (1995)|01-Jan-1995||\n"
        "2|GoldenEye (1995)|01-Jan-1995||\n"
        "5|Copycat (1995)|01-Jan-1995||\n",
        encoding="latin-1",
    )
    titles = export.load_item_titles(uitem, 3, movie_source_ids=[5, 1, 2])
    assert titles == ["Copycat (1995)", "Toy Story (1995)", "GoldenEye (1995)"]

    listed = tmp_path / "titles.json"
    listed.write_text(json.dumps(["One", "Two", "Three"]), encoding="utf-8")
    assert export.load_item_titles(listed, 3) == ["One", "Two", "Three"]

    try:
        export.load_item_titles(uitem, 3, movie_source_ids=None)
    except export.ExportError as exc:
        assert "processed-dir" in str(exc)
    else:
        raise AssertionError("expected ExportError when source ids are not 1..N")


def _tiny_split():
    import sagerec_prep as prep

    def row(user_id: int, movie_id: int, timestamp: int) -> prep.NormalizedInteraction:
        return prep.NormalizedInteraction(user_id, movie_id, timestamp, 1)

    interactions = [
        row(0, 0, 10),
        row(0, 1, 20),
        row(0, 2, 30),
        row(0, 3, 40),
        row(0, 4, 50),
        row(1, 5, 11),
        row(1, 6, 21),
        row(1, 7, 31),
        row(1, 8, 41),
        row(2, 0, 1),
        row(2, 9, 2),
    ]
    return prep.split_interactions(interactions, num_users=3, num_movies=20)


def test_graphsage_export_matches_score_pairs_and_calls_native_sampler(tmp_path: Path) -> None:
    from unittest import mock

    import graph_sampler
    import sagerec_export as export
    import sagerec_graphsage as graphsage
    import sagerec_minibatch as minibatch
    import sagerec_reference_sampler as reference
    import sagerec_serve as serve

    split = _tiny_split()
    config = graphsage.GraphSAGEConfig(
        embedding_dim=4,
        hidden_dim=4,
        n_layers=2,
        fanouts=(3, 2),
        n_epochs=1,
        batch_size=4,
        learning_rate=0.1,
        n_negatives=1,
        l2=1e-4,
        seed=7,
    )
    model = graphsage.train_graphsage(split, config)
    checkpoint = tmp_path / "graphsage.pt"
    serve.save_checkpoint(checkpoint, model, train_pairs=split.train_positive_pairs())
    artifact = tmp_path / "graphsage.npz"
    with mock.patch.object(
        minibatch,
        "call_native_sample_neighbors",
        wraps=minibatch.call_native_sample_neighbors,
    ) as native:
        with mock.patch.object(
            reference, "sample_neighbors", wraps=reference.sample_neighbors
        ) as ref:
            export.export_checkpoint(checkpoint, artifact)
    ref.assert_not_called()
    assert native.call_count >= 1
    assert isinstance(native.call_args_list[0].args[0], graph_sampler.BipartiteCSR)

    index = serving.load_serving_artifact(artifact)
    assert index.metadata["model"] == "graphsage"
    assert index.embedding_dim == 4
    train_pairs = set(split.train_positive_pairs())
    seen = set(int(movie_id) for movie_id in index.seen_item_ids(0))
    assert seen == {movie_id for user_id, movie_id in train_pairs if user_id == 0}
    held_out = split.positive_pair_set("validation") | split.positive_pair_set("test")
    assert (0, next(movie_id for user_id, movie_id in held_out if user_id == 0)) not in {
        (0, movie_id) for movie_id in seen
    }

    candidates = [movie_id for movie_id in range(index.num_items) if movie_id not in seen]
    live = model.score_pairs([0] * len(candidates), candidates)
    served = serving.recommend_topk(index, 0, k=len(candidates))
    served_by_id = {row.item_id: row.score for row in served}
    assert set(served_by_id) == set(candidates)
    np.testing.assert_allclose(
        [served_by_id[movie_id] for movie_id in candidates],
        np.asarray(live, dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    assert serve.TIE_BREAK_POLICY == serving.TIE_BREAK_POLICY
    assert serve.DEFAULT_K == serving.DEFAULT_K

    from fastapi.testclient import TestClient

    import sagerec_api

    client = TestClient(sagerec_api.create_app(artifact))
    response = client.get("/recommend", params={"user_id": 0, "k": 5})
    assert response.status_code == 200
    ids = [row["item_id"] for row in response.json()["recommendations"]]
    assert seen.isdisjoint(ids)
    assert len(ids) == 5


def test_mf_export_matches_score_pairs_and_requires_train_pairs(tmp_path: Path) -> None:
    import sagerec_baseline as baseline
    import sagerec_export as export
    import sagerec_serve as serve

    split = _tiny_split()
    config = baseline.MFConfig(
        n_factors=4,
        n_epochs=1,
        learning_rate=0.05,
        n_negatives=1,
        l2=0.01,
        seed=7,
        init_std=0.1,
    )
    model = baseline.train_implicit_mf(split, config)
    checkpoint = tmp_path / "mf.pt"
    serve.save_checkpoint(checkpoint, model)
    artifact = tmp_path / "mf.npz"
    try:
        export.export_checkpoint(checkpoint, artifact)
    except export.ExportError as exc:
        assert "train pairs" in str(exc).lower() or "train-pairs" in str(exc)
    else:
        raise AssertionError("MF export must require train pairs")

    pairs_path = tmp_path / "train_pairs.json"
    pairs_path.write_text(
        json.dumps(split.train_positive_pairs()),
        encoding="utf-8",
    )
    export.export_checkpoint(checkpoint, artifact, train_pairs_path=pairs_path)
    index = serving.load_serving_artifact(artifact)
    assert index.metadata["model"] == "implicit_mf"
    seen = {int(movie_id) for movie_id in index.seen_item_ids(0)}
    candidates = [movie_id for movie_id in range(index.num_items) if movie_id not in seen]
    live = np.asarray(model.score_pairs([0] * len(candidates), candidates), dtype=np.float64)
    served = serving.recommend_topk(index, 0, k=len(candidates))
    served_scores = [next(row.score for row in served if row.item_id == movie_id) for movie_id in candidates]
    np.testing.assert_allclose(served_scores, live, rtol=1e-8, atol=1e-8)

    from fastapi.testclient import TestClient

    import sagerec_api

    client = TestClient(sagerec_api.create_app(artifact))
    response = client.get("/recommend", params={"user_id": 0, "k": 3})
    assert response.status_code == 200
    assert seen.isdisjoint(row["item_id"] for row in response.json()["recommendations"])


def test_export_cli_missing_checkpoint_exits_nonzero(tmp_path: Path) -> None:
    import sagerec_export as export

    code = export.main(
        [
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--output",
            str(tmp_path / "out.npz"),
        ]
    )
    assert code == 1


def test_latency_percentiles_are_ordered() -> None:
    import sagerec_loadtest as loadtest

    stats = loadtest.latency_percentiles_ms([0.001, 0.002, 0.010, 0.020, 0.050])
    assert stats["percentile_method"] == "linear"
    assert stats["min"] <= stats["p50"] <= stats["p95"] <= stats["p99"] <= stats["max"]
    assert stats["min"] == 1.0
    assert stats["max"] == 50.0


def test_committed_serving_latency_matches_markdown() -> None:
    import sagerec_loadtest as loadtest

    assert _LATENCY_JSON.is_file()
    assert _LATENCY_MD.is_file()
    payload = json.loads(_LATENCY_JSON.read_text(encoding="utf-8"))
    for key in loadtest.REQUIRED_LATENCY_KEYS:
        assert key in payload
    assert payload["protocol_version"] == loadtest.PROTOCOL_VERSION
    assert payload["adr"] == "ADR-010"
    assert payload["retrieval_adr"] == "ADR-009"
    assert payload["endpoint"] == "GET /recommend"
    assert payload["n_errors"] == 0
    assert payload["n_requests"] == loadtest.DEFAULT_REQUESTS
    assert payload["catalog"]["kind"] == "synthetic"
    assert payload["catalog"]["num_users"] == loadtest.DEFAULT_NUM_USERS
    assert payload["catalog"]["num_items"] == loadtest.DEFAULT_NUM_ITEMS
    assert payload["catalog"]["embedding_dim"] == loadtest.DEFAULT_EMBEDDING_DIM
    assert payload["workload"]["concurrency"] == loadtest.DEFAULT_CONCURRENCY
    assert payload["workload"]["warmup_requests"] == loadtest.DEFAULT_WARMUP
    assert payload["workload"]["k"] == loadtest.DEFAULT_K
    assert payload["workload"]["stage1_n"] == loadtest.DEFAULT_STAGE1_N
    latency = payload["latency_ms"]
    assert latency["percentile_method"] == "linear"
    assert latency["min"] <= latency["p50"] <= latency["p95"] <= latency["p99"] <= latency["max"]
    assert payload["throughput_requests_per_s"] > 0
    assert payload["catalog"]["rerank_kind"] == "synthetic"
    modes = payload["modes"]
    assert list(modes) == list(loadtest.LATENCY_MODES)
    retrieval = modes["retrieval_only"]
    reranked = modes["retrieval_plus_rerank"]
    assert retrieval["rerank"] is False
    assert reranked["rerank"] is True
    assert reranked["stage1_n"] == loadtest.DEFAULT_STAGE1_N
    assert retrieval["n_errors"] == 0
    assert reranked["n_errors"] == 0
    assert retrieval["n_requests"] == loadtest.DEFAULT_REQUESTS
    assert reranked["n_requests"] == loadtest.DEFAULT_REQUESTS
    for mode in (retrieval, reranked):
        mode_latency = mode["latency_ms"]
        assert mode_latency["percentile_method"] == "linear"
        assert (
            mode_latency["min"]
            <= mode_latency["p50"]
            <= mode_latency["p95"]
            <= mode_latency["p99"]
            <= mode_latency["max"]
        )
        assert mode["throughput_requests_per_s"] > 0
    assert retrieval["latency_ms"] == payload["latency_ms"]
    assert isinstance(payload["cpu"]["model"], str) and payload["cpu"]["model"]
    assert isinstance(payload["cpu"]["logical_cpus"], int)
    assert payload["platform"]["system"]
    note = payload["note"]
    assert "not a ranking-quality" in note.lower() or "Not a ranking-quality" in note
    assert "synthetic" in note.lower()
    rendered = loadtest.render_serving_latency_markdown(payload)
    assert _LATENCY_MD.read_text(encoding="utf-8") == rendered
    assert "p50" in rendered
    assert "retrieval_only" in rendered
    assert "retrieval_plus_rerank" in rendered
    assert payload["cpu"]["model"] in rendered


def _popularity_bundle():
    import sagerec_rerank as rerank

    tables = rerank.FeatureTables(
        item_popularity=np.asarray([0.0, 0.0, 10.0, 0.0], dtype=np.float64),
        item_recency=np.zeros(4, dtype=np.float64),
        item_genres=np.zeros((4, 1), dtype=np.float64),
        user_genre_profile=np.zeros((2, 1), dtype=np.float64),
        genres_available=False,
    )
    return rerank.zero_linear_bundle(
        tables,
        stage1_model="fixture",
        popularity_weight=1.0,
    )


def _client_with_rerank(tmp_path: Path):
    from fastapi.testclient import TestClient

    import sagerec_api
    import sagerec_rerank as rerank

    artifact = tmp_path / "embeddings.npz"
    rerank_path = tmp_path / "rerank.npz"
    serving.save_serving_artifact(artifact, _fixture_index())
    rerank.save_rerank_artifact(rerank_path, _popularity_bundle())
    return TestClient(sagerec_api.create_app(artifact, rerank_path))


def test_rerank_false_keeps_retrieval_order_when_a_model_is_loaded(tmp_path: Path) -> None:
    client = _client_with_rerank(tmp_path)
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["rerank_loaded"] is True
    response = client.get("/recommend", params={"user_id": 0, "k": 2, "rerank": "false"})
    assert response.status_code == 200
    body = response.json()
    assert body["rerank"] is False
    assert [row["item_id"] for row in body["recommendations"]] == [0, 1]
    assert "retrieval_score" not in body["recommendations"][0]
    ignored_n = client.get(
        "/recommend",
        params={"user_id": 0, "k": 2, "rerank": "false", "n": 1},
    )
    assert ignored_n.status_code == 200
    assert [row["item_id"] for row in ignored_n.json()["recommendations"]] == [0, 1]


def test_rerank_true_reorders_the_stage1_pool_and_keeps_seen_items_out(tmp_path: Path) -> None:
    client = _client_with_rerank(tmp_path)
    response = client.get("/recommend", params={"user_id": 0, "k": 2, "rerank": "true", "n": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["rerank"] is True
    assert body["stage1_n"] == 3
    rows = body["recommendations"]
    assert [row["item_id"] for row in rows] == [2, 0]
    assert rows[0]["score"] == 10.0
    assert rows[0]["retrieval_score"] == 0.5
    assert rows[1]["retrieval_score"] == 1.0
    assert all(row["item_id"] != 3 for row in rows)

    narrow = client.get("/recommend", params={"user_id": 0, "k": 2, "rerank": "true", "n": 2})
    assert narrow.status_code == 200
    assert [row["item_id"] for row in narrow.json()["recommendations"]] == [0, 1]


def test_rerank_true_without_a_model_is_422_and_unknown_user_stays_404(tmp_path: Path) -> None:
    client = _client_for(_fixture_index(), tmp_path)
    missing = client.get("/recommend", params={"user_id": 0, "k": 1, "rerank": "true"})
    assert missing.status_code == 422
    detail = missing.json()["detail"]
    assert isinstance(detail, str)
    assert "sagerec_rerank_artifact" in detail
    assert "SAGEREC_RERANK_ARTIFACT" in detail

    loaded = _client_with_rerank(tmp_path)
    unknown = loaded.get("/recommend", params={"user_id": 99, "k": 1, "rerank": "true"})
    assert unknown.status_code == 404
    assert "unknown user_id 99" in unknown.json()["detail"]
    too_small = loaded.get("/recommend", params={"user_id": 0, "k": 2, "rerank": "true", "n": 1})
    assert too_small.status_code == 422
    assert "n must be >= k" in too_small.json()["detail"]
    bad_flag = loaded.get("/recommend", params={"user_id": 0, "k": 1, "rerank": "maybe"})
    assert bad_flag.status_code == 422


def test_mismatched_rerank_artifact_fails_at_startup(tmp_path: Path) -> None:
    import sagerec_api
    import sagerec_rerank as rerank

    artifact = tmp_path / "embeddings.npz"
    rerank_path = tmp_path / "rerank.npz"
    serving.save_serving_artifact(artifact, _fixture_index())
    bundle = _popularity_bundle()
    bundle.metadata["stage1_model"] = "other_model"
    rerank.save_rerank_artifact(rerank_path, bundle)
    try:
        sagerec_api.create_app(artifact, rerank_path)
    except rerank.RerankError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RerankError")
    assert "other_model" in message
    assert "fixture" in message
