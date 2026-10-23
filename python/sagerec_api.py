"""FastAPI serving layer for a precomputed SageRec artifact (ADR-009, ADR-010).

Routes:

- ``GET /healthz``
- ``GET /recommend?user_id=&k=&rerank=&n=``

``item_id`` in each row is the zero-based movie id. Already-seen train
positives are excluded. Unknown users return HTTP 404. Invalid query
parameters return HTTP 422. Top-K is exact NumPy (``sagerec_serving``);
FAISS is not required.

``rerank`` defaults to false, which is the ADR-009 retrieval path. When
``rerank=true`` and ``SAGEREC_RERANK_ARTIFACT`` (or ``rerank_path``) is
loaded, stage 1 returns ``n`` candidates and the exported MLP re-ranks them
to ``k``. The service does not train, sample, or download on the request.

This module does not import PyTorch, PyG, or ``graph_sampler``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query

import sagerec_rerank as rerank
import sagerec_serving as serving

_RERANK_MISSING = (
    "rerank=true requires a loaded sagerec_rerank_artifact. Set "
    "SAGEREC_RERANK_ARTIFACT or pass rerank_path to create_app. The service "
    "does not train a re-ranker on the request path."
)


def create_app(
    artifact_path: str | Path | None = None,
    rerank_path: str | Path | None = None,
) -> FastAPI:
    """Build an app bound to one serving artifact and an optional re-ranker.

    Uvicorn's factory calls this with no arguments and reads
    ``SAGEREC_SERVING_ARTIFACT`` plus optional ``SAGEREC_RERANK_ARTIFACT``.
    Tests pass ``artifact_path`` directly. An explicit artifact path does
    not inherit an ambient re-rank env var; pass ``rerank_path`` in tests.
    """
    factory_mode = artifact_path is None and rerank_path is None
    if artifact_path is None:
        artifact_path = os.environ.get("SAGEREC_SERVING_ARTIFACT")
    if not artifact_path:
        raise serving.ServingError(
            "SAGEREC_SERVING_ARTIFACT is not set. Point it at a "
            f"{serving.SERVING_SCHEMA} .npz, or pass artifact_path to create_app. "
            "The service does not download MovieLens or train a model."
        )
    index = serving.load_serving_artifact(artifact_path)
    if rerank_path is None and factory_mode:
        raw_rerank = os.environ.get("SAGEREC_RERANK_ARTIFACT", "").strip()
        rerank_path = raw_rerank or None
    bundle: rerank.RerankBundle | None = None
    if rerank_path:
        bundle = rerank.load_rerank_artifact(rerank_path)
        rerank.assert_rerank_compatible(
            bundle,
            num_users=index.num_users,
            num_items=index.num_items,
            stage1_model=str(index.metadata["model"]),
        )
    app = FastAPI(
        title="SageRec",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "schema": serving.SERVING_SCHEMA,
            "schema_version": serving.SERVING_SCHEMA_VERSION,
            "model": index.metadata["model"],
            "num_users": index.num_users,
            "num_items": index.num_items,
            "embedding_dim": index.embedding_dim,
            "rerank_loaded": bundle is not None,
        }

    @app.get("/recommend")
    def recommend(
        user_id: int = Query(
            ...,
            description="Zero-based user id. Unknown ids return 404.",
        ),
        k: int = Query(
            serving.DEFAULT_K,
            ge=1,
            description="Number of unseen items to return (default 10).",
        ),
        rerank_enabled: bool = Query(
            False,
            alias="rerank",
            description=(
                "When true, re-rank the stage-1 pool with the loaded MLP. "
                "Default false keeps ADR-009 retrieval order."
            ),
        ),
        n: int | None = Query(
            None,
            ge=1,
            description=(
                "Stage-1 candidate count used only when rerank=true. "
                "Defaults to 200, or the artifact's stage1_n when that is set."
            ),
        ),
    ) -> dict[str, Any]:
        if not rerank_enabled:
            try:
                rows = serving.recommend_topk(index, user_id, k)
            except serving.UnknownUserError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return {
                "user_id": user_id,
                "k": k,
                "rerank": False,
                "recommendations": [
                    {
                        "item_id": row.item_id,
                        "title": row.title,
                        "score": row.score,
                    }
                    for row in rows
                ],
            }
        if bundle is None:
            raise HTTPException(status_code=422, detail=_RERANK_MISSING)
        if n is None:
            stage1_n = int(bundle.metadata.get("stage1_n", rerank.DEFAULT_STAGE1_N))
        else:
            stage1_n = int(n)
        if stage1_n < k:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"n must be >= k when rerank=true so stage 1 can fill the "
                    f"re-rank pool, got n={stage1_n} k={k}"
                ),
            )
        try:
            stage1 = serving.recommend_topk(index, user_id, stage1_n)
            ordered = rerank.order_candidates(
                bundle,
                user_id,
                [row.item_id for row in stage1],
                [row.score for row in stage1],
            )
        except serving.UnknownUserError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except rerank.RerankError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        retrieval_by_id = {row.item_id: row.score for row in stage1}
        return {
            "user_id": user_id,
            "k": k,
            "rerank": True,
            "stage1_n": stage1_n,
            "recommendations": [
                {
                    "item_id": item_id,
                    "title": index.item_titles[item_id],
                    "score": score,
                    "retrieval_score": retrieval_by_id[item_id],
                }
                for item_id, score in ordered[:k]
            ],
        }

    return app
