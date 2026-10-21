"""FastAPI serving layer for a precomputed SageRec artifact (ADR-009).

Routes:

- ``GET /healthz``
- ``GET /recommend?user_id=&k=``

``item_id`` in each row is the zero-based movie id. Already-seen train
positives are excluded. Unknown users return HTTP 404. Invalid query
parameters return HTTP 422. Top-K is exact NumPy (``sagerec_serving``);
FAISS is not required.

This module does not import PyTorch, PyG, or ``graph_sampler``, and it does
not download MovieLens or train a model. Load a ``.npz`` produced by
``sagerec_export``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query

import sagerec_serving as serving


def create_app(artifact_path: str | Path | None = None) -> FastAPI:
    """Build an app bound to one serving artifact.

    Uvicorn's factory calls this with no arguments and reads
    ``SAGEREC_SERVING_ARTIFACT``. Tests pass ``artifact_path`` directly.
    """
    if artifact_path is None:
        artifact_path = os.environ.get("SAGEREC_SERVING_ARTIFACT")
    if not artifact_path:
        raise serving.ServingError(
            "SAGEREC_SERVING_ARTIFACT is not set. Point it at a "
            f"{serving.SERVING_SCHEMA} .npz, or pass artifact_path to create_app. "
            "The service does not download MovieLens or train a model."
        )
    index = serving.load_serving_artifact(artifact_path)
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
    ) -> dict[str, Any]:
        try:
            rows = serving.recommend_topk(index, user_id, k)
        except serving.UnknownUserError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "user_id": user_id,
            "k": k,
            "recommendations": [
                {
                    "item_id": row.item_id,
                    "title": row.title,
                    "score": row.score,
                }
                for row in rows
            ],
        }

    return app
