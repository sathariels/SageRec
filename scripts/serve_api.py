#!/usr/bin/env python3
"""Thin CLI: serve GET /recommend from a precomputed artifact (ADR-009/010)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Serve SageRec recommendations from a sagerec_serving_artifact "
            ".npz. Does not download MovieLens or train a model."
        )
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument(
        "--rerank",
        type=Path,
        default=None,
        help=(
            "Optional sagerec_rerank_artifact .npz. Requests stay retrieval-only "
            "until the client passes rerank=true."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 0 if code is None else int(code)
    if not args.artifact.is_file():
        print(
            f"error: serving artifact {args.artifact} does not exist. "
            "Export one with scripts/export_embeddings.py.",
            file=sys.stderr,
        )
        return 1
    os.environ["SAGEREC_SERVING_ARTIFACT"] = str(args.artifact.resolve())
    if args.rerank is not None:
        if not args.rerank.is_file():
            print(
                f"error: re-rank artifact {args.rerank} does not exist. "
                "Train one with scripts/run_rerank_eval.py --rerank-out.",
                file=sys.stderr,
            )
            return 1
        os.environ["SAGEREC_RERANK_ARTIFACT"] = str(args.rerank.resolve())
    try:
        import uvicorn
    except ImportError:
        print(
            "error: uvicorn is required to serve the API. "
            "Install python/requirements-serve.txt.",
            file=sys.stderr,
        )
        return 1
    uvicorn.run(
        "sagerec_api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
