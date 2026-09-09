#!/usr/bin/env python3
"""Thin CLI: ADR-003 on-disk prep from a local MovieLens 100K u.data file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_dataset as dataset
import sagerec_download as download


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MovieLens 100K processed artifacts (ADR-003)."
    )
    parser.add_argument(
        "--udata",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "ml-100k" / "u.data",
        help="Path to u.data (default: <repo>/data/raw/ml-100k/u.data)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "processed",
        help="Output directory (default: <repo>/data/processed)",
    )
    parser.add_argument(
        "--source-url",
        default=download.MOVIELENS_100K_URL,
        help="Provenance URL recorded in the manifest",
    )
    parser.add_argument(
        "--checksum",
        default=None,
        help="Override manifest checksum (default: sha256 of the ratings file)",
    )
    parser.add_argument(
        "--license",
        default=download.MOVIELENS_100K_LICENSE,
        help="License string recorded in the manifest",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(f"udata={args.udata.resolve()}")
    print(f"processed_dir={args.processed_dir.resolve()}")
    print(f"source_url={args.source_url}")
    prepared = dataset.prepare_movielens_100k(
        args.udata,
        args.processed_dir,
        source_url=args.source_url,
        checksum=args.checksum,
        license=args.license,
    )
    print(json.dumps(prepared.manifest, indent=2))
    print(f"wrote {prepared.manifest_path}")
    print(f"wrote {prepared.assignments_path}")
    print(f"wrote {prepared.train_pairs_path}")
    print(f"wrote {prepared.mappings_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (dataset.MovieLensPrepError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
