#!/usr/bin/env python3
"""Thin CLI: fetch the official MovieLens 100K zip and extract u.data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_download as download


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GroupLens MovieLens 100K and extract u.data."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "raw",
        help="Destination directory (default: <repo>/data/raw)",
    )
    parser.add_argument(
        "--url",
        default=download.MOVIELENS_100K_URL,
        help="Archive URL (100K only; default is the official GroupLens zip)",
    )
    parser.add_argument(
        "--expected-md5",
        default=download.MOVIELENS_100K_ARCHIVE_MD5,
        help="Archive MD5 hex digest (empty string skips verification)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    expected = args.expected_md5.strip() or None
    print(f"url={args.url}")
    print(f"expected_md5={expected}")
    print(f"raw_dir={args.raw_dir.resolve()}")
    result = download.download_movielens_100k(
        args.raw_dir,
        url=args.url,
        expected_md5=expected,
    )
    print(f"archive={result.archive_path}")
    print(f"udata={result.udata_path}")
    print(f"archive_md5={result.archive_md5}")
    print(f"udata_sha256={result.udata_sha256}")
    print(f"member={result.archive_member}")
    print(f"used_unverified_tls={result.used_unverified_tls}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except download.MovieLensDownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
