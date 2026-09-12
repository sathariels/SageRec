#!/usr/bin/env python3
"""Thin CLI: write MF vs GraphSAGE comparison artifacts from stored results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_compare as compare


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare stored MovieLens 100K MF and GraphSAGE ranking JSONs. "
            "Does not train models or invent metrics."
        )
    )
    parser.add_argument(
        "--mf-result",
        type=Path,
        default=_REPO_ROOT / "results" / "mf_movielens_100k.json",
    )
    parser.add_argument(
        "--graphsage-result",
        type=Path,
        default=_REPO_ROOT / "results" / "graphsage_movielens_100k.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=_REPO_ROOT / "results" / "gnn_vs_mf_movielens_100k.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=_REPO_ROOT / "results" / "gnn_vs_mf_movielens_100k.md",
    )
    parser.add_argument(
        "--chart-output",
        type=Path,
        default=_REPO_ROOT / "results" / "gnn_vs_mf_movielens_100k.svg",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifacts = compare.write_comparison_artifacts(
        args.mf_result,
        args.graphsage_result,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
        chart_path=args.chart_output,
    )
    print(json.dumps(artifacts.payload, indent=2))
    print(f"wrote {artifacts.json_path}")
    print(f"wrote {artifacts.markdown_path}")
    print(f"wrote {artifacts.chart_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except compare.ComparisonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
