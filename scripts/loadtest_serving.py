#!/usr/bin/env python3
"""Thin CLI: measure GET /recommend latency and write results/ (ADR-010).

Records retrieval-only and retrieval-plus-re-rank (protocol serving-latency-v2).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_loadtest as loadtest


def main() -> int:
    return loadtest.main()


if __name__ == "__main__":
    raise SystemExit(main())
