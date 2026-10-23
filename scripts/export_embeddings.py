#!/usr/bin/env python3
"""Thin CLI: export checkpoint embeddings to a serving artifact (ADR-009)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_export as export


def main() -> int:
    return export.main()


if __name__ == "__main__":
    raise SystemExit(main())
