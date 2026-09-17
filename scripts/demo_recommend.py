#!/usr/bin/env python3
"""Thin CLI: load a SageRec checkpoint, score candidates, print top-K (ADR-008)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_serve as serve


def main() -> int:
    return serve.main()


if __name__ == "__main__":
    raise SystemExit(main())
