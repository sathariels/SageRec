# Python Binding Test Guide

Tests here import the compiled `graph_sampler` extension and exercise a tiny
synthetic graph. They must stay deterministic and free of MovieLens files.

- Fail with a nonzero exit (`python3 -m unittest` or CTest).
- Cover construction, seeded sampling, isolated/`k = 0` behavior, and invalid IDs.
- Do not add training, download, baseline, or benchmark tests in this directory
  until those phases are explicitly opened.
- Keep type hints on test helpers that form part of a public-looking fixture.
