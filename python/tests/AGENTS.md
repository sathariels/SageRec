# Python Binding Test Guide

Tests here import the compiled `graph_sampler` extension and exercise a tiny
synthetic graph plus in-memory `u.data` strings. They must stay deterministic
and free of MovieLens files. Reference-sampler parity tests also import
`sagerec_reference_sampler` from `python/` (the test file inserts that
directory on `sys.path` so `PYTHONPATH=build` still works).

- Fail with a nonzero exit (`python3 -m unittest` or CTest).
- Cover construction, seeded sampling, isolated/`k = 0` behavior, and invalid IDs.
- Cover `parse_movielens_100k` happy-path mappings, `local_pairs()` into
  `BipartiteCSR`, and at least one malformed-input `GraphError`.
- Cover native-vs-reference `sample_neighbors` equality on synthetic graphs
  for empty, full-neighborhood, subset, reproducibility, and invalid-input cases.
- Do not add training, download, baseline, or timing-benchmark tests in this
  directory until those phases are explicitly opened.
- Keep type hints on test helpers that form part of a public-looking fixture.
