# Python Binding Test Guide

Tests here import the compiled `graph_sampler` extension and exercise a tiny
synthetic graph plus in-memory `u.data` strings. They must stay deterministic
and free of MovieLens files. Reference-sampler and split-prep tests also
import `sagerec_reference_sampler` / `sagerec_prep` from `python/` (those
test files insert that directory on `sys.path` so `PYTHONPATH=build` still
works). Prefer `PYTHONPATH=build:python`.

- Fail with a nonzero exit (`python3 -m unittest` or CTest).
- Cover construction, seeded sampling, isolated/`k = 0` behavior, and invalid IDs.
- Cover `parse_movielens_100k` happy-path mappings, `local_pairs()` into
  `BipartiteCSR`, and at least one malformed-input `GraphError`.
- Cover native-vs-reference `sample_neighbors` equality on synthetic graphs
  for empty, full-neighborhood, subset, reproducibility, and invalid-input cases.
- Cover ADR-003 leave-one-out assignment, cold-start eligibility, timestamp
  tie-break, determinism, manifest fields, and leakage (validation/test
  positives absent from train pairs and from a train-only `BipartiteCSR`).
- Do not add training, download, baseline, or timing-benchmark tests in this
  directory until those phases are explicitly opened.
- Keep type hints on test helpers that form part of a public-looking fixture.
