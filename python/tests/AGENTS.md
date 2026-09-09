# Python Binding Test Guide

Tests here import the compiled `graph_sampler` extension and exercise a tiny
synthetic graph plus in-memory `u.data` strings. They must stay deterministic.
Reference-sampler, split-prep, and download/prep tests also import
`sagerec_*` modules from `python/` (those test files insert that directory on
`sys.path` so `PYTHONPATH=build` still works). Prefer `PYTHONPATH=build:python`.

- Fail with a nonzero exit (`python3 -m unittest` or CTest).
- Cover construction, seeded sampling, isolated/`k = 0` behavior, and invalid IDs.
- Cover `parse_movielens_100k` happy-path mappings, `local_pairs()` into
  `BipartiteCSR`, and at least one malformed-input `GraphError`.
- Cover native-vs-reference `sample_neighbors` equality on synthetic graphs
  for empty, full-neighborhood, subset, reproducibility, and invalid-input cases.
- Cover ADR-003 leave-one-out assignment, cold-start eligibility, timestamp
  tie-break, determinism, manifest fields, and leakage (validation/test
  positives absent from train pairs and from a train-only `BipartiteCSR`).
- Cover Phase 2 download/prep with tiny fixtures and temp dirs: path/bytes
  ingestion, manifest writing, checksum mismatch, and bad zip layout. A live
  download test may exist only if it is skipped without
  `SAGEREC_LIVE_MOVIELENS=1`.
- Cover Phase 3 MF baseline and shared ranking metrics on tiny synthetic
  data: Recall@10 / NDCG@10 bounds, macro averaging, ADR-003 eligibility,
  candidate filtering, negative-sample validity, train-edge leakage, and
  seeded reproducibility. Label those metrics as verification only.
- Do not add GraphSAGE training or timing-benchmark tests in this directory
  until those phases are explicitly opened.
- Keep type hints on test helpers that form part of a public-looking fixture.
