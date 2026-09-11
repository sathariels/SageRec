# Python Binding Test Guide

Tests here import the compiled `graph_sampler` extension and exercise a tiny
synthetic graph plus in-memory `u.data` strings. They must stay deterministic.
Reference-sampler, split-prep, download/prep, MF, and mini-batch tests also
import `sagerec_*` modules from `python/` (those test files insert that
directory on `sys.path` so `PYTHONPATH=build` still works). Prefer
`PYTHONPATH=build:python`.

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
- Cover Phase 4 mini-batch native sampling on a tiny synthetic graph:
  compiled `graph_sampler` import, failure when the extension is missing
  or bypassed, seed reproducibility, empty / `k = 0` / full-neighborhood,
  multi-hop frontier expansion, and train-only CSR leakage.
- Cover Phase 4 GraphSAGE training on tiny synthetic ADR-003 data:
  native `sample_neighbors` is invoked (spy), reference/PyG samplers are
  not used, `PairScorer` + `evaluate_ranking` smoke metrics are finite
  in `[0, 1]` and reproducible, and held-out edges stay out of the train
  CSR. Label those metrics as protocol smoke, not MovieLens 100K.
- Do not add timing-benchmark tests or a GNN-versus-baseline quality
  table in this directory until those slices are explicitly opened.
- Keep type hints on test helpers that form part of a public-looking fixture.
