# Python System Agent Guide

## Responsibility

Own native-extension stubs, binding tests, the reference sampler,
MovieLens 100K split/prep (ADR-003), official 100K download and on-disk
`processed/` writes, the implicit MF baseline (ADR-004), the shared
ranking evaluator, the native-backed mini-batch neighborhood helper,
GraphSAGE training on that harness, and the Phase 5 GNN-versus-MF
comparison writer. Later: benchmark coordination and remaining result
serialization.

## Current slice

- Public type hints live in `graph_sampler.pyi`.
- Binding tests import the compiled `graph_sampler` module.
- The module exposes `BipartiteCSR` and in-memory `parse_movielens_100k`.
- `sagerec_reference_sampler.py` is a naive Python neighbor sampler for
  correctness comparison. It consumes CSR `offsets`/`neighbors` views and
  matches the native ADR-005 `sample_neighbors` contract (Fisher–Yates
  prefix, `std::mt19937_64`, unbiased `uniform_below`). It does not build
  graphs, apply splits, or ingest MovieLens files.
- `sagerec_prep.py` assigns ADR-003 leave-one-out splits in memory and
  emits train-only pairs plus a manifest dict. It does not download data
  or construct a CSR.
- `sagerec_download.py` fetches the official GroupLens 100K zip, verifies
  the published archive MD5, and extracts `u.data`. No 1M.
- `sagerec_dataset.py` reads `u.data` from a path or bytes, calls
  `graph_sampler.parse_movielens_100k` and `sagerec_prep`, and writes
  `data/processed/` artifacts plus a filled manifest.
  `ensure_movielens_100k_processed` reuses on-disk artifacts or
  downloads/preps 100K when they are missing.
- `sagerec_scoring.py` defines the `PairScorer` protocol used by the
  baseline and GraphSAGE. Metric logic must not live in the model.
- `sagerec_negatives.py` draws training negatives that do not overlap
  known positives in the supplied scope.
- `sagerec_baseline.py` is seeded implicit-feedback matrix factorization
  (NumPy logistic SGD). It trains on `train_positive_pairs()` only.
- `sagerec_metrics.py` is the shared ranking evaluator: per-user then
  macro-averaged Recall@10 and NDCG@10, ADR-003 eligibility, and candidate
  filtering.
- `sagerec_minibatch.py` is the Phase 4 harness: train-only
  `BipartiteCSR` from local pairs and seeded multi-hop expansion via
  native `graph_sampler.sample_neighbors` (ADR-005). GraphSAGE training
  must call this helper; do not fall back to a PyG NeighborLoader.
- `sagerec_graphsage.py` is the GraphSAGE training slice: PyTorch
  mean-aggregation layers, `PairScorer` scoring, and a seeded trainer
  that expands neighborhoods through `sagerec_minibatch`. CPU-only
  PyTorch is pinned in `requirements-train.txt`. PyG remains the intended
  production stack; this slice does not install or call PyG (no
  NeighborLoader / SAGEConv). `movielens_100k_config` is the modest
  CPU-friendly 100K hyperparameter set (seed 7). Tiny synthetic metrics
  are protocol smoke and must stay separate from stored 100K numbers.
- `sagerec_compare.py` writes the Phase 5 MF-versus-GraphSAGE JSON,
  markdown table, and SVG chart from stored result files. It does not
  invent metrics.
- Do not implement node2vec, timing charts, MovieLens 1M, or GCN.

## Boundaries

- The selected GNN is GraphSAGE and will use PyTorch and PyTorch Geometric.
- Training neighborhood expansion must call `graph_sampler`; do not quietly fall
  back to a PyG sampler in primary experiments.
- A naive Python sampler exists only for reference, correctness comparison, and timing.
- Separate modules for data, sampler adapter, models, trainer, evaluator, benchmark,
  and CLI entry points.
- Depend on a narrow sampler protocol so unit tests can inject a deterministic fake.
- Baseline and GNN evaluation must call `sagerec_metrics.evaluate_ranking`.
- NumPy is required for the MF baseline. GraphSAGE training requires
  CPU PyTorch (`python/requirements-train.txt`). Do not add PyTorch to
  the C++ graph core. Do not use a PyG neighbor sampler for primary
  experiments.

## Correctness

- Apply the split before constructing the training graph or fitting MF / GraphSAGE.
- Exclude known positive pairs from negatives and ranked candidates as specified.
- Keep GNN and baseline evaluation paths identical.
- Seed Python, NumPy, PyTorch, data-loader workers, and native sampling explicitly.
- Record nondeterministic backend settings and device information.
- Validate tensor shapes, node type/range, and mapping compatibility early.
- Do not claim MovieLens 100K quality from synthetic unittest metrics.

## Tests

- Binding construction, exceptions, lifetime, seeded smoke sampling, and
  in-memory MovieLens 100K parser tests (tiny strings, CSR handoff, GraphError).
- Reference-vs-native `sample_neighbors` parity on synthetic graphs
  (`k = 0`, full neighborhood, subset reproducibility, invalid inputs).
- ADR-003 leave-one-out leakage tests on tiny synthetic interactions:
  held-out positives absent from the train-pair/CSR edge set, eligibility,
  timestamp order and tie-break, cold-start assignment, and determinism.
- Phase 2 download/prep fixture tests: checksum mismatch, bad zip layout,
  path/bytes ingestion, filled manifest, train-only pair writes. Do not
  require a live download in default CI.
- Phase 3: negative-sample validity, Recall@10 / NDCG@10 unit cases,
  eligibility/filtering, MF train-edge leakage, and a tiny seeded e2e smoke
  with reproducible metrics in `[0, 1]`.
- Phase 4 mini-batch: compiled `graph_sampler` required (actionable
  ImportError if missing or not a `.so`); native `sample_neighbors` is
  actually called; reference sampler is not used; seed reproducibility;
  empty / `k = 0` / full-neighborhood; train-only CSR excludes held-out
  positives.
- Phase 4 GraphSAGE training: native `sample_neighbors` is actually
  called during neighborhood expansion (spy/assert); `PairScorer` feeds
  `evaluate_ranking`; negatives exclude train positives; held-out edges
  stay out of the train CSR; tiny seeded synthetic smoke metrics are
  finite in `[0, 1]` and reproducible. Label those metrics as protocol
  smoke, not a MovieLens 100K result.
- Phase 5 comparison writer: schema/protocol-match tests on fixture
  JSON; GraphSAGE 100K wiring still requires
  `NativeMinibatchSampler` and spies on native sampling. Do not add a
  live 100K quality-table job to default CI.
