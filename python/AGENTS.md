# Python System Agent Guide

## Responsibility

Own native-extension stubs, binding tests, the reference sampler,
in-memory MovieLens 100K split/prep (ADR-003), the implicit MF baseline
(ADR-004), and the shared ranking evaluator. Later: download/on-disk
orchestration, GraphSAGE training, configuration, benchmark coordination,
and result serialization.

## Current slice

- Public type hints live in `graph_sampler.pyi`.
- Binding tests import the compiled `graph_sampler` module.
- The module exposes `BipartiteCSR` and in-memory `parse_movielens_100k`.
- `sagerec_reference_sampler.py` is a naive Python neighbor sampler for
  correctness comparison. It consumes CSR `offsets`/`neighbors` views and
  matches the native `sample_neighbors` contract (Fisher–Yates prefix,
  `std::mt19937_64`, unbiased `uniform_below`). It does not build graphs,
  apply splits, or ingest MovieLens files.
- `sagerec_prep.py` assigns ADR-003 leave-one-out splits in memory and
  emits train-only pairs plus a manifest dict. It does not download data,
  read dataset paths, or construct a CSR.
- `sagerec_scoring.py` defines the `PairScorer` protocol used by the
  baseline and (later) GraphSAGE. Metric logic must not live in the model.
- `sagerec_negatives.py` draws training negatives that do not overlap
  known positives in the supplied scope.
- `sagerec_baseline.py` is seeded implicit-feedback matrix factorization
  (NumPy logistic SGD). It trains on `train_positive_pairs()` only.
- `sagerec_metrics.py` is the shared ranking evaluator: per-user then
  macro-averaged Recall@10 and NDCG@10, ADR-003 eligibility, and candidate
  filtering.
- Do not implement MovieLens download, on-disk prep, GraphSAGE training,
  node2vec, timing charts, or remaining Phase 4–5 work until those phases
  are opened.
- Do not add MovieLens 1M or GCN modules.

## Boundaries

- The selected GNN is GraphSAGE and will use PyTorch and PyTorch Geometric.
- Training neighborhood expansion must call `graph_sampler`; do not quietly fall
  back to a PyG sampler in primary experiments.
- A naive Python sampler exists only for reference, correctness comparison, and timing.
- Separate modules for data, sampler adapter, models, trainer, evaluator, benchmark,
  and CLI entry points.
- Depend on a narrow sampler protocol so unit tests can inject a deterministic fake.
- Baseline and GNN evaluation must call `sagerec_metrics.evaluate_ranking`.
- NumPy is required for the MF baseline. Do not add heavier ML dependencies
  for Phase 3.

## Correctness

- Apply the split before constructing the training graph or fitting MF.
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
- Phase 3: negative-sample validity, Recall@10 / NDCG@10 unit cases,
  eligibility/filtering, MF train-edge leakage, and a tiny seeded e2e smoke
  with reproducible metrics in `[0, 1]`.
