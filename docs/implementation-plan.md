# Phased Implementation Plan

ADR-001 (MovieLens 100K), ADR-002 (GraphSAGE), ADR-003 (per-user
chronological leave-one-out), ADR-004 (matrix factorization), and ADR-005
(uniform sampling without replacement) are accepted. MovieLens 1M remains
deferred. Phase 2 download and on-disk `processed/` prep are open for
MovieLens 100K. Phase 2 timing charts stay closed. Phase 4 is open for the
native-backed mini-batch neighborhood harness. Full GraphSAGE training
remains incomplete.

## Phase 1: Native foundation

Completed in the first verified slice:

- Define native public contracts and CMake targets.
- Implement validated train-only CSR construction from explicit interactions.
- Implement seeded sampling using the ADR-005 without-replacement contract.
- Add native tests for construction, edge cases, and seed reproducibility.
- Add thin pybind11 bindings, Python binding tests, and Release CI.

Completed in the ingestion slice:

- Native MovieLens 100K `u.data` parser (`parse_movielens_100k`) with
  deterministic zero-based mappings and preserved rating/timestamp.
- Actionable `GraphError` for empty input, malformed rows/fields/types,
  nonpositive source IDs, and duplicate source user-movie pairs.
- Parser stays separate from CSR; callers feed `local_pairs()` into
  `BipartiteCSR` after they have selected training positives.
- Native parser tests use tiny in-memory strings only.

Completed in the parser-binding slice:

- `graph_sampler.parse_movielens_100k` and `MovieLens100kRatings` pybind11
  surface with matching `.pyi` stubs.
- Python binding smoke tests use tiny in-memory `u.data` strings only:
  mappings, `local_pairs()` CSR handoff, and malformed-input `GraphError`.

Still not started in Phase 1:

- Do not add MovieLens 1M parsers, paths, or downloads. Download and
  on-disk prep belong to the opened Phase 2 slice, not the native parser.

Exit condition: clean Release build, CTest pass, extension import and smoke test pass.

## Phase 2: Data and benchmark harness

Opened in the reference-sampler slice:

- Naive Python sampler in `python/sagerec_reference_sampler.py` matching the
  native `sample_neighbors` contract (Fisher–Yates prefix, `std::mt19937_64`,
  unbiased `uniform_below`).
- Unittest parity cases against compiled `graph_sampler` on synthetic graphs.

Opened in the ADR-003 split/prep slice (Phase 2 is not complete):

- In-memory leave-one-out assignment in `python/sagerec_prep.py` from
  already-normalized 100K interactions.
- Train-only `(user_id, movie_id)` pairs for `BipartiteCSR`.
- Manifest schema (`data/processed/manifest.schema.json`) and builder.
- Leakage, eligibility, timestamp-order, cold-start, and determinism tests.

Opened in the Phase 2 download / on-disk prep slice:

- Official GroupLens MovieLens 100K download (`python/sagerec_download.py`)
  with published archive MD5 verification and `u.data` extraction under
  `data/raw/` (gitignored).
- Path/bytes ingestion through `graph_sampler.parse_movielens_100k` and
  ADR-003 `sagerec_prep`, writing `data/processed/` artifacts plus a filled
  manifest (`python/sagerec_dataset.py`).
- Thin scripts: `scripts/download_movielens_100k.py`,
  `scripts/prepare_movielens_100k.py`.
- Fixture tests for checksum, zip layout, path ingestion, and manifest
  writes. Live download is optional and skipped in default CI.

Still not started:

- Performance workloads, stored timing results, and generated charts.

Exit condition: selected dataset prepares reproducibly and benchmark evidence is complete.

## Phase 3: Baseline

Opened in the ADR-004 matrix-factorization slice:

- Common `PairScorer` interface (`python/sagerec_scoring.py`); models expose
  ranking scores only. Metric logic lives in the shared evaluator.
- Implicit-feedback MF baseline (`python/sagerec_baseline.py`): seeded NumPy
  logistic SGD with configurable factors, epochs, learning rate, L2, and
  negatives-per-positive.
- Training negative sampling (`python/sagerec_negatives.py`) that excludes
  known positives in the training scope.
- Shared ranking evaluator (`python/sagerec_metrics.py`): per-user
  Recall@10 and NDCG@10 with macro averaging, ADR-003 eligibility, and
  train (plus validation when scoring test) candidate filtering.
- Tiny deterministic unittest smoke: synthetic interactions →
  `sagerec_prep` leave-one-out → brief seeded MF → finite metrics in
  `[0, 1]` that match across two runs. Leakage checks confirm held-out
  positives are absent from the MF train edge set / train CSR.

Still not started:

- Full GraphSAGE training, PyG model weights, and a GNN-versus-baseline
  quality table (Phase 4 remaining work).
- node2vec (rejected unless a superseding ADR accepts it).
- Do not invent MovieLens 100K quality numbers. A real single-seed 100K MF
  run is stored under `results/mf_movielens_100k.json` when generated from
  the official archive (not a GraphSAGE or multi-seed leaderboard).

Exit condition: baseline produces reproducible Recall@10 and NDCG@10 on
the tiny synthetic path (verification only, not a MovieLens 100K result).

## Phase 4: GNN

Opened in the mini-batch harness slice:

- Native-backed neighborhood helper (`python/sagerec_minibatch.py`) that
  builds a train-only `BipartiteCSR` from local pairs and expands seeded
  multi-hop neighborhoods by calling `graph_sampler.sample_neighbors`
  (ADR-005 without replacement).
- Deterministic tests (`python/tests/test_minibatch_native.py`) on a tiny
  synthetic graph: compiled-extension requirement, native-call spying
  (reference sampler must not be used), seed reproducibility, empty /
  `k = 0` / full-neighborhood cases, and train-only CSR leakage.

Still not started:

- GraphSAGE layers, PyTorch Geometric tensors, model weights, and training
  loops.
- GNN ranking metrics or a GNN-versus-baseline quality table.
- Do not add PyG/PyTorch as a default-CI dependency until a later slice
  actually trains a model.

Exit condition: reproducible GNN run and test metrics exist.

## Phase 5: Comparison and documentation

- Run consistent multi-seed experiments where practical.
- Generate final metric table and chart.
- Document setup, architecture, commands, results, limitations, and troubleshooting.
- Run all quality gates from a clean environment.

Exit condition: all project acceptance criteria pass.
