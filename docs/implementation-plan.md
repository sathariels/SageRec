# Phased Implementation Plan

ADR-001 (MovieLens 100K) and ADR-002 (GraphSAGE) are accepted. MovieLens 1M
remains deferred. Code work may proceed within the current phase only.

## Phase 1: Native foundation

Completed in the first verified slice:

- Define native public contracts and CMake targets.
- Implement validated train-only CSR construction from explicit interactions.
- Implement seeded sampling using the proposed ADR-005 defaults as the
  implementation contract (ADR-005 is still proposed).
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

- MovieLens download or file-path ingestion helpers.
- Do not add MovieLens 1M parsers, paths, or downloads.

Exit condition: clean Release build, CTest pass, extension import and smoke test pass.

## Phase 2: Data and benchmark harness

- Implement deterministic preparation, ID mappings, split, and manifest.
- Implement the equivalent naive Python sampler.
- Create correctness-parity and performance workloads.
- Store machine-readable benchmark results and generated chart.

Exit condition: selected dataset prepares reproducibly and benchmark evidence is complete.

## Phase 3: Baseline

- Implement accepted baseline and common scoring interface.
- Add tiny deterministic training tests.
- Evaluate through the shared ranking evaluator.

Exit condition: baseline produces reproducible Recall@10 and NDCG@10.

## Phase 4: GNN

- Implement accepted GNN and native-backed mini-batch loader.
- Verify native sampler usage during training.
- Add unit and tiny end-to-end tests.
- Tune only on validation data.

Exit condition: reproducible GNN run and test metrics exist.

## Phase 5: Comparison and documentation

- Run consistent multi-seed experiments where practical.
- Generate final metric table and chart.
- Document setup, architecture, commands, results, limitations, and troubleshooting.
- Run all quality gates from a clean environment.

Exit condition: all project acceptance criteria pass.
