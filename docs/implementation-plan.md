# Phased Implementation Plan

No phase containing code begins until ADR-001 and ADR-002 are accepted.

## Phase 1: Native foundation

- Define native public contracts and CMake targets.
- Implement validated CSR construction and selected MovieLens ingestion.
- Implement seeded sampling with accepted semantics.
- Add native tests for construction, parsing, edge cases, and reproducibility.
- Add thin pybind11 bindings and Python binding tests.

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
