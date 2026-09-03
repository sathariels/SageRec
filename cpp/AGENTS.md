# C++ Backend Agent Guide

## Responsibility

Own native CSR construction and validation, neighbor sampling, pybind11
bindings, native tests, and native-side benchmark support. MovieLens 100K
ingestion is remaining Phase 1 work and is not in the tree yet. Do not add
MovieLens 1M code.

## Required design properties

- Use C++17 or newer only if a decision explicitly raises the minimum.
- Keep the graph core independent of Python and machine-learning libraries.
- Store adjacency contiguously in CSR and validate all structural invariants.
- Implement sampling directly; do not delegate core work to a graph library.
- Make random sampling reproducible from an explicit seed.
- Clearly define ownership and avoid dangling views across the Python boundary.
- Release the Python GIL around substantial native construction and sampling work.
- Return actionable errors for malformed input, invalid dimensions, and bad node IDs.
- Benchmark optimized Release builds, never Debug builds.
- Public module name is `graph_sampler` unless `docs/decisions.md` changes it.

## Current public contract

- `sagerec::BipartiteCSR` in `include/sagerec/bipartite_csr.hpp`.
- Local `(user_id, movie_id)` construction; bidirectional train-only adjacency.
- Duplicate interactions are deduplicated (proposed default).
- `sample_neighbors(node_id, k, seed)` uses proposed ADR-005 defaults.
- Invalid IDs and `k < 0` throw `sagerec::GraphError` with the expected range.

## Required tests

- Empty graph and isolated node behavior.
- Directed construction helper behavior, if one exists.
- Required undirected bipartite construction.
- Duplicate and self-edge policy.
- Exact CSR offsets and neighbor storage on a known graph.
- `k = 0`, `k < degree`, `k = degree`, and `k > degree`.
- Sample uniqueness, bounds, seed reproducibility, and statistical sanity.
- MovieLens parsing and malformed-row errors, once ingestion exists.
- Python construction, lifetime, exception, and sampling smoke tests.

Do not add GNN training, downloads, or a second sampler module name.
