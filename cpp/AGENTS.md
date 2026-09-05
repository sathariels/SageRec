# C++ Backend Agent Guide

## Responsibility

Own native CSR construction and validation, neighbor sampling, MovieLens 100K
`u.data` parsing, pybind11 bindings, native tests, and native-side benchmark
support. Do not add MovieLens 1M code, downloads, or file-path helpers.

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
- `sagerec::parse_movielens_100k` in `include/sagerec/movielens_100k.hpp` parses
  in-memory tab-separated `u.data` text. It is not part of `BipartiteCSR`.
- `graph_sampler` also exposes `parse_movielens_100k`, `MovieLens100kRatings`,
  and `MovieLens100kInteraction`. Bindings copy Python text into owned storage
  before releasing the GIL; they do not add file-path or download helpers.
- Parser mappings: local ID is the rank of each source ID among sorted unique
  IDs of that type. Rating and timestamp are preserved. No split is applied.
- Parser `GraphError` cases: empty input, wrong field count, non-integer fields,
  nonpositive source IDs, and duplicate source user-movie pairs.

## Required tests

- Empty graph and isolated node behavior.
- Directed construction helper behavior, if one exists.
- Required undirected bipartite construction.
- Duplicate and self-edge policy.
- Exact CSR offsets and neighbor storage on a known graph.
- `k = 0`, `k < degree`, `k = degree`, and `k > degree`.
- Sample uniqueness, bounds, seed reproducibility, and statistical sanity.
- MovieLens 100K parsing from tiny in-memory strings, mapping order, preserved
  rating/timestamp, CSR handoff via `local_pairs()`, and malformed-row errors.
- Python construction, lifetime, exception, sampling, and in-memory parser
  smoke tests (`local_pairs()` CSR handoff and `GraphError` on bad input).
- CTest `python_binding_smoke` discovers `python/tests`, including
  native-vs-reference sampler parity. PYTHONPATH includes the build
  directory and `python/` so `sagerec_reference_sampler` imports cleanly.

Do not add GNN training, downloads, or a second sampler module name.
