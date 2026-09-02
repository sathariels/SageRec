# C++ Backend Agent Guide

## Responsibility

Own native MovieLens ingestion, CSR construction and validation, neighbor
sampling, pybind11 bindings, native tests, and native-side benchmark support.

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

## Required tests

- Empty graph and isolated node behavior.
- Directed construction helper behavior, if one exists.
- Required undirected bipartite construction.
- Duplicate and self-edge policy.
- Exact CSR offsets and neighbor storage on a known graph.
- `k = 0`, `k < degree`, `k = degree`, and `k > degree`.
- Sample uniqueness, bounds, seed reproducibility, and statistical sanity.
- MovieLens parsing and malformed-row errors.
- Python construction, lifetime, exception, and sampling smoke tests.

Do not create C++ files until ADR-001 and ADR-002 are accepted by the owner.
