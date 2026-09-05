# SageRec

SageRec is a graph-neural-network recommendation engine with a C++ graph
preprocessing and neighbor-sampling backend and a Python/PyTorch Geometric
training stack. The verified slice is the native CSR graph, seeded sampler,
MovieLens 100K `u.data` parser, and `graph_sampler` bindings. GNN training,
MovieLens download, the baseline, and benchmark charts are not implemented yet.

## Intended system

```mermaid
flowchart LR
    A[MovieLens ratings] --> B[Data preparation and leakage-safe split]
    B --> C[C++ CSR graph builder]
    C --> D[C++ neighbor sampler]
    D --> E[pybind11 graph_sampler module]
    E --> F[Python mini-batch loader]
    F --> G[GraphSAGE recommender]
    B --> H[Baseline recommender]
    G --> I[Recall@10 and NDCG@10]
    H --> I
    D --> J[C++ versus Python benchmark]
```

Users and movies are distinct node types in one bipartite graph. Ratings form
user-movie edges. The C++ sampler is intended to be the actual training data
backend rather than a demonstration wrapper.

## Repository map

| Directory | Responsibility |
| --- | --- |
| `cpp/` | Native CSR construction, ML-100K parser, sampling, bindings, tests |
| `python/` | Binding stubs, reference sampler, tests; later data/GNN/baseline |
| `data/` | Local raw inputs and reproducible processed artifacts |
| `results/` | Metrics, benchmark summaries, and charts |
| `docs/` | Architecture, decisions, protocols, and plans |
| `scripts/` | Future thin, reproducible workflow entry points |

Read [AGENTS.md](AGENTS.md) before making changes. Each subsystem also has a
local `AGENTS.md` with narrower requirements.

## Owner decisions

Recorded in [docs/decisions.md](docs/decisions.md):

1. **ADR-001 (accepted, 2026-09-02, Nithilan Kumaran):** MovieLens 100K.
   MovieLens 1M is deferred; do not add 1M paths, configs, or downloads.
2. **ADR-002 (accepted, 2026-09-02, Nithilan Kumaran):** GraphSAGE, not GCN.

ADR-003 (split), ADR-004 (baseline), and ADR-005 (sampler replacement) remain
proposals. The native sampler uses the proposed ADR-005 defaults as its
implementation contract: uniform sampling without replacement, full neighborhood
when `k >= degree`, empty result for isolated nodes or `k = 0`.

## Build and test the native foundation

Dependencies on Debian/Ubuntu: `cmake`, a C++17 compiler, `python3-dev`,
`pybind11-dev`, and `python3-pybind11`.

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build --config Release
ctest --test-dir build --output-on-failure --build-config Release
```

Use `g++` (or another complete C++17 toolchain). A `c++` symlink that points at
Clang without a discoverable `libstdc++` will fail at configure time.

CTest runs the native CSR/sampler cases and the Python unittest discover
suite (binding smoke tests plus native-vs-reference sampler parity).
To run those Python tests directly after a successful build:

```bash
PYTHONPATH=build python3 -m unittest discover -s python/tests -v
```

`python/sagerec_reference_sampler.py` is a naive Python neighbor sampler
that matches the native `sample_neighbors` contract for correctness
comparison. It reads CSR `offsets`/`neighbors` from `BipartiteCSR` and
does not construct graphs or ingest MovieLens files. Timing charts and
benchmark reports are not implemented.

`import graph_sampler` loads the compiled extension. Construction takes local
`(user_id, movie_id)` pairs on a synthetic graph; do not vendor MovieLens data.

MovieLens 100K ingestion is `sagerec::parse_movielens_100k` in
`cpp/include/sagerec/movielens_100k.hpp`, also bound as
`graph_sampler.parse_movielens_100k`. It parses in-memory tab-separated
`u.data` text, remaps source IDs to contiguous zero-based local IDs, and
preserves rating and timestamp. It does not download data, apply a split, or
build the CSR; pass training-positive `local_pairs()` into `BipartiteCSR`.
Parser tests use tiny strings only.

## Target deliverables

- A clean CMake build for the C++17/pybind11 module.
- A tested CSR graph and native neighbor sampler.
- A tested MovieLens 100K `u.data` parser with deterministic ID mappings.
- A naive Python reference sampler with native parity tests (timing
  charts and stored benchmark numbers not yet).
- A PyTorch Geometric GraphSAGE model trained with negative sampling and the
  native sampler (not yet).
- A matrix-factorization or node2vec baseline (ADR-004 still proposed).
- Leakage-safe evaluation with Recall@10 and NDCG@10 (not yet).
- Reproducible result tables and charts (not yet).

See [docs/project-requirements.md](docs/project-requirements.md) for acceptance
criteria and [docs/architecture.md](docs/architecture.md) for component contracts.
