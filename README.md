# SageRec

SageRec is a graph-neural-network recommendation engine with a C++ graph
preprocessing and neighbor-sampling backend and a Python/PyTorch Geometric
training stack. The verified slice is the native CSR graph, seeded sampler,
MovieLens 100K `u.data` parser, `graph_sampler` bindings, a Python reference
sampler, in-memory ADR-003 leave-one-out prep, an implicit matrix-factorization
baseline, and a shared Recall@10 / NDCG@10 evaluator. GNN training, MovieLens
download, and benchmark charts are not implemented yet.

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
| `python/` | Binding stubs, reference sampler, split/prep, MF baseline, ranking metrics, tests |
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
3. **ADR-003 (accepted, 2026-09-07, Nithilan Kumaran):** Per-user chronological
   leave-one-out. Eligible users need at least three interactions (latest →
   test, second-latest → validation, earlier → train). Users with fewer than
   three interactions are cold-start: all rows stay in train and the user is
   excluded from ranking eligibility. Timestamp ties break on
   `(user_id, movie_id)` (local IDs; equivalent to source-ID order).
4. **ADR-004 (accepted, 2026-09-08, Nithilan Kumaran):** Implicit-feedback
   matrix factorization, not node2vec. Phase 3 uses the same split, eligibility,
   candidate filtering, and Recall@10 / NDCG@10 protocol as the future GNN.

ADR-005 (sampler replacement) remains a proposal. The native sampler uses
the proposed ADR-005 defaults as its implementation contract: uniform
sampling without replacement, full neighborhood when `k >= degree`, empty
result for isolated nodes or `k = 0`. Do not add node2vec unless a
superseding ADR accepts it.

## Build and test the native foundation

Dependencies on Debian/Ubuntu: `cmake`, a C++17 compiler, `python3-dev`,
`pybind11-dev`, `python3-pybind11`, and `python3-numpy` (Phase 3 MF baseline).

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build --config Release
ctest --test-dir build --output-on-failure --build-config Release
```

Use `g++` (or another complete C++17 toolchain). A `c++` symlink that points at
Clang without a discoverable `libstdc++` will fail at configure time.

CTest runs the native CSR/sampler cases and the Python unittest discover
suite (binding smoke tests, native-vs-reference sampler parity,
leave-one-out prep/leakage tests, and the Phase 3 MF ranking smoke). To run
those Python tests directly after a successful build:

```bash
PYTHONPATH=build:python python3 -m unittest discover -s python/tests -v
```

`python/sagerec_reference_sampler.py` is a naive Python neighbor sampler
that matches the native `sample_neighbors` contract for correctness
comparison. It reads CSR `offsets`/`neighbors` from `BipartiteCSR` and
does not construct graphs or ingest MovieLens files. Timing charts and
benchmark reports are not implemented.

`python/sagerec_prep.py` assigns ADR-003 splits in memory and returns
train-only `(user_id, movie_id)` pairs for `BipartiteCSR`. It does not
download MovieLens data or write `data/processed/`. The manifest schema is
`data/processed/manifest.schema.json`.

The Phase 3 baseline is implicit-feedback matrix factorization
(`python/sagerec_baseline.py`) behind `sagerec_scoring.PairScorer`. Training
negatives (`python/sagerec_negatives.py`) exclude known training positives.
`python/sagerec_metrics.py` ranks candidates and reports macro-averaged
Recall@10 and NDCG@10 for ADR-003-eligible users, filtering train positives
(and validation positives when the target split is test). Tiny synthetic
unittests in `python/tests/test_mf_baseline.py` are protocol verification
only; they are not MovieLens 100K quality results.

```bash
PYTHONPATH=build:python python3 -m unittest python/tests/test_mf_baseline.py -v
```

The unittest discover command in the native-build section also runs that
smoke. GraphSAGE training is not implemented.

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
- Deterministic in-memory MovieLens 100K leave-one-out prep with leakage tests.
- A seeded implicit matrix-factorization baseline and shared ranking
  evaluator with tiny synthetic Recall@10 / NDCG@10 smoke tests
  (verification only, not a MovieLens 100K leaderboard).
- A PyTorch Geometric GraphSAGE model trained with negative sampling and the
  native sampler (not yet).
- Leakage-safe MovieLens 100K evaluation tables and charts (not yet).

See [docs/project-requirements.md](docs/project-requirements.md) for acceptance
criteria and [docs/architecture.md](docs/architecture.md) for component contracts.
