# SageRec

[![CI](https://github.com/sathariels/SageRec/actions/workflows/ci.yml/badge.svg)](https://github.com/sathariels/SageRec/actions/workflows/ci.yml)

SageRec is a MovieLens recommender whose systems core is a **C++17 bipartite CSR graph** and **seeded neighbor sampler**, exposed to Python as the pybind11 module `graph_sampler`. Python owns leakage-safe leave-one-out prep, an implicit matrix-factorization baseline, and a shared Recall@10 / NDCG@10 evaluator.

GraphSAGE is the accepted GNN direction. **Training is not implemented yet** — there is no PyTorch Geometric model, mini-batch loader, or MovieLens 100K quality table.

Public source: [github.com/sathariels/SageRec](https://github.com/sathariels/SageRec).

## Status at a glance

| Built | Not yet |
| --- | --- |
| C++ bipartite CSR + seeded neighbor sampler | MovieLens download / on-disk `data/processed/` writes |
| MovieLens 100K in-memory `u.data` parser | C++ vs Python timing charts and stored speedups |
| pybind11 `graph_sampler` bindings | GraphSAGE training / PyG mini-batch path |
| Python reference sampler + native parity tests | MovieLens 100K leaderboard / published Recall@10·NDCG@10 |
| ADR-003 in-memory leave-one-out split/prep | |
| Implicit MF baseline (NumPy logistic SGD) | |
| Shared Recall@10 / NDCG@10 evaluator (synthetic smoke) | |
| Release CMake build + GitHub Actions CI | |

The MF ranking smoke is a tiny synthetic unittest. It is **not** a MovieLens 100K result.

## Why this project

This is a systems + ML-infra portfolio piece, not a notebook demo.

- **Leakage-safe graph:** the sampling CSR is built from training positives only; held-out edges never enter preprocessing, negatives, or candidate filtering.
- **Native sampler:** neighbor sampling is real C++ CSR code with an explicit seed, not a wrapper around a graph library.
- **Harness first:** native tests, binding tests, sampler parity, split leakage tests, and a seeded MF metric smoke run in CI.

The intended training path will call `graph_sampler` from Python. That path does not exist yet.

## Architecture

Intended system. Solid arrows are implemented. Dashed arrows lead to work that is **not implemented**.

```mermaid
flowchart LR
    A[MovieLens ratings] --> B[Leakage-safe split and prep]
    B --> C[C++ CSR graph]
    C --> D[C++ neighbor sampler]
    D --> E[pybind11 graph_sampler]
    E -.-> F[Python mini-batch loader]
    F -.-> G[GraphSAGE]
    B --> H[MF baseline]
    G -.-> I[Recall@10 and NDCG@10]
    H --> I
    D -.-> J[C++ vs Python timing charts]
    D --> R[Python reference sampler]
```

Users and movies are distinct node types in one bipartite graph. Each training interaction is stored in both directions. Users occupy `[0, num_users)`; movies occupy `[num_users, num_users + num_movies)`.

## What's implemented

| Piece | Where |
| --- | --- |
| CSR + `sample_neighbors` | [`cpp/include/sagerec/bipartite_csr.hpp`](cpp/include/sagerec/bipartite_csr.hpp), [`cpp/src/bipartite_csr.cpp`](cpp/src/bipartite_csr.cpp) |
| MovieLens 100K parser | [`cpp/include/sagerec/movielens_100k.hpp`](cpp/include/sagerec/movielens_100k.hpp) — in-memory `u.data` text only |
| Python extension | `graph_sampler` via [`cpp/src/bindings.cpp`](cpp/src/bindings.cpp); stubs in [`python/graph_sampler.pyi`](python/graph_sampler.pyi) |
| Reference sampler | [`python/sagerec_reference_sampler.py`](python/sagerec_reference_sampler.py) — matches the native contract for parity tests |
| Leave-one-out prep | [`python/sagerec_prep.py`](python/sagerec_prep.py) — in-memory ADR-003 split; train-only pairs for `BipartiteCSR` |
| Scoring protocol | [`python/sagerec_scoring.py`](python/sagerec_scoring.py) — `PairScorer` for baseline and future GNN |
| Training negatives | [`python/sagerec_negatives.py`](python/sagerec_negatives.py) — exclude known positives in the caller-supplied scope |
| MF baseline | [`python/sagerec_baseline.py`](python/sagerec_baseline.py) — seeded NumPy logistic SGD |
| Ranking metrics | [`python/sagerec_metrics.py`](python/sagerec_metrics.py) — per-user then macro-averaged Recall@10 and NDCG@10 |

Parser and prep do not download MovieLens. Callers must pass **training-positive** `local_pairs()` into `BipartiteCSR`. The reference sampler reads CSR `offsets`/`neighbors`; it does not build graphs or ingest ratings files.

Layout: [`cpp/`](cpp/) native core, [`python/`](python/) prep/baseline/metrics/tests, [`docs/`](docs/) contracts and ADRs, [`data/`](data/) schemas (no raw dataset), [`results/`](results/) for future metrics and charts.

## Owner decisions

Recorded in [`docs/decisions.md`](docs/decisions.md):

| ADR | Status | Choice |
| --- | --- | --- |
| ADR-001 | Accepted | MovieLens 100K (1M deferred) |
| ADR-002 | Accepted | GraphSAGE (not GCN) |
| ADR-003 | Accepted | Per-user chronological leave-one-out; min 3 interactions; cold-start users stay in train and are excluded from ranking |
| ADR-004 | Accepted | Implicit-feedback matrix factorization (not node2vec) |
| ADR-005 | Proposed | Uniform sampling without replacement |

The native sampler already implements the proposed ADR-005 defaults (full neighborhood when `k >= degree`; empty for isolated nodes or `k = 0`). That is an implementation contract, not an accepted experiment decision.

## Build and test

Debian/Ubuntu: `cmake`, a C++17 compiler (`g++`), `python3-dev`, `pybind11-dev`, `python3-pybind11`, and `python3-numpy`.

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build --config Release
ctest --test-dir build --output-on-failure --build-config Release
```

Use `g++` (or another complete C++17 toolchain). A `c++` symlink that points at Clang without a discoverable `libstdc++` will fail at configure time.

CTest runs native CSR/sampler/parser tests and Python unittest discovery (bindings, sampler parity, leave-one-out leakage, MF ranking smoke). After a successful build:

```bash
PYTHONPATH=build:python python3 -m unittest discover -s python/tests -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_mf_baseline.py -v
```

`import graph_sampler` loads the compiled extension. Construction takes local `(user_id, movie_id)` pairs. Do not vendor MovieLens data.

## Honesty

- No trained GraphSAGE, no production users, no claimed latency speedups.
- No invented metrics. Tiny synthetic MF smoke ≠ MovieLens 100K evaluation.
- MovieLens download, on-disk prep, timing charts, and GNN training are still ahead.
- This GitHub repository is the public homepage. Do not treat an Origin (or other private) URL as the project home.

Acceptance criteria and component contracts: [`docs/project-requirements.md`](docs/project-requirements.md), [`docs/architecture.md`](docs/architecture.md). Contributors: read [`AGENTS.md`](AGENTS.md) before changing code.
