# SageRec

[![CI](https://github.com/sathariels/SageRec/actions/workflows/ci.yml/badge.svg)](https://github.com/sathariels/SageRec/actions/workflows/ci.yml)

SageRec is a MovieLens recommender whose systems core is a **C++17 bipartite CSR graph** and **seeded neighbor sampler**, exposed to Python as the pybind11 module `graph_sampler`. Python owns leakage-safe leave-one-out prep, MovieLens 100K download / on-disk prep, an implicit matrix-factorization baseline, a shared Recall@10 / NDCG@10 evaluator, a native-backed mini-batch neighborhood helper, and GraphSAGE training on that harness.

GraphSAGE is the accepted GNN direction. Phase 4 trains GraphSAGE on the native mini-batch harness (Python calls `graph_sampler` / `sagerec_minibatch`; not a PyG NeighborLoader). Phase 5 stores a **single-seed MovieLens 100K GraphSAGE quality run** and an honest **GNN-versus-MF** table/chart under [`results/`](results/). Tiny synthetic GraphSAGE metrics remain protocol smoke and are not those 100K numbers.

Public source: [github.com/sathariels/SageRec](https://github.com/sathariels/SageRec).

## Status at a glance

| Built | Not yet |
| --- | --- |
| C++ bipartite CSR + seeded neighbor sampler (ADR-005 without replacement) | C++ vs Python timing charts and stored speedups |
| MovieLens 100K in-memory `u.data` parser | PyG NeighborLoader / SAGEConv (intended later) |
| Official 100K download + on-disk `data/processed/` prep | Production serving |
| Native-backed mini-batch neighborhood helper | Multi-seed published leaderboard |
| GraphSAGE training on native samples (CPU PyTorch) | |
| pybind11 `graph_sampler` bindings | |
| Python reference sampler + native parity tests | |
| ADR-003 leave-one-out split/prep (in-memory and on-disk) | |
| Implicit MF baseline (NumPy logistic SGD) | |
| Shared Recall@10 / NDCG@10 evaluator | |
| Single-seed MovieLens 100K MF metrics in [`results/mf_movielens_100k.json`](results/mf_movielens_100k.json) | |
| Single-seed MovieLens 100K GraphSAGE metrics in [`results/graphsage_movielens_100k.json`](results/graphsage_movielens_100k.json) | |
| GNN-versus-MF comparison JSON, markdown table, and SVG chart | |
| Release CMake build + GitHub Actions CI | |

The MF ranking smoke unittest is a tiny synthetic path. MovieLens 100K implicit-MF and GraphSAGE metrics (Recall@10 / NDCG@10) with split, seed, hyperparameter, and eligibility provenance are stored under [`results/`](results/). Each file is a **single-seed** 100K run, not a published multi-seed leaderboard. The MF JSON is not a GraphSAGE result.

## Why this project

This is a systems + ML-infra portfolio piece, not a notebook demo.

- **Leakage-safe graph:** the sampling CSR is built from training positives only; held-out edges never enter preprocessing, negatives, or candidate filtering.
- **Native sampler:** neighbor sampling is real C++ CSR code with an explicit seed, not a wrapper around a graph library.
- **Harness first:** native tests, binding tests, sampler parity, split leakage tests, download/prep fixture tests, a seeded MF metric smoke run, mini-batch native-sampler tests, and a GraphSAGE training smoke that spies on `graph_sampler` in CI.

The mini-batch helper in [`python/sagerec_minibatch.py`](python/sagerec_minibatch.py) calls `graph_sampler` for seeded multi-hop neighborhood expansion. GraphSAGE training in [`python/sagerec_graphsage.py`](python/sagerec_graphsage.py) uses those native samples with PyTorch mean aggregation. It does not use a PyG neighbor sampler.

## Architecture

Intended system. Solid arrows are implemented. Dashed arrows lead to work that is **not implemented** (timing charts). The Phase 5 100K comparison table/chart is implemented from stored MF and GraphSAGE result files.

```mermaid
flowchart LR
    A[MovieLens ratings] --> B[Leakage-safe split and prep]
    B --> C[C++ CSR graph]
    C --> D[C++ neighbor sampler]
    D --> E[pybind11 graph_sampler]
    E --> F[Python mini-batch neighborhood helper]
    F --> G[GraphSAGE]
    B --> H[MF baseline]
    G --> I[Recall@10 and NDCG@10]
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
| Reference sampler | [`python/sagerec_reference_sampler.py`](python/sagerec_reference_sampler.py) — matches the native ADR-005 contract for parity tests |
| Leave-one-out prep | [`python/sagerec_prep.py`](python/sagerec_prep.py) — in-memory ADR-003 split; train-only pairs for `BipartiteCSR` |
| 100K download | [`python/sagerec_download.py`](python/sagerec_download.py) — official GroupLens zip, published MD5, extract `u.data` |
| On-disk prep | [`python/sagerec_dataset.py`](python/sagerec_dataset.py) — path/bytes → parser → ADR-003 → `data/processed/` + manifest |
| Scoring protocol | [`python/sagerec_scoring.py`](python/sagerec_scoring.py) — `PairScorer` for baseline and future GNN |
| Training negatives | [`python/sagerec_negatives.py`](python/sagerec_negatives.py) — exclude known positives in the caller-supplied scope |
| MF baseline | [`python/sagerec_baseline.py`](python/sagerec_baseline.py) — seeded NumPy logistic SGD |
| Ranking metrics | [`python/sagerec_metrics.py`](python/sagerec_metrics.py) — per-user then macro-averaged Recall@10 and NDCG@10 |
| Mini-batch harness | [`python/sagerec_minibatch.py`](python/sagerec_minibatch.py) — train-only CSR + native multi-hop `sample_neighbors` |
| GraphSAGE trainer | [`python/sagerec_graphsage.py`](python/sagerec_graphsage.py) — CPU PyTorch mean layers + `PairScorer` on native samples |
| Comparison writer | [`python/sagerec_compare.py`](python/sagerec_compare.py) — MF vs GraphSAGE JSON, markdown table, SVG chart from stored results |

Callers must pass **training-positive** `local_pairs()` into `BipartiteCSR`. The reference sampler reads CSR `offsets`/`neighbors`; it does not build graphs or ingest ratings files.

Layout: [`cpp/`](cpp/) native core, [`python/`](python/) prep/baseline/metrics/mini-batch/GraphSAGE/comparison/tests, [`docs/`](docs/) contracts and ADRs, [`data/`](data/) schemas (no raw dataset), [`results/`](results/) for metrics and charts, [`scripts/`](scripts/) thin download/prep/MF/GraphSAGE launchers.

## MovieLens 100K download and prep

Official archive (GroupLens):

- URL: `https://files.grouplens.org/datasets/movielens/ml-100k.zip`
- Published MD5: `0e33842e24a9c977be4e0107933c0723` (from `ml-100k.zip.md5`)

```bash
PYTHONPATH=build:python python3 scripts/download_movielens_100k.py --raw-dir data/raw
PYTHONPATH=build:python python3 scripts/prepare_movielens_100k.py \
  --udata data/raw/ml-100k/u.data --processed-dir data/processed
```

`data/raw/` and `data/processed/` stay gitignored except schemas and agent guides. Do not commit the zip or `u.data`. MovieLens 1M is refused.

If `files.grouplens.org` presents an expired TLS certificate, the downloader retries without TLS verification **only when** the expected archive MD5 will still be checked.

## MovieLens 100K GraphSAGE quality run and GNN-versus-MF comparison

Shared protocol with the stored MF run: ADR-003 leave-one-out, test split, Recall@10 / NDCG@10, 943 eligible users, seed **7**. GraphSAGE uses a modest CPU-friendly set (`embedding_dim=16`, `hidden_dim=16`, two layers, fanouts `(8, 8)`, **3 Adam epochs**, `batch_size=256`, `learning_rate=0.01`, 2 negatives). MF used **1 SGD epoch**; fairness is the shared eval protocol, not identical wall-clock or optimizer. Extra training seeds are derived from 7 (`derived_sample_seed(seed, epoch, step)` per mini-batch; hop/source mixing inside multi-hop). Ranking encodes each graph node once at seed 7 via native sampling, then dots cached embeddings.

```bash
PYTHONPATH=build:python python3 scripts/run_graphsage_movielens_100k.py \
  --processed-dir data/processed --raw-dir data/raw --seed 7
PYTHONPATH=build:python python3 scripts/write_gnn_vs_mf_comparison.py
```

The first command downloads and preps 100K if `data/processed/` is missing. Outputs:

- [`results/graphsage_movielens_100k.json`](results/graphsage_movielens_100k.json)
- [`results/gnn_vs_mf_movielens_100k.json`](results/gnn_vs_mf_movielens_100k.json)
- [`results/gnn_vs_mf_movielens_100k.md`](results/gnn_vs_mf_movielens_100k.md)
- [`results/gnn_vs_mf_movielens_100k.svg`](results/gnn_vs_mf_movielens_100k.svg)

Keep [`results/mf_movielens_100k.json`](results/mf_movielens_100k.json) as the MF side of the comparison. Do not retcon it as GraphSAGE.

## Owner decisions

Recorded in [`docs/decisions.md`](docs/decisions.md):

| ADR | Status | Choice |
| --- | --- | --- |
| ADR-001 | Accepted | MovieLens 100K (1M deferred) |
| ADR-002 | Accepted | GraphSAGE (not GCN) |
| ADR-003 | Accepted | Per-user chronological leave-one-out; min 3 interactions; cold-start users stay in train and are excluded from ranking |
| ADR-004 | Accepted | Implicit-feedback matrix factorization (not node2vec) |
| ADR-005 | Accepted | Uniform sampling without replacement |

The native and Python reference samplers implement ADR-005 (full neighborhood when `k >= degree`; empty for isolated nodes or `k = 0`). Changing replacement policy requires a superseding ADR.

## Build and test

Debian/Ubuntu: `cmake`, a C++17 compiler (`g++`), `python3-dev`, `pybind11-dev`, `python3-pybind11`, and `python3-numpy`. GraphSAGE tests also need CPU PyTorch (`python/requirements-train.txt`):

```bash
python3 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Native C++ does not depend on PyTorch. PyTorch Geometric is the intended later production stack; this slice trains with in-repo mean layers on native samples.

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build --config Release
ctest --test-dir build --output-on-failure --build-config Release
```

Use `g++` (or another complete C++17 toolchain). A `c++` symlink that points at Clang without a discoverable `libstdc++` will fail at configure time.

CTest runs native CSR/sampler/parser tests and Python unittest discovery (bindings, sampler parity, leave-one-out leakage, download/prep fixtures, MF ranking smoke, mini-batch native sampling, GraphSAGE training smoke). After a successful build:

```bash
PYTHONPATH=build:python python3 -m unittest discover -s python/tests -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_mf_baseline.py -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_minibatch_native.py -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_graphsage_train.py -v
PYTHONPATH=build:python python3 scripts/run_graphsage_synthetic_smoke.py
PYTHONPATH=build:python python3 -m unittest python/tests/test_gnn_vs_mf_compare.py -v
```

Default CI does **not** download MovieLens. Set `SAGEREC_LIVE_MOVIELENS=1` only for the optional live-archive test.

`import graph_sampler` loads the compiled extension. Construction takes local `(user_id, movie_id)` pairs. Do not vendor MovieLens data.

## Honesty

- No production users, no claimed latency speedups, no invented metrics.
- Tiny synthetic MF and GraphSAGE smokes are protocol verification, not MovieLens 100K evaluation.
- A single-seed 100K MF run is recorded under `results/mf_movielens_100k.json`. It is not GraphSAGE and not a multi-seed leaderboard.
- A single-seed 100K GraphSAGE run is recorded under `results/graphsage_movielens_100k.json`. Neighborhoods come from `graph_sampler` via `sagerec_minibatch`, not a PyG NeighborLoader.
- The Phase 5 comparison table/chart is generated from those two JSON files.
- Timing charts and a PyG production training path are still ahead.
- This GitHub repository is the public homepage. Do not treat an Origin (or other private) URL as the project home.

Acceptance criteria and component contracts: [`docs/project-requirements.md`](docs/project-requirements.md), [`docs/architecture.md`](docs/architecture.md). Contributors: read [`AGENTS.md`](AGENTS.md) before changing code.
