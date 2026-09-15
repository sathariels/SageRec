# SageRec

[![CI](https://github.com/sathariels/SageRec/actions/workflows/ci.yml/badge.svg)](https://github.com/sathariels/SageRec/actions/workflows/ci.yml)

SageRec is a MovieLens recommender whose systems core is a **C++17 bipartite CSR graph** and **seeded neighbor sampler**, exposed to Python as the pybind11 module `graph_sampler`. Python owns leakage-safe leave-one-out prep, MovieLens 100K download / on-disk prep, an implicit matrix-factorization baseline, a shared Recall@10 / NDCG@10 evaluator, a native-backed mini-batch neighborhood helper, and GraphSAGE training on that harness (PyG `SAGEConv`, native neighborhoods).

GraphSAGE is the accepted GNN direction. Phase 4 trains GraphSAGE on the native mini-batch harness (Python calls `graph_sampler` / `sagerec_minibatch`; not a PyG NeighborLoader). Phase 5 stores a **single-seed MovieLens 100K GraphSAGE quality run** and an honest **GNN-versus-MF** table/chart under [`results/`](results/). Phase 6 uses PyG `SAGEConv` for message passing while neighborhoods still come from the native sampler. Phase 7 stores a **5-seed MF vs GraphSAGE leaderboard** with mean±sample-std under [`results/multiseed_gnn_vs_mf_movielens_100k.json`](results/multiseed_gnn_vs_mf_movielens_100k.json). Tiny synthetic GraphSAGE metrics remain protocol smoke and are not those 100K numbers.

Public source: [github.com/sathariels/SageRec](https://github.com/sathariels/SageRec).

## Status at a glance

| Built | Not yet |
| --- | --- |
| C++ bipartite CSR + seeded neighbor sampler (ADR-005 without replacement) | PyG NeighborLoader / ClusterLoader (rejected for primary experiments) |
| MovieLens 100K in-memory `u.data` parser | Production serving |
| Official 100K download + on-disk `data/processed/` prep | |
| Native-backed mini-batch neighborhood helper | |
| GraphSAGE training on native samples (CPU PyTorch Geometric `SAGEConv`) | |
| pybind11 `graph_sampler` bindings | |
| Python reference sampler + native parity tests | |
| ADR-003 leave-one-out split/prep (in-memory and on-disk) | |
| Implicit MF baseline (NumPy logistic SGD) | |
| Shared Recall@10 / NDCG@10 evaluator | |
| Single-seed MovieLens 100K MF metrics in [`results/mf_movielens_100k.json`](results/mf_movielens_100k.json) | |
| Single-seed MovieLens 100K GraphSAGE metrics in [`results/graphsage_movielens_100k.json`](results/graphsage_movielens_100k.json) | |
| GNN-versus-MF comparison JSON, markdown table, and SVG chart | |
| 5-seed MF vs GraphSAGE leaderboard ([`results/multiseed_gnn_vs_mf_movielens_100k.json`](results/multiseed_gnn_vs_mf_movielens_100k.json)) | |
| C++ vs Python reference sampler timing JSON, markdown, and SVG | |
| Release CMake build + GitHub Actions CI | |

The MF ranking smoke unittest is a tiny synthetic path. MovieLens 100K implicit-MF and GraphSAGE metrics (Recall@10 / NDCG@10) with split, seed, hyperparameter, and eligibility provenance are stored under [`results/`](results/). Phase 5 files are **single-seed** historical provenance. The ADR-007 leaderboard is a separate 5-seed SAGEConv measurement with mean±sample std. The MF JSON is not a GraphSAGE result.

## Why this project

This is a systems + ML-infra portfolio piece, not a notebook demo.

- **Leakage-safe graph:** the sampling CSR is built from training positives only; held-out edges never enter preprocessing, negatives, or candidate filtering.
- **Native sampler:** neighbor sampling is real C++ CSR code with an explicit seed, not a wrapper around a graph library.
- **Harness first:** native tests, binding tests, sampler parity, split leakage tests, download/prep fixture tests, a seeded MF metric smoke run, mini-batch native-sampler tests, a GraphSAGE training smoke that spies on `graph_sampler` in CI, and native-versus-reference sampler timing tests.

The mini-batch helper in [`python/sagerec_minibatch.py`](python/sagerec_minibatch.py) calls `graph_sampler` for seeded multi-hop neighborhood expansion. GraphSAGE training in [`python/sagerec_graphsage.py`](python/sagerec_graphsage.py) converts those native hops into PyG `edge_index` tensors and applies `SAGEConv`. It does not use a PyG neighbor sampler.

## Architecture

Intended system. Solid arrows are implemented. The Phase 5 100K comparison table/chart is implemented from stored MF and GraphSAGE result files. The ADR-007 multi-seed leaderboard reuses that protocol across five seeds. The Phase 2 sampler timing chart is generated from measured native vs Python reference timings.

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
    D --> J[C++ vs Python timing charts]
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
| Scoring protocol | [`python/sagerec_scoring.py`](python/sagerec_scoring.py) — `PairScorer` for baseline and GraphSAGE |
| Training negatives | [`python/sagerec_negatives.py`](python/sagerec_negatives.py) — exclude known positives in the caller-supplied scope |
| MF baseline | [`python/sagerec_baseline.py`](python/sagerec_baseline.py) — seeded NumPy logistic SGD |
| Ranking metrics | [`python/sagerec_metrics.py`](python/sagerec_metrics.py) — per-user then macro-averaged Recall@10 and NDCG@10 |
| Mini-batch harness | [`python/sagerec_minibatch.py`](python/sagerec_minibatch.py) — train-only CSR + native multi-hop `sample_neighbors` |
| GraphSAGE trainer | [`python/sagerec_graphsage.py`](python/sagerec_graphsage.py) — PyG `SAGEConv` + `PairScorer` on native samples |
| Comparison writer | [`python/sagerec_compare.py`](python/sagerec_compare.py) — MF vs GraphSAGE JSON, markdown table, SVG chart from stored results |
| Multi-seed aggregator | [`python/sagerec_multiseed.py`](python/sagerec_multiseed.py) — per-seed rows plus mean±sample std (ADR-007) |
| Sampler timing | [`python/sagerec_sampler_benchmark.py`](python/sagerec_sampler_benchmark.py) — native vs reference `sample_neighbors` JSON/SVG |

Callers must pass **training-positive** `local_pairs()` into `BipartiteCSR`. The reference sampler reads CSR `offsets`/`neighbors`; it does not build graphs or ingest ratings files.

Layout: [`cpp/`](cpp/) native core, [`python/`](python/) prep/baseline/metrics/mini-batch/GraphSAGE/comparison/multi-seed/benchmark/tests, [`docs/`](docs/) contracts and ADRs, [`data/`](data/) schemas (no raw dataset), [`results/`](results/) for metrics and charts, [`scripts/`](scripts/) thin download/prep/MF/GraphSAGE/timing/multi-seed launchers.

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

Shared protocol with the stored MF run: ADR-003 leave-one-out, test split, Recall@10 / NDCG@10, 943 eligible users, seed **7**. GraphSAGE uses a modest CPU-friendly set (`embedding_dim=16`, `hidden_dim=16`, two layers, fanouts `(8, 8)`, **3 Adam epochs**, `batch_size=256`, `learning_rate=0.01`, 2 negatives). Current training applies PyG `SAGEConv` to native mini-batches. MF used **1 SGD epoch**; fairness is the shared eval protocol, not identical wall-clock or optimizer. Extra training seeds are derived from 7 (`derived_sample_seed(seed, epoch, step)` per mini-batch; hop/source mixing inside multi-hop). Ranking encodes each graph node once at seed 7 via native sampling, then dots cached embeddings.

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

Single-seed test-split numbers copied from those files (seed 7, 943 users):

| Model | Epochs | Recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: |
| implicit MF | 1 SGD | 0.038176 | 0.020303 |
| GraphSAGE | 3 Adam | 0.047720 | 0.020969 |

This is historical **single-seed** provenance, not the published multi-seed leaderboard, and not a production-quality claim. The GraphSAGE row is the stored Phase 5 run (in-repo mean layers); it is not a SAGEConv 100K measurement. Do not retcon these files.

## MovieLens 100K multi-seed MF vs GraphSAGE leaderboard

ADR-007 publishes uncertainty on the **same** split, eligible users, candidate protocol, and Recall@10 / NDCG@10 as Phase 5. Both models use seeds **7, 11, 13, 17, 19** (seed 7 kept for continuity). GraphSAGE is the current Phase 6 path: PyG `SAGEConv` on native `graph_sampler` neighborhoods via `sagerec_minibatch` (not a NeighborLoader). MF is still **1 SGD epoch**; GraphSAGE is still **3 Adam epochs**. Fairness is the shared eval protocol, not matched wall-clock or optimizer budget.

```bash
PYTHONPATH=build:python python3 scripts/run_multiseed_gnn_vs_mf.py \
  --processed-dir data/processed --raw-dir data/raw
```

Default CI does **not** run that 100K job. Outputs (measured, not invented):

- [`results/multiseed_gnn_vs_mf_movielens_100k.json`](results/multiseed_gnn_vs_mf_movielens_100k.json)
- [`results/multiseed_gnn_vs_mf_movielens_100k.md`](results/multiseed_gnn_vs_mf_movielens_100k.md)
- [`results/multiseed_gnn_vs_mf_movielens_100k.svg`](results/multiseed_gnn_vs_mf_movielens_100k.svg)

Headline uncertainty is **mean ± sample standard deviation (n-1)** over the five seeds; median is also stored. Copy numbers from those files after a measured run. Phase 5 single-seed JSONs are left unchanged.

## Native vs Python reference sampler timing

Parity is checked on the same ADR-005 without-replacement queries before any timed repetition. The committed artifacts use a deterministic synthetic bipartite graph (512 users, 1024 movies, degree 32, `k=10`, 4000 queries, seed 7) so default CI never downloads MovieLens.

```bash
PYTHONPATH=build:python python3 scripts/run_sampler_timing.py
```

Outputs:

- [`results/sampler_timing.json`](results/sampler_timing.json)
- [`results/sampler_timing.md`](results/sampler_timing.md)
- [`results/sampler_timing.svg`](results/sampler_timing.svg)

Headline latency, throughput, and speedup use the **median** of measured repetitions after warm-up. Optional `--source movielens-100k` times a train-only CSR from existing `data/processed/` artifacts and does not download.

Copied from the stored JSON (Release, GNU 13.3.0, Python 3.12.3, 4-way Xeon; seed 7):

| Sampler | Median seconds | Median queries/s |
| --- | ---: | ---: |
| native `graph_sampler` | 0.003911 | 1,022,638 |
| Python reference | 0.434863 | 9,198 |

Median speedup (reference / native): **111.177×**. This is one-machine sampler evidence, not a production latency or SOTA claim.

## Owner decisions

Recorded in [`docs/decisions.md`](docs/decisions.md):

| ADR | Status | Choice |
| --- | --- | --- |
| ADR-001 | Accepted | MovieLens 100K (1M deferred) |
| ADR-002 | Accepted | GraphSAGE (not GCN) |
| ADR-003 | Accepted | Per-user chronological leave-one-out; min 3 interactions; cold-start users stay in train and are excluded from ranking |
| ADR-004 | Accepted | Implicit-feedback matrix factorization (not node2vec) |
| ADR-005 | Accepted | Uniform sampling without replacement |
| ADR-006 | Accepted | PyG `SAGEConv` on native `graph_sampler` neighborhoods (not NeighborLoader) |
| ADR-007 | Accepted | 5-seed MovieLens 100K MF vs GraphSAGE leaderboard (seeds 7, 11, 13, 17, 19) |

The native and Python reference samplers implement ADR-005 (full neighborhood when `k >= degree`; empty for isolated nodes or `k = 0`). Changing replacement policy requires a superseding ADR.

## Build and test

Debian/Ubuntu: `cmake`, a C++17 compiler (`g++`), `python3-dev`, `pybind11-dev`, `python3-pybind11`, and `python3-numpy`. GraphSAGE tests also need CPU PyTorch and PyTorch Geometric (`python/requirements-train.txt`):

```bash
python3 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install torch-geometric==2.6.1
```

Native C++ does not depend on PyTorch or PyG. Mean `SAGEConv` does not need `torch-scatter` / `torch-sparse` / `pyg-lib`. Neighborhoods still come from `graph_sampler`, not a PyG NeighborLoader.

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build --config Release
ctest --test-dir build --output-on-failure --build-config Release
```

Use `g++` (or another complete C++17 toolchain). A `c++` symlink that points at Clang without a discoverable `libstdc++` will fail at configure time.

CTest runs native CSR/sampler/parser tests and Python unittest discovery (bindings, sampler parity, leave-one-out leakage, download/prep fixtures, MF ranking smoke, mini-batch native sampling, GraphSAGE training smoke, sampler timing schema/parity). After a successful build:

```bash
PYTHONPATH=build:python python3 -m unittest discover -s python/tests -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_mf_baseline.py -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_minibatch_native.py -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_graphsage_train.py -v
PYTHONPATH=build:python python3 scripts/run_graphsage_synthetic_smoke.py
PYTHONPATH=build:python python3 -m unittest python/tests/test_gnn_vs_mf_compare.py -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_multiseed_gnn_vs_mf.py -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_sampler_benchmark.py -v
```

Default CI does **not** download MovieLens. Set `SAGEREC_LIVE_MOVIELENS=1` only for the optional live-archive test.

`import graph_sampler` loads the compiled extension. Construction takes local `(user_id, movie_id)` pairs. Do not vendor MovieLens data.

## Honesty

- No production users, no invented metrics, and no production-latency or SOTA speedup claims.
- Tiny synthetic MF and GraphSAGE smokes are protocol verification, not MovieLens 100K evaluation.
- A single-seed 100K MF run is recorded under `results/mf_movielens_100k.json`. It is not GraphSAGE. It remains Phase 5 single-seed provenance.
- A single-seed 100K GraphSAGE run is recorded under `results/graphsage_movielens_100k.json`. Neighborhoods come from `graph_sampler` via `sagerec_minibatch`, not a PyG NeighborLoader. Those stored numbers were measured with the Phase 5 in-repo mean layers; they are not a SAGEConv 100K rerun.
- The Phase 5 comparison table/chart is generated from those two JSON files.
- The ADR-007 5-seed SAGEConv leaderboard is stored under `results/multiseed_gnn_vs_mf_movielens_100k.json` (mean ± sample std). It does not overwrite the Phase 5 files. It is not a SOTA claim.
- Native vs Python reference sampler timings are stored under `results/sampler_timing.json` (synthetic graph, median of measured repetitions). They are one-machine evidence, not a production latency or SOTA claim.
- Production message passing is PyG `SAGEConv` on native mini-batches (ADR-006). PyG NeighborLoader is not the neighborhood source.
- This GitHub repository is the public homepage. Do not treat an Origin (or other private) URL as the project home.

Acceptance criteria and component contracts: [`docs/project-requirements.md`](docs/project-requirements.md), [`docs/architecture.md`](docs/architecture.md). Contributors: read [`AGENTS.md`](AGENTS.md) before changing code.
