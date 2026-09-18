# SageRec

[![CI](https://github.com/sathariels/SageRec/actions/workflows/ci.yml/badge.svg)](https://github.com/sathariels/SageRec/actions/workflows/ci.yml)

**Problem:** GraphSAGE training needs fast, seeded neighbor sampling on a leakage-safe user–movie graph. A PyG `NeighborLoader` wrapper would hide the systems work.

**Result:** SageRec puts a real C++17 CSR sampler in the Python training loop, then compares GraphSAGE to implicit matrix factorization on the same MovieLens 100K protocol, then serves top-K from a checkpoint.

| 6-second scan | Stored evidence |
| --- | --- |
| Native C++ `graph_sampler` is **111.177×** faster than a matching Python reference (median) | [`results/sampler_timing.md`](results/sampler_timing.md) |
| GraphSAGE **beats** same-protocol MF on single-seed ML-100K **Recall@10** (0.047720 vs 0.038176, seed 7) | [`results/gnn_vs_mf_movielens_100k.md`](results/gnn_vs_mf_movielens_100k.md) |
| Demo CLI loads a checkpoint and prints top-K | [`scripts/demo_recommend.py`](scripts/demo_recommend.py) |

Those figures are copied from committed `results/` (and the existing README tables below). They are **not** rounded further, **not** SOTA, and **not** production latency. Sampler timing is a synthetic CSR on one machine. The Recall@10 pair is Phase 5 single-seed provenance (in-repo mean layers); the 5-seed PyG `SAGEConv` leaderboard is a separate table.

Public source: [github.com/sathariels/SageRec](https://github.com/sathariels/SageRec).

## Systems story

1. **C++ bipartite CSR + seeded neighbor sampler** — uniform without replacement (ADR-005).
2. **pybind11 `graph_sampler`** → **`sagerec_minibatch`** — the Python training data path calls native `sample_neighbors`.
3. **GraphSAGE** — PyG `SAGEConv` on those native neighborhoods (ADR-006; not a `NeighborLoader`).
4. **Implicit MF baseline** — same split, eligible users, candidates, Recall@10 / NDCG@10 (ADR-004).
5. **5-seed leaderboard** — seeds 7, 11, 13, 17, 19; mean ± sample std (ADR-007).
6. **Demo CLI** — load a GraphSAGE or PairScorer checkpoint, score candidate movie IDs, print top-K (ADR-008).

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
    G --> K[Demo CLI top-K]
    H --> K
    D --> J[C++ vs Python timing charts]
    D --> R[Python reference sampler]
```

Try the demo (needs a local `sagerec_checkpoint`; 100K weights are gitignored):

```bash
PYTHONPATH=build:python python3 scripts/demo_recommend.py \
  --checkpoint results/checkpoints/graphsage.pt \
  --user 0 \
  --candidates 1,2,3,4,5 \
  --k 10
```

## Native vs Python reference sampler timing

Parity is checked on the same ADR-005 without-replacement queries before any timed repetition. The committed artifacts use a deterministic synthetic bipartite graph (512 users, 1024 movies, degree 32, `k=10`, 4000 queries, seed 7) so default CI never downloads MovieLens.

```bash
PYTHONPATH=build:python python3 scripts/run_sampler_timing.py
```

Outputs: [`results/sampler_timing.json`](results/sampler_timing.json), [`.md`](results/sampler_timing.md), [`.svg`](results/sampler_timing.svg).

Headline latency, throughput, and speedup use the **median** of measured repetitions after warm-up. Optional `--source movielens-100k` times a train-only CSR from existing `data/processed/` artifacts and does not download.

Copied from the stored JSON (Release, GNU 13.3.0, Python 3.12.3, 4-way Xeon; seed 7):

| Sampler | Median seconds | Median queries/s |
| --- | ---: | ---: |
| native `graph_sampler` | 0.003911 | 1,022,638 |
| Python reference | 0.434863 | 9,198 |

Median speedup (reference / native): **111.177×**. This is one-machine sampler evidence, not a production latency or SOTA claim.

## MovieLens 100K GraphSAGE vs MF (single-seed)

Shared protocol: ADR-003 leave-one-out, test split, Recall@10 / NDCG@10, 943 eligible users, seed **7**. GraphSAGE uses a modest CPU-friendly set (`embedding_dim=16`, `hidden_dim=16`, two layers, fanouts `(8, 8)`, **3 Adam epochs**, `batch_size=256`, `learning_rate=0.01`, 2 negatives). Current training applies PyG `SAGEConv` to native mini-batches. MF used **1 SGD epoch**; fairness is the shared eval protocol, not identical wall-clock or optimizer. Extra training seeds are derived from 7 (`derived_sample_seed(seed, epoch, step)` per mini-batch; hop/source mixing inside multi-hop). Ranking encodes each graph node once at seed 7 via native sampling, then dots cached embeddings.

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

## MovieLens 100K multi-seed leaderboard

ADR-007 publishes uncertainty on the **same** split, eligible users, candidate protocol, and Recall@10 / NDCG@10 as Phase 5. Both models use seeds **7, 11, 13, 17, 19** (seed 7 kept for continuity). GraphSAGE is the current Phase 6 path: PyG `SAGEConv` on native `graph_sampler` neighborhoods via `sagerec_minibatch` (not a NeighborLoader). MF is still **1 SGD epoch**; GraphSAGE is still **3 Adam epochs**. Fairness is the shared eval protocol, not matched wall-clock or optimizer budget.

```bash
PYTHONPATH=build:python python3 scripts/run_multiseed_gnn_vs_mf.py \
  --processed-dir data/processed --raw-dir data/raw
```

Default CI does **not** run that 100K job. Outputs (measured, not invented):

- [`results/multiseed_gnn_vs_mf_movielens_100k.json`](results/multiseed_gnn_vs_mf_movielens_100k.json)
- [`results/multiseed_gnn_vs_mf_movielens_100k.md`](results/multiseed_gnn_vs_mf_movielens_100k.md)
- [`results/multiseed_gnn_vs_mf_movielens_100k.svg`](results/multiseed_gnn_vs_mf_movielens_100k.svg)

Copied from the stored JSON (seeds 7, 11, 13, 17, 19; 943 eligible users; mean ± sample std, n-1):

| Model | Epochs | Recall@10 mean±std | Recall@10 median | NDCG@10 mean±std | NDCG@10 median |
| --- | ---: | ---: | ---: | ---: | ---: |
| implicit MF | 1 SGD | 0.039024 ± 0.006848 | 0.038176 | 0.020082 ± 0.002658 | 0.020303 |
| GraphSAGE | 3 Adam | 0.044963 ± 0.007742 | 0.042418 | 0.020948 ± 0.005239 | 0.020354 |

This is not a SOTA claim. GraphSAGE is PyG `SAGEConv` on native `graph_sampler` neighborhoods. Phase 5 single-seed JSONs were not overwritten; seed-7 MF here matches that historical MF run, while seed-7 GraphSAGE is a new SAGEConv measurement (not the Phase 5 in-repo mean-layer number). Mean Recall@10 is higher for GraphSAGE, but the sample-std bands overlap — do not treat this table as a significance test.

## Demo CLI (ADR-008)

Load a schema-v1 `sagerec_checkpoint`, score caller-supplied candidate movie IDs for one user, and print top-K. Ranking is **score descending**; ties break on **smaller `movie_id`**. GraphSAGE serve-time neighborhoods still come from native `graph_sampler` via `sagerec_minibatch` (not a NeighborLoader). The CLI does **not** download MovieLens, export a batch, or start an HTTP server.

Save a checkpoint after training (synthetic example; 100K weights are gitignored):

```python
import sagerec_serve as serve
serve.save_checkpoint("results/checkpoints/graphsage.pt", model)
```

Then:

```bash
PYTHONPATH=build:python python3 scripts/demo_recommend.py \
  --checkpoint results/checkpoints/graphsage.pt \
  --user 0 \
  --candidates 1,2,3,4,5 \
  --k 10
```

Stdout is a TSV table: `rank`, `movie_id`, `score`. GraphSAGE checkpoints embed train-only pairs so the native CSR can be rebuilt. If those pairs are omitted, pass `--train-pairs path.json` or `--processed-dir data/processed` (existing artifacts only). Missing or corrupt checkpoints exit nonzero with an actionable error.

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
| Demo CLI | [`python/sagerec_serve.py`](python/sagerec_serve.py) + [`scripts/demo_recommend.py`](scripts/demo_recommend.py) — schema-v1 checkpoint load, candidate scores, top-K print (ADR-008) |
| Comparison writer | [`python/sagerec_compare.py`](python/sagerec_compare.py) — MF vs GraphSAGE JSON, markdown table, SVG chart from stored results |
| Multi-seed aggregator | [`python/sagerec_multiseed.py`](python/sagerec_multiseed.py) — per-seed rows plus mean±sample std (ADR-007) |
| Sampler timing | [`python/sagerec_sampler_benchmark.py`](python/sagerec_sampler_benchmark.py) — native vs reference `sample_neighbors` JSON/SVG |

Callers must pass **training-positive** `local_pairs()` into `BipartiteCSR`. The reference sampler reads CSR `offsets`/`neighbors`; it does not build graphs or ingest ratings files.

Layout: [`cpp/`](cpp/) native core, [`python/`](python/) prep/baseline/metrics/mini-batch/GraphSAGE/serve/comparison/multi-seed/benchmark/tests, [`docs/`](docs/) contracts and ADRs, [`data/`](data/) schemas (no raw dataset), [`results/`](results/) for metrics and charts, [`scripts/`](scripts/) thin download/prep/MF/GraphSAGE/timing/multi-seed/demo-CLI launchers.

**In this repo:** C++ CSR + seeded sampler, MovieLens 100K parser/download/prep, native mini-batch GraphSAGE (`SAGEConv`), implicit MF, shared evaluator, single-seed and 5-seed comparisons, sampler timing charts, demo CLI, Release CMake + GitHub Actions CI.

**Not in this repo:** PyG NeighborLoader / ClusterLoader (rejected for primary experiments), batch export, local HTTP API, MovieLens 1M, GCN, node2vec.

The MF ranking smoke unittest is a tiny synthetic path. MovieLens 100K implicit-MF and GraphSAGE metrics (Recall@10 / NDCG@10) with split, seed, hyperparameter, and eligibility provenance are stored under [`results/`](results/). Phase 5 files are **single-seed** historical provenance. The ADR-007 leaderboard is a separate 5-seed SAGEConv measurement with mean±sample std. The MF JSON is not a GraphSAGE result.

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
| ADR-008 | Accepted | Demo CLI (load checkpoint, score candidates, print top-K); not batch export or HTTP |

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

CTest runs native CSR/sampler/parser tests and Python unittest discovery (bindings, sampler parity, leave-one-out leakage, download/prep fixtures, MF ranking smoke, mini-batch native sampling, GraphSAGE training smoke, sampler timing schema/parity, multi-seed aggregation, demo-CLI checkpoint/top-K). After a successful build:

```bash
PYTHONPATH=build:python python3 -m unittest discover -s python/tests -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_mf_baseline.py -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_minibatch_native.py -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_graphsage_train.py -v
PYTHONPATH=build:python python3 scripts/run_graphsage_synthetic_smoke.py
PYTHONPATH=build:python python3 -m unittest python/tests/test_gnn_vs_mf_compare.py -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_multiseed_gnn_vs_mf.py -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_sampler_benchmark.py -v
PYTHONPATH=build:python python3 -m unittest python/tests/test_serve_demo.py -v
```

Default CI does **not** download MovieLens. Set `SAGEREC_LIVE_MOVIELENS=1` only for the optional live-archive test.

`import graph_sampler` loads the compiled extension. Construction takes local `(user_id, movie_id)` pairs. Do not vendor MovieLens data.

## Whiteboard follow-ups

These are real contracts, just not the first bullets a recruiter needs:

- Users occupy `[0, num_users)`; movies occupy `[num_users, num_users + num_movies)`.
- Each training interaction is stored in both directions in the sampling CSR.
- The sampling graph is **train-positives only**; held-out edges never enter preprocessing, message passing, negatives, or candidate filtering.
- Duplicate edges are dropped; neighborhoods are sorted by ascending global node ID.
- Sampling is uniform **without replacement**, Fisher–Yates prefix, per-call `std::mt19937_64` from an explicit seed.

The mini-batch helper in [`python/sagerec_minibatch.py`](python/sagerec_minibatch.py) calls `graph_sampler` for seeded multi-hop neighborhood expansion. GraphSAGE training in [`python/sagerec_graphsage.py`](python/sagerec_graphsage.py) converts those native hops into PyG `edge_index` tensors and applies `SAGEConv`. It does not use a PyG neighbor sampler.

## Honesty

- No production users, no invented metrics, and no production-latency or SOTA speedup claims.
- Tiny synthetic MF and GraphSAGE smokes are protocol verification, not MovieLens 100K evaluation.
- A single-seed 100K MF run is recorded under `results/mf_movielens_100k.json`. It is not GraphSAGE. It remains Phase 5 single-seed provenance.
- A single-seed 100K GraphSAGE run is recorded under `results/graphsage_movielens_100k.json`. Neighborhoods come from `graph_sampler` via `sagerec_minibatch`, not a PyG NeighborLoader. Those stored numbers were measured with the Phase 5 in-repo mean layers; they are not a SAGEConv 100K rerun.
- The Phase 5 comparison table/chart is generated from those two JSON files.
- The ADR-007 5-seed SAGEConv leaderboard is stored under `results/multiseed_gnn_vs_mf_movielens_100k.json` (mean ± sample std). It does not overwrite the Phase 5 files. It is not a SOTA claim.
- Native vs Python reference sampler timings are stored under `results/sampler_timing.json` (synthetic graph, median of measured repetitions). They are one-machine evidence, not a production latency or SOTA claim.
- Production message passing is PyG `SAGEConv` on native mini-batches (ADR-006). PyG NeighborLoader is not the neighborhood source.
- The ADR-008 demo CLI loads a local checkpoint and prints top-K; it is not an HTTP API, not a batch exporter, and not a production latency claim.
- This GitHub repository is the public homepage. Do not treat an Origin (or other private) URL as the project home.

Acceptance criteria and component contracts: [`docs/project-requirements.md`](docs/project-requirements.md), [`docs/architecture.md`](docs/architecture.md). Contributors: read [`AGENTS.md`](AGENTS.md) before changing code.
