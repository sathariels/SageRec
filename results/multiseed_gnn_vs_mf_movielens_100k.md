# MovieLens 100K multi-seed GraphSAGE vs implicit MF

Same ADR-003 split, eligible users, candidate protocol, and Recall@10 / NDCG@10 across seeds [7, 11, 13, 17, 19]. Fairness is the shared eval protocol, not identical wall-clock or optimizer budget (MF n_epochs=1 SGD, GraphSAGE n_epochs=3 adam). GraphSAGE uses PyG SAGEConv on native graph_sampler neighborhoods (ADR-006); not a NeighborLoader run.

Uncertainty is **mean ± sample standard deviation (n-1)** over seeds 7, 11, 13, 17, 19 (n=5). Median is also stored. Continuity seed 7 is included so Phase 5 single-seed runs stay comparable as historical provenance, not mixed into these aggregates.

| Model | Epochs | Recall@10 mean±std | Recall@10 median | NDCG@10 mean±std | NDCG@10 median | Evaluated users |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| implicit MF | 1 | 0.039024 ± 0.006848 | 0.038176 | 0.020082 ± 0.002658 | 0.020303 | 943 |
| GraphSAGE | 3 | 0.044963 ± 0.007742 | 0.042418 | 0.020948 ± 0.005239 | 0.020354 | 943 |

## Per-seed rows

| Seed | Model | Epochs | Recall@10 | NDCG@10 |
| ---: | --- | ---: | ---: | ---: |
| 7 | implicit MF | 1 | 0.038176 | 0.020303 |
| 7 | GraphSAGE | 3 | 0.042418 | 0.020354 |
| 11 | implicit MF | 1 | 0.038176 | 0.018871 |
| 11 | GraphSAGE | 3 | 0.039236 | 0.016077 |
| 13 | implicit MF | 1 | 0.030753 | 0.016423 |
| 13 | GraphSAGE | 3 | 0.037116 | 0.015818 |
| 17 | implicit MF | 1 | 0.038176 | 0.021276 |
| 17 | GraphSAGE | 3 | 0.055143 | 0.027522 |
| 19 | implicit MF | 1 | 0.049841 | 0.023537 |
| 19 | GraphSAGE | 3 | 0.050901 | 0.024969 |

GraphSAGE neighborhoods come from `graph_sampler` via `sagerec_minibatch.NativeMinibatchSampler` (ADR-005 uniform without replacement). Message passing is PyG `SAGEConv` (ADR-006). This comparison does not use a PyG NeighborLoader.

Historical single-seed files (`results/mf_movielens_100k.json`, `results/graphsage_movielens_100k.json`, `results/gnn_vs_mf_movielens_100k.json`) stay Phase 5 provenance and were not overwritten.

Values are copied from measured per-seed ranking runs in this environment. Historical Phase 5 files remain single-seed provenance and were not overwritten. This is not a SOTA claim.
