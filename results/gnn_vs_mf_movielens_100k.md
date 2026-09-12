# MovieLens 100K GraphSAGE vs implicit MF

Same ADR-003 split, eligible users, candidate protocol, and Recall@10 / NDCG@10. Single-seed comparison, not a multi-seed leaderboard. Fairness is the shared eval protocol, not identical wall-clock or epoch count (MF n_epochs=1, GraphSAGE n_epochs=3).

| Model | Seed | Epochs | Recall@10 | NDCG@10 | Evaluated users | Result file |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| implicit MF | 7 | 1 | 0.038176 | 0.020303 | 943 | `results/mf_movielens_100k.json` |
| GraphSAGE | 7 | 3 | 0.047720 | 0.020969 | 943 | `results/graphsage_movielens_100k.json` |

GraphSAGE neighborhoods come from `graph_sampler` via `sagerec_minibatch.NativeMinibatchSampler` (ADR-005 uniform without replacement). This comparison does not use a PyG NeighborLoader.

Values are copied from the stored result files. Do not treat this as a published multi-seed leaderboard.
