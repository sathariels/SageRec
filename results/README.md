# Results Layout

Machine-readable run summaries belong here. `mf_movielens_100k.json` is a
single-seed implicit-MF run on the official MovieLens 100K ADR-003 split, with
dataset URL, checksum, hyperparameters, eligibility counts, Recall@10, and
NDCG@10. It is not a GraphSAGE result and not a published leaderboard.

Future additions may include:

- a GNN-versus-baseline table for Recall@10 and NDCG@10;
- a generated recommendation-quality chart;
- a C++-versus-Python sampler benchmark table and chart.

Do not invent placeholder metrics. Checkpoints and bulky logs stay ignored.
