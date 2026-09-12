# Results Layout

Machine-readable run summaries belong here.

- `mf_movielens_100k.json` is a single-seed implicit-MF run on the official
  MovieLens 100K ADR-003 split (seed 7, 1 epoch). It is not a GraphSAGE
  result and not a published leaderboard.
- `graphsage_movielens_100k.json` is the matching single-seed GraphSAGE run
  (seed 7, native `sagerec_minibatch` neighborhoods). Not a multi-seed
  leaderboard.
- `gnn_vs_mf_movielens_100k.json`, `.md`, and `.svg` are the Phase 5
  comparison artifacts. Values are copied from the two result files.

Timing charts and stored sampler speedups are still not present.

Do not invent placeholder metrics. Checkpoints and bulky logs stay ignored.
