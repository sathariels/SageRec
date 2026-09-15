# Results Layout

Machine-readable run summaries belong here.

- `mf_movielens_100k.json` is a single-seed implicit-MF run on the official
  MovieLens 100K ADR-003 split (seed 7, 1 epoch). It is not a GraphSAGE
  result and not a published leaderboard.
- `graphsage_movielens_100k.json` is the matching single-seed GraphSAGE run
  (seed 7, native `sagerec_minibatch` neighborhoods). Not a multi-seed
  leaderboard. Those metrics were measured with the Phase 5 in-repo mean
  layers; do not retcon them as a Phase 6 SAGEConv rerun.
- `gnn_vs_mf_movielens_100k.json`, `.md`, and `.svg` are the Phase 5
  comparison artifacts. Values are copied from the two result files.

- `multiseed_gnn_vs_mf_movielens_100k.json`, `.md`, and `.svg` are the
  ADR-007 5-seed leaderboard (seeds 7, 11, 13, 17, 19). Values are
  measured per-seed then aggregated (mean ± sample std). Do not mix
  these with the Phase 5 single-seed files or invent numbers.

- `sampler_timing.json`, `.md`, and `.svg` are the Phase 2 native vs
  Python reference neighbor-sampler timings on a deterministic synthetic
  graph (ADR-005 without replacement). Values are measured, not invented.
  Optional live MovieLens 100K timing is a script flag and is not the
  committed default.

Do not invent placeholder metrics or hand-edit timings. Checkpoints and
bulky logs stay ignored.
