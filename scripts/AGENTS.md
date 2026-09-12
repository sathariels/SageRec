# Workflow Script Agent Guide

Scripts are thin, reproducible entry points rather than homes for business logic.

- Fail fast and propagate nonzero exit codes.
- Resolve paths from the repository or explicit arguments, never a developer home path.
- Accept configuration, seed, dataset path, output path, and device explicitly.
- Print the effective command/config and versions needed for provenance.
- Delegate implementation to tested native or Python modules.
- Avoid implicit destructive cleanup and environment mutation.
- Document each supported script in the root README.

Opened Phase 2/3 scripts (MovieLens 100K only):

- `download_movielens_100k.py` — fetch the official zip, verify MD5, extract `u.data`.
- `prepare_movielens_100k.py` — ADR-003 on-disk prep from a local `u.data` path.
- `run_mf_movielens_100k.py` — seeded implicit MF on prepared artifacts.

Opened Phase 4 GraphSAGE synthetic smoke:

- `run_graphsage_synthetic_smoke.py` — tiny ADR-003 GraphSAGE protocol
  smoke using native `sagerec_minibatch`. Prints labeled protocol-smoke
  metrics; do not write MovieLens 100K GNN result files.

Opened Phase 5 100K GraphSAGE + comparison:

- `run_graphsage_movielens_100k.py` — download/prep if needed, train
  GraphSAGE on the native mini-batch harness (seed 7), evaluate with
  shared `PairScorer` metrics, write `results/graphsage_movielens_100k.json`.
- `write_gnn_vs_mf_comparison.py` — read stored MF + GraphSAGE JSONs and
  write comparison JSON, markdown table, and SVG chart. Does not invent
  metrics.

Do not add MovieLens 1M or timing-benchmark scripts. The Phase 4
mini-batch helper lives in `python/sagerec_minibatch.py`. GraphSAGE
training lives in `python/sagerec_graphsage.py`.
