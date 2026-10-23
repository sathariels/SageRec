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
  GraphSAGE on the native mini-batch harness (seed 7) with PyG SAGEConv,
  evaluate with shared `PairScorer` metrics, write
  `results/graphsage_movielens_100k.json`. Do not retcon stored Phase 5
  numbers without a measured rerun.
- `write_gnn_vs_mf_comparison.py` — read stored MF + GraphSAGE JSONs and
  write comparison JSON, markdown table, and SVG chart. Does not invent
  metrics.

Opened Phase 7 multi-seed leaderboard (ADR-007):

- `run_multiseed_gnn_vs_mf.py` — download/prep if needed, train implicit
  MF and native-backed GraphSAGE (PyG SAGEConv) across seeds 7, 11, 13,
  17, 19, write `results/multiseed_gnn_vs_mf_movielens_100k.{json,md,svg}`.
  Does not overwrite Phase 5 single-seed files. Do not run this 100K job
  in default CI.

Opened Phase 2 sampler timing:

- `run_sampler_timing.py` — verify native/reference `sample_neighbors`
  parity, then time both on the same ADR-005 workload. Writes
  `results/sampler_timing.json`, `.md`, and `.svg`. Default source is a
  deterministic synthetic graph. `--source movielens-100k` uses existing
  train-only `processed/` artifacts and never downloads.

Opened Phase 8 demo CLI (ADR-008):

- `demo_recommend.py` — load a schema-v1 checkpoint, score
  `--candidates` for `--user`, print top-K. Delegates to
  `python/sagerec_serve.py`. Does not download MovieLens, start HTTP,
  or export a batch.

Opened Phase 9 HTTP serving (ADR-009):

- `export_embeddings.py` — write a serving `.npz` from a checkpoint.
  Delegates to `python/sagerec_export.py`. Does not download MovieLens
  or bind a port.
- `serve_api.py` — serve `GET /recommend` from that artifact. Delegates
  to uvicorn and `python/sagerec_api.py`.
- `loadtest_serving.py` — measure p50/p95/p99 and throughput for
  retrieval-only and retrieval-plus-re-rank, and write
  `results/serving_latency.json` and `.md` (`serving-latency-v2`).
  Default catalog and re-ranker are synthetic. Does not download
  MovieLens. Do not hand-edit the output.

Opened Phase 10 two-stage re-ranking (ADR-010):

- `run_rerank_eval.py` — train implicit MF on MovieLens 100K, fit the
  NumPy MLP on validation labels inside the stage-1 pool, and write
  `results/rerank_eval.json` and `.md`. Requires `u.item` (extracted
  from the official zip when it is already on disk). Does not download
  on its own beyond `ensure_movielens_100k_processed`, does not start
  HTTP, and does not overwrite Phase 5 or Phase 7 files. Optional
  `--rerank-out` writes a gitignored artifact under `results/serving/`.
  Not a default CI job.

Do not add MovieLens 1M scripts. The Phase 4 mini-batch helper lives in
`python/sagerec_minibatch.py`. GraphSAGE training lives in
`python/sagerec_graphsage.py` (Phase 6: PyG SAGEConv, native
neighborhoods). Sampler timing logic lives in
`python/sagerec_sampler_benchmark.py`. Demo serving lives in
`python/sagerec_serve.py`. HTTP serving lives in `python/sagerec_api.py`
and `python/sagerec_serving.py`. Embedding export lives in
`python/sagerec_export.py`. Two-stage features and the MLP live in
`python/sagerec_rerank.py`. The MovieLens comparison lives in
`python/sagerec_rerank_eval.py`.
