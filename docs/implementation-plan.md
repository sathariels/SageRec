# Phased Implementation Plan

ADR-001 (MovieLens 100K), ADR-002 (GraphSAGE), ADR-003 (per-user
chronological leave-one-out), ADR-004 (matrix factorization), ADR-005
(uniform sampling without replacement), ADR-006 (PyG SAGEConv on
native neighborhoods), ADR-007 (5-seed MF vs GraphSAGE leaderboard),
ADR-008 (demo CLI serving), ADR-009 (HTTP serving on precomputed
embeddings), and ADR-010 (two-stage retrieval and re-ranking) are
accepted. MovieLens 1M remains
deferred. Phase 2 is delivered/complete for MovieLens 100K: download
and on-disk `processed/` prep, and timing charts (native vs Python
reference sampler, stored JSON/SVG). Phase 4 GraphSAGE training on the
native-backed mini-batch harness is delivered. Phase 5 is delivered for
the single-seed 100K GraphSAGE run and GNN-versus-MF comparison
(historical single-seed provenance; not overwritten by later multi-seed
artifacts). Phase 6 is delivered for PyG `SAGEConv` message passing
still fed by `graph_sampler` / `sagerec_minibatch`. Phase 7 is delivered
for the 5-seed MovieLens 100K MF vs GraphSAGE leaderboard with mean±std.
Phase 8 is delivered for the ADR-008 demo CLI (checkpoint load, candidate
scoring, top-K print) with synthetic harness tests. Phase 9 is delivered
for the ADR-009 HTTP API: offline embedding export, FastAPI
`GET /recommend` and `/healthz`, NumPy top-K, pytest `TestClient`
coverage, and a measured `results/serving_latency.*` load test.
Phase 10 is delivered for ADR-010: a NumPy MLP re-ranker on the stage-1
top-N pool, MovieLens 100K retrieval-only versus retrieval-plus-re-rank
under `results/rerank_eval.json`, optional `rerank` on `GET /recommend`,
and load-test p50/p95/p99 for both modes (`serving-latency-v2`).

## Phase 1: Native foundation

Completed in the first verified slice:

- Define native public contracts and CMake targets.
- Implement validated train-only CSR construction from explicit interactions.
- Implement seeded sampling using the ADR-005 without-replacement contract.
- Add native tests for construction, edge cases, and seed reproducibility.
- Add thin pybind11 bindings, Python binding tests, and Release CI.

Completed in the ingestion slice:

- Native MovieLens 100K `u.data` parser (`parse_movielens_100k`) with
  deterministic zero-based mappings and preserved rating/timestamp.
- Actionable `GraphError` for empty input, malformed rows/fields/types,
  nonpositive source IDs, and duplicate source user-movie pairs.
- Parser stays separate from CSR; callers feed `local_pairs()` into
  `BipartiteCSR` after they have selected training positives.
- Native parser tests use tiny in-memory strings only.

Completed in the parser-binding slice:

- `graph_sampler.parse_movielens_100k` and `MovieLens100kRatings` pybind11
  surface with matching `.pyi` stubs.
- Python binding smoke tests use tiny in-memory `u.data` strings only:
  mappings, `local_pairs()` CSR handoff, and malformed-input `GraphError`.

Still not started in Phase 1:

- Do not add MovieLens 1M parsers, paths, or downloads. Download and
  on-disk prep belong to the opened Phase 2 slice, not the native parser.

Exit condition: clean Release build, CTest pass, extension import and smoke test pass.

## Phase 2: Data and benchmark harness

Opened in the reference-sampler slice:

- Naive Python sampler in `python/sagerec_reference_sampler.py` matching the
  native `sample_neighbors` contract (Fisher–Yates prefix, `std::mt19937_64`,
  unbiased `uniform_below`).
- Unittest parity cases against compiled `graph_sampler` on synthetic graphs.

Opened in the ADR-003 split/prep slice:

- In-memory leave-one-out assignment in `python/sagerec_prep.py` from
  already-normalized 100K interactions.
- Train-only `(user_id, movie_id)` pairs for `BipartiteCSR`.
- Manifest schema (`data/processed/manifest.schema.json`) and builder.
- Leakage, eligibility, timestamp-order, cold-start, and determinism tests.

Opened in the Phase 2 download / on-disk prep slice:

- Official GroupLens MovieLens 100K download (`python/sagerec_download.py`)
  with published archive MD5 verification and `u.data` extraction under
  `data/raw/` (gitignored).
- Path/bytes ingestion through `graph_sampler.parse_movielens_100k` and
  ADR-003 `sagerec_prep`, writing `data/processed/` artifacts plus a filled
  manifest (`python/sagerec_dataset.py`).
- Thin scripts: `scripts/download_movielens_100k.py`,
  `scripts/prepare_movielens_100k.py`.
- Fixture tests for checksum, zip layout, path ingestion, and manifest
  writes. Live download is optional and skipped in default CI.

Opened in the Phase 2 timing-chart slice:

- Native vs Python reference `sample_neighbors` benchmark
  (`python/sagerec_sampler_benchmark.py`) with ADR-005 without-replacement
  parity required before any timed repetition.
- Warm-up plus multiple measured repetitions; headline statistic is the
  median. Provenance records graph size, workload, `k`, replacement
  policy, seed, build type, compiler, Python, CPU, and speedup computed
  from measured seconds only.
- Thin launcher `scripts/run_sampler_timing.py` writes
  `results/sampler_timing.json`, `.md`, and `.svg`. Default graph is
  synthetic. `--source movielens-100k` uses existing train-only
  `processed/` artifacts and never downloads.
- Unittests cover parity-before-timing and the required JSON schema on a
  tiny synthetic graph. Default CI does not time MovieLens 100K.

Exit condition: selected dataset prepares reproducibly and benchmark evidence is complete.

## Phase 3: Baseline

Opened in the ADR-004 matrix-factorization slice:

- Common `PairScorer` interface (`python/sagerec_scoring.py`); models expose
  ranking scores only. Metric logic lives in the shared evaluator.
- Implicit-feedback MF baseline (`python/sagerec_baseline.py`): seeded NumPy
  logistic SGD with configurable factors, epochs, learning rate, L2, and
  negatives-per-positive.
- Training negative sampling (`python/sagerec_negatives.py`) that excludes
  known positives in the training scope.
- Shared ranking evaluator (`python/sagerec_metrics.py`): per-user
  Recall@10 and NDCG@10 with macro averaging, ADR-003 eligibility, and
  train (plus validation when scoring test) candidate filtering.
- Tiny deterministic unittest smoke: synthetic interactions →
  `sagerec_prep` leave-one-out → brief seeded MF → finite metrics in
  `[0, 1]` that match across two runs. Leakage checks confirm held-out
  positives are absent from the MF train edge set / train CSR.

Still not started:

- node2vec (rejected unless a superseding ADR accepts it).
- The Phase 5 GNN-versus-baseline table is generated from stored
  100K result files (do not invent numbers).
- Do not invent MovieLens 100K quality numbers. A real single-seed 100K MF
  run is stored under `results/mf_movielens_100k.json` when generated from
  the official archive (not a GraphSAGE or multi-seed leaderboard).

Exit condition: baseline produces reproducible Recall@10 and NDCG@10 on
the tiny synthetic path (verification only, not a MovieLens 100K result).

## Phase 4: GNN

Opened in the mini-batch harness slice:

- Native-backed neighborhood helper (`python/sagerec_minibatch.py`) that
  builds a train-only `BipartiteCSR` from local pairs and expands seeded
  multi-hop neighborhoods by calling `graph_sampler.sample_neighbors`
  (ADR-005 without replacement).
- Deterministic tests (`python/tests/test_minibatch_native.py`) on a tiny
  synthetic graph: compiled-extension requirement, native-call spying
  (reference sampler must not be used), seed reproducibility, empty /
  `k = 0` / full-neighborhood cases, and train-only CSR leakage.

Opened in the GraphSAGE training slice:

- PyTorch GraphSAGE trainer (`python/sagerec_graphsage.py`) that implements
  `PairScorer` and calls `NativeMinibatchSampler` for every neighborhood
  expansion. CPU-only `torch==2.6.0` is pinned in
  `python/requirements-train.txt`.
- Primary experiments must not use a PyG NeighborLoader or other Python
  sampler. Phase 6 replaces the in-repo mean layers with PyG `SAGEConv`
  while keeping this native neighborhood contract.
- Tiny synthetic ADR-003 smoke (`python/tests/test_graphsage_train.py`):
  native `sample_neighbors` spy, train-only CSR leakage, negatives exclude
  known positives, and seeded Recall@10 / NDCG@10 that stay finite in
  `[0, 1]` and match across two runs. Those metrics are protocol smoke,
  not a MovieLens 100K result.

Exit condition: reproducible tiny GraphSAGE training smoke exists and
proves native sampling is on the training path. Phase 5 owns the 100K
GNN-versus-baseline report. Phase 6 owns the PyG conv stack.

## Phase 5: Comparison and documentation

Opened in the MovieLens 100K GraphSAGE quality slice:

- Runnable 100K train+eval path (`scripts/run_graphsage_movielens_100k.py`)
  using existing download/prep, train-only CSR, native mini-batch
  GraphSAGE, and shared `PairScorer` Recall@10 / NDCG@10.
- Modest CPU-friendly hyperparams in `movielens_100k_config` (seed 7 to
  match the stored MF run; 3 Adam epochs, documented as not wall-clock
  or optimizer-matched to MF's 1 SGD epoch).
- Provenance JSON `results/graphsage_movielens_100k.json` (single-seed,
  not a multi-seed leaderboard).
- Comparison writer `python/sagerec_compare.py` plus
  `scripts/write_gnn_vs_mf_comparison.py`: machine-readable JSON,
  markdown table, and SVG chart generated from the stored MF and
  GraphSAGE files (no invented metrics).
- Tests for comparison schema/protocol match and native 100K wiring.

Historical single-seed artifacts remain labeled as such. The multi-seed
leaderboard is Phase 7 (`results/multiseed_gnn_vs_mf_movielens_100k.*`).

Exit condition: stored 100K GraphSAGE metrics with provenance, comparison
table/chart consistent with both result files, and CI green (CTest +
Python tests). Phase 2 timing charts are delivered separately under
`results/sampler_timing.*`. Phase 6 does not retcon those 100K numbers.

## Phase 6: PyG SAGEConv on native samples

Opened in the PyG production-path slice (ADR-006):

- GraphSAGE message passing uses `torch_geometric.nn.SAGEConv` (mean
  aggregation) in `python/sagerec_graphsage.py`.
- A thin adapter (`neighborhood_hop_to_edge_index`) converts native
  multi-hop `NeighborhoodBatch` hops into the `edge_index` form SAGEConv
  expects. It does not sample.
- Neighborhoods still come only from `graph_sampler` via
  `sagerec_minibatch.NativeMinibatchSampler`. Tests spy native
  `sample_neighbors` and forbid `NeighborLoader` / `ClusterLoader` on the
  primary path.
- CPU-friendly pins: `torch==2.6.0` and `torch-geometric==2.6.1` in
  `python/requirements-train.txt`. Optional PyG extension wheels are not
  required for mean SAGEConv and are not installed in default CI. Native
  C++ stays free of PyTorch/PyG.

Still not started:

- PyG NeighborLoader / ClusterLoader as a neighborhood source (rejected
  for primary experiments).

The measured SAGEConv MovieLens 100K quality numbers live in the Phase 7
multi-seed artifacts. Do not invent or hand-edit
`results/graphsage_movielens_100k.json` or the Phase 5 comparison files.
Tiny synthetic metrics remain protocol smoke.

Exit condition: training path uses SAGEConv + native neighborhoods,
harness tests prove the native sampler is called and NeighborLoader is
not, docs/requirements match, and Release CTest / unittest / CI stay
green on CPU.

## Phase 7: Multi-seed MF vs GraphSAGE leaderboard

Opened in the ADR-007 multi-seed slice:

- Same seed list for implicit MF and GraphSAGE: **7, 11, 13, 17, 19**
  (seed 7 kept for continuity with Phase 5).
- Shared ADR-003 split, eligible users, candidate construction, and
  Recall@10 / NDCG@10 via `PairScorer`.
- GraphSAGE stays on PyG `SAGEConv` + native `sagerec_minibatch` /
  `graph_sampler` (ADR-006). No NeighborLoader.
- Aggregator `python/sagerec_multiseed.py` computes per-seed rows plus
  mean, sample std (n-1), and median. Thin launcher
  `scripts/run_multiseed_gnn_vs_mf.py` writes
  `results/multiseed_gnn_vs_mf_movielens_100k.{json,md,svg}`.
- Historical Phase 5 files are not overwritten and stay labeled
  single-seed.
- Tests cover aggregation math, schema, historical-file labels, native
  wiring, and a tiny synthetic two-seed smoke. Default CI does not
  download MovieLens or run the 100K multi-seed job.

Exit condition: measured 5-seed JSON/markdown/SVG committed from a real
run, docs record ADR-007, and the fast CI path stays green.

## Phase 8: Demo CLI serving

Opened in the ADR-008 demo-CLI slice:

- Schema-v1 checkpoint (`sagerec_checkpoint`) with model name,
  hyperparams, `state_dict`, and GraphSAGE train-only pairs so serve can
  rebuild the native CSR. Implicit MF is PairScorer-compatible.
- `python/sagerec_serve.py` loads the checkpoint, scores caller-supplied
  candidate movie IDs for one user through `PairScorer`, and ranks top-K
  (score descending, `movie_id` ascending ties). GraphSAGE neighborhoods
  still come from `graph_sampler` / `sagerec_minibatch` (no NeighborLoader).
- Thin launcher `scripts/demo_recommend.py` (`--checkpoint`, `--user`,
  `--candidates`, `--k`). Optional `--train-pairs` / `--processed-dir`
  when a GraphSAGE checkpoint omits pairs. No auto-download.
- Harness tests: missing/corrupt checkpoint → actionable nonzero exit;
  deterministic top-K; finite scores; documented tie-break; synthetic
  temp-dir graphs only. Default CI does not download MovieLens.

Still not started in this slice:

- Batch export of scores (still rejected; ADR-009 does not accept it).
- MovieLens 1M, GCN, node2vec, NeighborLoader.

HTTP recommend moved to Phase 9 under ADR-009. The demo CLI remains.

Exit condition: ADR-008 recorded, demo CLI + tests on the default CI
path (CTest + Python unittest), and docs match the checkpoint/CLI
contract. Do not invent or edit stored 100K quality result files.

## Phase 9: HTTP serving API

Opened in the ADR-009 slice. ADR-009 supersedes ADR-008's no-HTTP
scope. The demo CLI stays.

- Offline export (`python/sagerec_export.py`,
  `scripts/export_embeddings.py`) writes a `sagerec_serving_artifact`
  version 1 `.npz` from a schema-v1 checkpoint. GraphSAGE reuses
  `GraphSAGERecommender.eval_embedding_matrices` (native
  `graph_sampler` via `sagerec_minibatch`, once, at `config.seed`).
  Implicit MF copies the factors and biases `score_pairs` already
  uses and requires train pairs so already-seen items can be masked.
- FastAPI service (`python/sagerec_api.py`, `scripts/serve_api.py`):
  `GET /healthz` and `GET /recommend?user_id=&k=`. Rows are item id,
  title, and score. Train positives are excluded. Unknown users return
  404. Invalid queries return 422. Top-K is exact NumPy
  (`python/sagerec_serving.py`). FAISS is not required.
- `Dockerfile` serves the artifact only (no training stack). `Makefile`
  documents export, serve, load test, pytest, and docker targets.
- Load test (`python/sagerec_loadtest.py`,
  `scripts/loadtest_serving.py`) writes measured p50/p95/p99 and
  throughput to `results/serving_latency.json` and `.md`, with CPU and
  dependency provenance. The committed run uses a synthetic catalog
  whose user/item counts match public MovieLens 100K cardinality.
  Embeddings are seeded noise. Do not hand-edit those numbers and do
  not treat them as ranking quality.
- Pytest + FastAPI `TestClient` on a tiny fixture
  (`python/tests/test_serving_api.py`), wired into CTest as
  `python_serving_api`. Default CI does not download MovieLens, does
  not regenerate the latency file, and does not bind a public port.

Still not started in the Phase 9 slice (moved or still rejected):

- Optional re-ranking of the same embeddings (Phase 10 / ADR-010).
- Batch export.
- MovieLens 1M, GCN, node2vec, NeighborLoader.
- FAISS as a required index.

Exit condition: ADR-009 recorded, export and API tests on the default
CI path, measured latency files committed from a real run, and docs
match the artifact and HTTP contract. Do not invent or edit stored
100K quality result files.

## Phase 10: Two-stage retrieval and re-ranking

Opened in the ADR-010 slice. Stage 1 stays the ADR-009 embedding
retriever. Stage 2 is optional.

- `python/sagerec_rerank.py` builds train-only features (retrieval
  score, `log1p` popularity, genre affinity from `u.item`, item
  recency), trains a NumPy MLP with one ReLU hidden layer and a linear
  skip by listwise softmax, and orders the stage-1 pool. It does not
  import PyTorch, PyG, LightGBM, or `graph_sampler`.
- `python/sagerec_rerank_eval.py` and `scripts/run_rerank_eval.py` fit
  implicit MF (`movielens_100k_config`, seed 7) as stage 1, fit the MLP
  on validation items retrieved inside top-N (default 200), and compare
  both systems on the test split of that same pool. Metrics are
  Recall@10, NDCG@10, and MAP@10. Output is `results/rerank_eval.json`
  and `.md`. This is a single-seed MF retrieval measurement. It does
  not replace Phase 5 or the ADR-007 leaderboard and it is not a
  GraphSAGE result. Training pools exclude the test item; test pools
  exclude the validation item. The re-ranker cannot recover items
  outside top-N.
- `GET /recommend` accepts `rerank` (default false) and `n` (stage-1
  width). `rerank=true` needs `SAGEREC_RERANK_ARTIFACT` whose
  user/item counts and `stage1_model` match the serving artifact.
  Unknown users stay 404. Missing artifact or `n < k` is 422. The
  request does not train or sample.
- The load test writes protocol `serving-latency-v2` with p50/p95/p99
  and throughput for `retrieval_only` and `retrieval_plus_rerank`.
  Top-level `latency_ms` duplicates retrieval-only. The committed
  catalog and the timed MLP are synthetic. Do not hand-edit them.
- Tests: `python/tests/test_rerank.py` (unittest discovery) for MAP,
  feature leakage, the MLP, and artifact order; API tests in
  `python/tests/test_serving_api.py` for the flag, `n`, and startup
  mismatch. Default CI does not download MovieLens, does not run the
  100K re-rank job, and does not regenerate `results/serving_latency.*`.

Still rejected:

- LightGBM (or any extra native library) on the request path.
- Batch export, MovieLens 1M, GCN, node2vec, NeighborLoader, FAISS.

Exit condition: ADR-010 recorded, feature/MLP/API tests on the default
CI path, measured `results/rerank_eval.*` and regenerated
`results/serving_latency.*` committed from real runs, and docs match
the two-stage contract. Do not invent metrics or retcon Phase 5 or
Phase 7 files.

