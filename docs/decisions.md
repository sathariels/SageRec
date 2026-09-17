# Architecture Decision Log

Use one section per decision. Accepted entries must include date, owner, context,
choice, alternatives, rationale, and consequences.

## Decision status

| ID | Topic | Options | Status |
| --- | --- | --- | --- |
| ADR-001 | Dataset | MovieLens 100K / MovieLens 1M | Accepted: MovieLens 100K (1M deferred) |
| ADR-002 | GNN | GraphSAGE / GCN | Accepted: GraphSAGE |
| ADR-003 | Split | Per-user chronological leave-one-out / global time split | Accepted: per-user chronological leave-one-out |
| ADR-004 | Baseline | Matrix factorization / node2vec | Accepted: matrix factorization |
| ADR-005 | Sampler semantics | Uniform without replacement / with replacement | Accepted: uniform without replacement |
| ADR-006 | GraphSAGE production stack | PyG SAGEConv on native samples / PyG NeighborLoader / in-repo mean layers | Accepted: PyG SAGEConv on native `graph_sampler` neighborhoods |
| ADR-007 | Published quality comparison | Multi-seed MF vs GraphSAGE with uncertainty / keep Phase 5 single-seed only | Accepted: 5-seed MovieLens 100K leaderboard (seeds include 7) |
| ADR-008 | Production serving demo | Demo CLI / batch export / local HTTP API | Accepted: Demo CLI (load checkpoint, score candidates, print top-K) |

## Accepted

### ADR-001: MovieLens 100K (MovieLens 1M deferred)

- Date: 2026-09-02
- Owner: Nithilan Kumaran
- Status: Accepted

**Context:** Implementation of data loading, ID mapping, splits, and native
ingestion cannot start until the owner selects a MovieLens scale. The two
allowed options were MovieLens 100K and MovieLens 1M.

**Choice:** MovieLens 100K is accepted for the current project. MovieLens 1M is
explicitly deferred.

**Alternatives:** Adopt MovieLens 1M immediately, or support both editions in
parallel.

**Rationale:** The owner accepted 100K as the dataset for now. The smaller public
edition keeps the first verified native and evaluation slices tractable while
preserving the bipartite implicit-ranking setting required by the architecture.

**Consequences:**

- Parsers, docs, configs, and download instructions may target MovieLens 100K
  only (commonly tab-separated `u.data`).
- Do not add MovieLens 1M paths, configs, parser branches, or downloads until a
  superseding decision accepts 1M.
- Cold-start rules, manifests, and result provenance must record the 100K
  edition.
- ADR-003 (split) and ADR-004 (baseline) are accepted separately; this
  dataset choice does not settle either of them.

### ADR-002: GraphSAGE

- Date: 2026-09-02
- Owner: Nithilan Kumaran
- Status: Accepted

**Context:** Python training must implement exactly one owner-selected GNN
family. The two allowed options were GraphSAGE and GCN. The project's systems
contribution is a real C++ neighbor-sampling pipeline used by that GNN.

**Choice:** GraphSAGE.

**Alternatives:** GCN, or implementing both families before the first quality
comparison.

**Rationale:** The owner accepted GraphSAGE. Neighbor sampling is the native
backend's reason for existing; GraphSAGE is the family that consumes that
contract during mini-batch training.

**Consequences:**

- Phase 4 must implement GraphSAGE, not GCN.
- Do not add GCN model code, GCN configs, or GCN-versus-GraphSAGE experiment
  matrices unless a superseding decision accepts GCN.
- The native sampler remains the training neighborhood source; do not silently
  substitute a PyG neighbor sampler in primary experiments. ADR-006 records
  the production conv stack (PyG `SAGEConv`) while keeping that neighborhood
  contract.
- ADR-003 (split), ADR-004 (matrix factorization), and ADR-005
  (uniform sampling without replacement) are accepted separately.

### ADR-003: Per-user chronological leave-one-out

- Date: 2026-09-07
- Owner: Nithilan Kumaran
- Status: Accepted

**Context:** Leakage-safe train/validation/test assignment cannot start until
the owner selects a split. The two allowed options were per-user chronological
leave-one-out and a global time split. Implementation also needs documented
eligibility, cold-start handling, and a timestamp tie-break. The dataset is
MovieLens 100K only (ADR-001); MovieLens 1M remains deferred.

**Choice:** Per-user chronological leave-one-out on MovieLens 100K.

- For each **eligible** user, sort that user's interactions by
  `(timestamp, user_id, movie_id)` ascending (local IDs). Latest → **test**;
  second-latest → **validation**; all earlier → **train**.
- **Minimum interactions for eval eligibility: 3** (one train, one validation,
  and one test after leave-one-out).
- Users with **fewer than 3** interactions are **cold-start**: every
  interaction is assigned to **train**, and the user is **excluded from
  validation/test ranking eligibility**.
- The split is chronological, not randomized. Manifests record policy id
  `per_user_chronological_leave_one_out`, policy version `adr-003-v1`, and
  `seed: null`.

**Tie-break:** `parse_movielens_100k` keeps input-row order and remaps each
source ID to its rank among the sorted unique source IDs of that type. Local
IDs are therefore a strictly increasing function of source IDs, and
`(timestamp, user_id, movie_id)` matches `(timestamp, source_user_id,
source_movie_id)`. The parser rejects duplicate source user-movie pairs;
prep rejects duplicate local pairs. After the sort key, no further tie
remains. Ranking does not depend on input-row order.

**Alternatives:** A global time cutoff (one timestamp threshold for every
user), or leave-one-out without a documented cold-start rule.

**Rationale:** The owner accepted per-user chronological leave-one-out. It
matches top-N recommendation evaluation and keeps per-user coverage. The
minimum of three interactions is the smallest count that still yields a
train, validation, and test positive per eligible user. Assigning cold-start
rows to train keeps those movies in the sampling graph without letting
short-history users enter ranking eligibility.

**Consequences:**

- Python orchestration (`python/sagerec_prep.py`) assigns splits in memory
  from already-normalized interactions. Do not implement the split in the
  native parser or CSR builder.
- The training CSR must be built from training positives only. Validation and
  test positives must never enter preprocessing, message passing, negative
  sampling, or filtering.
- Manifests must record dataset edition `100k`, the split policy id/version,
  min-interaction and cold-start policies, per-split counts, and optional
  source URL / checksum placeholders. Do not require a download.
- Do not add a global time split, MovieLens 1M split paths, or on-disk
  MovieLens ingestion in this slice.
- ADR-004 (baseline) and ADR-005 (sampler semantics) are accepted separately.

### ADR-004: Matrix factorization

- Date: 2026-09-08
- Owner: Nithilan Kumaran
- Status: Accepted

**Context:** The GNN-versus-baseline quality comparison cannot start until
the owner selects the non-GNN recommender. The two allowed options were
implicit-feedback matrix factorization and node2vec. The baseline must use
the same split, eligible users, candidate protocol, and ranking metrics as
the future GraphSAGE model (ADR-002, ADR-003).

**Choice:** Implicit-feedback matrix factorization.

**Alternatives:** node2vec (or another graph-walk embedding baseline), or
implementing both families before the first quality comparison.

**Rationale:** The owner accepted matrix factorization on 2026-09-08.
MF is a direct, interpretable recommender baseline: it scores user–item
pairs from latent factors without introducing graph-walk hyperparameters
that would confound the GraphSAGE comparison. node2vec would mix random-walk
design choices with the GNN neighborhood-sampling story.

**Consequences:**

- Phase 3 implements implicit-feedback matrix factorization with the same
  split, candidate construction, eligibility, and ranking protocol as the
  future GNN (Recall@10 and NDCG@10, per-user then macro-averaged).
- Do not add node2vec model code, node2vec configs, or MF-versus-node2vec
  experiment matrices unless a superseding ADR accepts node2vec.
- The shared scoring interface and ranking evaluator own metric logic;
  the baseline (and later GraphSAGE) only produce ranking scores.
- Training negatives must not overlap known positives in the training
  scope. Held-out positives must not enter the MF training edge set.
- This slice verifies the protocol on tiny deterministic synthetic data.
  Do not publish MovieLens 100K quality numbers until a real 100K run is
  stored under `results/` with provenance. ADR-005 (sampler semantics)
  is accepted separately.

### Disjoint global node IDs

Users occupy `[0, num_users)` and movies occupy
`[num_users, num_users + num_movies)`. This prevents type collisions and makes
type recovery constant-time. This convention is architecture-neutral and may be
changed only through a superseding decision record.

### ADR-005: Uniform sampling without replacement

- Date: 2026-09-09
- Owner: Nithilan Kumaran
- Status: Accepted

**Context:** Neighbor sampling must have one owner-selected replacement policy
so the C++ sampler, the Python reference sampler, and later GraphSAGE
mini-batches share a single contract. The two allowed options were uniform
sampling without replacement and sampling with replacement. Phase 1 already
implemented without-replacement defaults so CSR construction and
`graph_sampler` had defined behavior; that implementation contract is now
the accepted experiment decision.

**Choice:** Uniform sampling without replacement.

- Sample up to `k` unique neighbors.
- Return the full stored neighborhood (CSR order) when `k >= degree`.
- Return empty for isolated nodes or `k = 0`.
- When `0 < k < degree`, return `k` unique neighbors in Fisher–Yates prefix
  order from a per-call `std::mt19937_64` seeded by the explicit `seed`.

**Alternatives:** Sampling with replacement (duplicates allowed within a hop),
or leaving replacement unspecified.

**Rationale:** The owner accepted uniform sampling without replacement on
2026-09-09. It avoids duplicated messages within a sampled hop, matches the
already-shipped native and Python reference samplers, and is straightforward
to compare for parity tests. With-replacement semantics would change both
implementations and every parity/benchmark comparison.

**Consequences:**

- The native sampler (`sagerec::BipartiteCSR::sample_neighbors`) and the
  Python reference (`python/sagerec_reference_sampler.py`) must keep
  without-replacement semantics, including the Fisher–Yates prefix and
  unbiased `uniform_below` draw.
- Changing replacement policy, allowing duplicate neighbors within a hop,
  or switching the reference/native pair to with-replacement requires a
  superseding ADR.
- Benchmarks and GraphSAGE training must use this contract; do not silently
  substitute a with-replacement PyG sampler. The Phase 4 mini-batch helper
  (`python/sagerec_minibatch.py`) calls native `sample_neighbors` with this
  without-replacement policy.

### ADR-006: PyG SAGEConv on native neighborhoods

- Date: 2026-09-14
- Owner: Nithilan Kumaran
- Status: Accepted

**Context:** Project requirements say Python training must use PyTorch and
PyTorch Geometric. Phase 4/5 already train GraphSAGE (ADR-002) on CPU
PyTorch with in-repo mean aggregation, while neighborhoods come from native
`graph_sampler` via `sagerec_minibatch` (ADR-005). The remaining gap is the
production conv stack. Switching neighborhood expansion to a PyG
`NeighborLoader` / `ClusterLoader` would abandon the systems contribution.

**Choice:** Use PyTorch Geometric `SAGEConv` (mean aggregation, or a
documented equivalent PyG GraphSAGE layer) as the message-passing stack.
Neighborhoods **must** still come from native `graph_sampler` via
`sagerec_minibatch.NativeMinibatchSampler`. Do not use `NeighborLoader`,
`ClusterLoader`, or any PyG sampler as the neighborhood source in primary
experiments.

**Alternatives:** Keep the in-repo `Linear(concat(self, mean(neighbors)))`
layers indefinitely; or adopt PyG `NeighborLoader` (or another PyG sampler)
for mini-batch expansion.

**Rationale:** The owner accepted a PyG production path on 2026-09-14. That
closes the PyTorch Geometric requirement without replacing the C++ CSR +
seeded ADR-005 sampler. A thin adapter from native multi-hop samples to the
`edge_index` form `SAGEConv` expects is allowed; replacing the sampler is
not.

**Consequences:**

- `python/sagerec_graphsage.py` must import and call PyG `SAGEConv` (or an
  equivalent documented PyG GraphSAGE conv). Pair scoring and the native
  mini-batch wiring stay in SageRec.
- `python/sagerec_minibatch.py` / `graph_sampler` remain the only
  neighborhood expansion path for primary training and the stored 100K
  experiment protocol.
- CPU-friendly PyG install pins live in `python/requirements-train.txt`.
  Optional PyG extension wheels (`torch-scatter`, `torch-sparse`,
  `pyg-lib`) are not required for mean `SAGEConv` and must not be added to
  default CI. Native C++ stays free of PyTorch and PyG.
- Tests must spy that native `sample_neighbors` still runs and must reject
  `NeighborLoader` / `ClusterLoader` on the primary path.
- Stored MovieLens 100K GraphSAGE metrics under `results/` were measured
  with the Phase 5 in-repo mean layers. Do not retcon or hand-edit those
  files; do not claim new 100K quality numbers without a measured rerun.
  Tiny synthetic metrics remain protocol smoke.
- MovieLens 1M, GCN, and node2vec remain deferred/rejected unless a
  superseding ADR accepts them.

### ADR-007: Multi-seed published MF vs GraphSAGE comparison

- Date: 2026-09-15
- Owner: Nithilan Kumaran
- Status: Accepted

**Context:** Phase 5 stored a single-seed MovieLens 100K GraphSAGE vs implicit
MF table and explicitly labeled it as not a leaderboard. Project requirements
ask for uncertainty across multiple seeds or a clearly labeled single-seed
result. The owner asked to run the multi-seed leaderboard slice on
2026-09-15: MF vs GraphSAGE with uncertainty on MovieLens 100K / ADR-003
only. Phase 6 already uses PyG `SAGEConv` fed by native `graph_sampler`.

**Choice:** Publish a modest 5-seed MovieLens 100K comparison of implicit MF
and GraphSAGE using the same seed list for both models:

- Seeds: **7, 11, 13, 17, 19** (seed **7** is required for continuity with
  the Phase 5 single-seed runs).
- Split/eval: ADR-003 leave-one-out; shared `PairScorer` / Recall@10 /
  NDCG@10 path; same eligible users and candidate construction.
- GraphSAGE: current Phase 6 path — PyG `SAGEConv` on native
  `graph_sampler` neighborhoods via `sagerec_minibatch`. Do not use
  `NeighborLoader` / `ClusterLoader`.
- Uncertainty: per-seed rows plus **mean ± sample standard deviation
  (n-1)** (median also stored) for Recall@10 and NDCG@10.
- Historical Phase 5 files (`results/mf_movielens_100k.json`,
  `results/graphsage_movielens_100k.json`, `results/gnn_vs_mf_*`) remain
  single-seed provenance and must not be retconned into multi-seed
  artifacts. New filenames hold the leaderboard.

**Alternatives:** Keep only the labeled single-seed table; run 10+ seeds;
mix Phase 5 in-repo-mean GraphSAGE numbers with SAGEConv runs; change
replacement policy or swap in a PyG sampler.

**Rationale:** The owner accepted this slice on 2026-09-15. Five seeds
including 7 are enough to show spread on CPU without a huge job, keep the
Phase 5 continuity seed, and avoid inventing or mixing protocols.

**Consequences:**

- `python/sagerec_multiseed.py` aggregates measured per-seed ranking JSON
  with the same protocol checks as `sagerec_compare.py`.
- `scripts/run_multiseed_gnn_vs_mf.py` trains both models across the seed
  list and writes `results/multiseed_gnn_vs_mf_movielens_100k.{json,md,svg}`.
- Fairness is the shared eval protocol. MF remains 1 SGD epoch; GraphSAGE
  remains 3 Adam epochs unless a later measured config change is stored.
- Default CI must not download MovieLens or run the 100K multi-seed job.
- Do not add MovieLens 1M, GCN, node2vec, or NeighborLoader on the
  primary path.

### ADR-008: Demo CLI serving

- Date: 2026-09-17
- Owner: Nithilan Kumaran
- Status: Accepted

**Context:** Phases 1–7 deliver native sampling, GraphSAGE training (PyG
`SAGEConv` on `graph_sampler` neighborhoods), implicit MF, and stored
MovieLens 100K ranking evidence. The remaining production-shaped slice is
how a trained model is *used*: score candidates for one user and print
top-K. The allowed serving shapes were a demo CLI, a batch export job,
or a local HTTP API. MovieLens 100K (ADR-001), GraphSAGE (ADR-002),
ADR-003 splits, implicit MF (ADR-004), without-replacement sampling
(ADR-005), native neighborhoods (ADR-006), and the multi-seed
leaderboard (ADR-007) stay locked.

**Choice:** A **demo CLI** that loads a GraphSAGE (or PairScorer-compatible
implicit-MF) checkpoint from disk, scores caller-supplied candidate movie
IDs for one user, and prints top-K (default 10) as `rank`, `movie_id`,
`score`. Ranking is score descending; ties break on smaller `movie_id`.

**Alternatives:** Batch export of scores for every eligible user; a local
HTTP recommend API.

**Rationale:** The owner accepted a demo CLI on 2026-09-17. A CLI keeps
the contract testable in default CI (synthetic checkpoints, no MovieLens
download, no network server) and reuses `PairScorer` plus native
mini-batch neighborhoods instead of standing up a second serving stack.
Batch export and HTTP are useful later, but they add job/API surface
without proving checkpoint load or ranking correctness first.

**Consequences:**

- `python/sagerec_serve.py` owns checkpoint schema v1
  (`sagerec_checkpoint`), load/save, candidate ranking, and CLI logic.
  `scripts/demo_recommend.py` is a thin launcher (`--checkpoint`,
  `--user`, `--candidates`, `--k`).
- GraphSAGE serve-time neighborhoods still come from native
  `graph_sampler` via `sagerec_minibatch`. Do not use `NeighborLoader` /
  `ClusterLoader`. Train-only pairs are required to rebuild that CSR
  (embedded in the checkpoint, or `--train-pairs` / `--processed-dir`).
  The CLI must not download MovieLens by default.
- Missing or corrupt checkpoints fail with an actionable nonzero exit.
- Do not add batch export or a local HTTP API unless a superseding ADR
  accepts them. Do not invent or edit stored 100K quality result files.
- MovieLens 1M, GCN, and node2vec remain deferred/rejected.

## Proposals

None open. MovieLens 1M, GCN, and node2vec remain deferred/rejected unless a
superseding ADR accepts them.
