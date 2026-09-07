# Architecture Decision Log

Use one section per decision. Accepted entries must include date, owner, context,
choice, alternatives, rationale, and consequences.

## Decision status

| ID | Topic | Options | Status |
| --- | --- | --- | --- |
| ADR-001 | Dataset | MovieLens 100K / MovieLens 1M | Accepted: MovieLens 100K (1M deferred) |
| ADR-002 | GNN | GraphSAGE / GCN | Accepted: GraphSAGE |
| ADR-003 | Split | Per-user chronological leave-one-out / global time split | Accepted: per-user chronological leave-one-out |
| ADR-004 | Baseline | Matrix factorization / node2vec | Proposed |
| ADR-005 | Sampler semantics | Uniform without replacement / with replacement | Proposed |

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
- ADR-003 (split) is accepted separately; ADR-004 (baseline) remains a
  proposal and is not settled by this dataset choice.

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
  substitute a PyG neighbor sampler in primary experiments.
- This decision does not accept ADR-004 or ADR-005. ADR-003 is accepted
  separately.

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
- ADR-004 (baseline) and ADR-005 (sampler replacement) remain proposals.

### Disjoint global node IDs

Users occupy `[0, num_users)` and movies occupy
`[num_users, num_users + num_movies)`. This prevents type collisions and makes
type recovery constant-time. This convention is architecture-neutral and may be
changed only through a superseding decision record.

## Proposals

### ADR-004: Matrix factorization baseline

Proposed choice: implicit-feedback matrix factorization with the same negative
sampling and ranking protocol. It is a direct, interpretable recommender baseline
and avoids conflating graph-walk hyperparameters with the GNN comparison.

### ADR-005: Uniform sampling without replacement

Proposed choice: sample up to `k` unique neighbors; return all neighbors when
degree is at most `k`. It avoids duplicated messages within a sampled hop and is
straightforward to compare with a Python reference.

The Phase 1 native sampler implements these proposed defaults as the current
code contract so CSR construction and `graph_sampler` have defined behavior.
ADR-005 itself remains a proposal, not an accepted experiment decision. Changing
replacement policy still requires accepting or superseding this ADR.
