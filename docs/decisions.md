# Architecture Decision Log

Use one section per decision. Accepted entries must include date, owner, context,
choice, alternatives, rationale, and consequences.

## Decision status

| ID | Topic | Options | Status |
| --- | --- | --- | --- |
| ADR-001 | Dataset | MovieLens 100K / MovieLens 1M | Accepted: MovieLens 100K (1M deferred) |
| ADR-002 | GNN | GraphSAGE / GCN | Accepted: GraphSAGE |
| ADR-003 | Split | Per-user chronological leave-one-out / global time split | Proposed |
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
- ADR-003 (split) and ADR-004 (baseline) remain proposals and are not settled
  by this dataset choice.

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
- This decision does not accept ADR-003, ADR-004, or ADR-005.

### Disjoint global node IDs

Users occupy `[0, num_users)` and movies occupy
`[num_users, num_users + num_movies)`. This prevents type collisions and makes
type recovery constant-time. This convention is architecture-neutral and may be
changed only through a superseding decision record.

## Proposals

### ADR-003: Per-user chronological leave-one-out

Proposed choice: reserve each eligible user's latest interaction for test and
second-latest for validation. This aligns with top-N recommendation and preserves
per-user evaluation coverage. Define the minimum interaction count and cold-start
handling before implementation.

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
