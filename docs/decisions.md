# Architecture Decision Log

Use one section per decision. Accepted entries must include date, owner, context,
choice, alternatives, rationale, and consequences.

## Decision status

| ID | Topic | Options | Status |
| --- | --- | --- | --- |
| ADR-001 | Dataset | MovieLens 100K / MovieLens 1M | Owner input required |
| ADR-002 | GNN | GraphSAGE / GCN | Owner input required |
| ADR-003 | Split | Per-user chronological leave-one-out / global time split | Proposed |
| ADR-004 | Baseline | Matrix factorization / node2vec | Proposed |
| ADR-005 | Sampler semantics | Uniform without replacement / with replacement | Proposed |

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

## Accepted foundational convention

### Disjoint global node IDs

Users occupy `[0, num_users)` and movies occupy
`[num_users, num_users + num_movies)`. This prevents type collisions and makes
type recovery constant-time. This convention is architecture-neutral and may be
changed only through a superseding decision record.
