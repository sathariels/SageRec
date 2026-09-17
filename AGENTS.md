# SageRec Agent Handbook

## Mission

Build a reproducible graph recommendation system whose key systems contribution
is a real C++ CSR preprocessing and neighbor-sampling pipeline used by Python GNN
training. Optimize for correctness, clear experiments, measurable performance,
and an agent-friendly repository.

## Current phase

Phase 1 native foundation is verified. ADR-001 (MovieLens 100K), ADR-002
(GraphSAGE), ADR-003 (per-user chronological leave-one-out), ADR-004
(matrix factorization), ADR-005 (uniform sampling without replacement),
ADR-006 (PyG SAGEConv on native `graph_sampler` neighborhoods),
ADR-007 (multi-seed MF vs GraphSAGE leaderboard), and ADR-008 (demo CLI
serving) are accepted; MovieLens 1M is deferred. The verified slice includes the C++ CSR graph, seeded
neighbor sampler, MovieLens 100K `u.data` parser, `graph_sampler`
bindings, a naive Python reference sampler with native parity tests,
ADR-003 leave-one-out prep, MovieLens 100K download and on-disk
`processed/` writes, an implicit MF baseline, a shared ranking evaluator,
a native-backed mini-batch neighborhood helper, GraphSAGE training on
that harness (PyG `SAGEConv`, native neighborhoods), and CI.

The current verified tip is Phase 8 / ADR-008. Phase 2 is delivered for
MovieLens 100K: download + on-disk `processed/` prep, and timing charts
(native `graph_sampler` versus the Python reference sampler, with stored
JSON/SVG under `results/`; synthetic graph by default; optional
train-only 100K is a script flag). Phase 3 is
delivered for the matrix-factorization baseline and the shared Recall@10 /
NDCG@10 evaluator. Phase 4 is delivered for GraphSAGE training on the
native mini-batch harness (the Python training data path must call
`graph_sampler` / `sagerec_minibatch`; do not silently use a PyG
NeighborLoader). Phase 5 is delivered for the single-seed MovieLens 100K
GraphSAGE quality run and the GNN-versus-MF comparison table/chart under
`results/` (historical single-seed provenance; do not retcon). Phase 6 is
delivered for PyG `SAGEConv` message passing still fed by native sampling.
Phase 7 is delivered for the 5-seed MovieLens 100K MF vs GraphSAGE
leaderboard with mean±sample-std (`results/multiseed_gnn_vs_mf_movielens_100k.*`).
Phase 8 is delivered for the ADR-008 demo CLI: load a GraphSAGE or
PairScorer-compatible checkpoint, score candidate movie IDs, print top-K
(`python/sagerec_serve.py`, `scripts/demo_recommend.py`). Tiny synthetic
GraphSAGE metrics remain protocol smoke and must stay separate from the
stored 100K numbers. Do not add MovieLens 1M, GCN, HTTP, or batch-export
paths. Do not invent metrics or hand-edit timings.

## Read order

Before changing the repository, read:

1. This file.
2. `docs/project-requirements.md`.
3. `docs/architecture.md`.
4. `docs/decisions.md`.
5. The closest directory-level `AGENTS.md` for files being changed.

The closest `AGENTS.md` adds local constraints but does not override project-wide
correctness, reproducibility, or owner-decision requirements.

## Owner decisions

Accepted and recorded in `docs/decisions.md`:

- ADR-001: MovieLens 100K (MovieLens 1M deferred).
- ADR-002: GraphSAGE (not GCN).
- ADR-003: Per-user chronological leave-one-out (min 3 interactions; cold-start
  users keep all rows in train and are excluded from ranking eligibility).
- ADR-004: Implicit-feedback matrix factorization (not node2vec).
- ADR-005: Uniform sampling without replacement (native and Python reference
  samplers must keep this contract; replacement policy changes need a
  superseding ADR).
- ADR-006: PyG `SAGEConv` on native `graph_sampler` neighborhoods (not a PyG
  NeighborLoader).
- ADR-007: Multi-seed published MF vs GraphSAGE comparison on MovieLens
  100K (seeds 7, 11, 13, 17, 19; mean ± sample std).
- ADR-008: Demo CLI serving (load checkpoint, score candidates, print
  top-K). Not batch export. Not a local HTTP API.

Do not add node2vec unless a superseding ADR accepts it.

Record any new owner choice in `docs/decisions.md` before creating dependent
code or configuration. Do not infer MovieLens 1M or GCN.

## System-wide invariants

- Use a bipartite graph: users and movies are nodes; interactions are edges.
- Use zero-based internal IDs and disjoint user/movie ID ranges.
- Store each training interaction in both directions in the sampling graph.
- Build the training graph from training positives only; held-out positives must
  never enter preprocessing, message passing, negative sampling, or filtering.
- Negative user-movie pairs must not overlap positives in the relevant dataset scope.
- Random behavior accepts explicit seeds and experiment outputs record them.
- Compare models with the same split, eligible users, candidate protocol, and metrics.
- The Python training data path must actually call the C++ sampler.
- The C++ benchmark and Python reference sampler must implement equivalent semantics.
- Raw data, build artifacts, compiled extensions, checkpoints, and large generated
  files are not source-controlled.

## Working rules

- Keep public contracts narrow and make invalid states fail with actionable errors.
- Separate reusable logic from command-line entry points.
- Pair behavior changes with tests at the lowest useful layer.
- Prefer explicit configuration over hidden constants or machine-specific paths.
- Preserve provenance: dataset, split, seed, config, commit, dependencies, and hardware.
- Update documentation whenever commands, data schemas, metrics, or architecture change.
- Do not claim performance or quality improvements without stored evidence.
- Do not silently broaden the scope beyond MovieLens implicit-ranking evaluation.

## Definition of done

The project is complete only when all acceptance criteria in
`docs/project-requirements.md` pass, including native tests, binding tests, an
end-to-end deterministic smoke run, benchmark parity, and a GNN-versus-baseline
quality report containing Recall@10 and NDCG@10.
