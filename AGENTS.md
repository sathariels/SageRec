# SageRec Agent Handbook

## Mission

Build a reproducible graph recommendation system whose key systems contribution
is a real C++ CSR preprocessing and neighbor-sampling pipeline used by Python GNN
training. Optimize for correctness, clear experiments, measurable performance,
and an agent-friendly repository.

## Current phase

Architecture and context only. Do not write implementation code until the owner
has confirmed both the dataset size and GNN family. Existing documentation may
be refined without that confirmation.

## Read order

Before changing the repository, read:

1. This file.
2. `docs/project-requirements.md`.
3. `docs/architecture.md`.
4. `docs/decisions.md`.
5. The closest directory-level `AGENTS.md` for files being changed.

The closest `AGENTS.md` adds local constraints but does not override project-wide
correctness, reproducibility, or owner-decision requirements.

## Owner decisions required before implementation

- Dataset: MovieLens 100K or MovieLens 1M.
- GNN: GraphSAGE or GCN.

Do not infer either choice. Record the owner's answer in `docs/decisions.md`
before creating dependent code or configuration.

The split policy and baseline may be proposed with tradeoffs, but must also be
recorded before experiments begin.

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
