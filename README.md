# SageRec

SageRec is a planned graph-neural-network recommendation engine with a
performance-oriented C++ graph preprocessing and neighbor-sampling backend and
a Python/PyTorch Geometric training stack.

This repository currently contains architecture and agent guidance only. No
implementation code has been added.

## Intended system

```mermaid
flowchart LR
    A[MovieLens ratings] --> B[Data preparation and leakage-safe split]
    B --> C[C++ CSR graph builder]
    C --> D[C++ neighbor sampler]
    D --> E[pybind11 graph_sampler module]
    E --> F[Python mini-batch loader]
    F --> G[GNN recommender]
    B --> H[Baseline recommender]
    G --> I[Recall@10 and NDCG@10]
    H --> I
    D --> J[C++ versus Python benchmark]
```

Users and movies are distinct node types in one bipartite graph. Ratings form
user-movie edges. The C++ sampler is intended to be the actual training data
backend rather than a demonstration wrapper.

## Repository map

| Directory | Responsibility |
| --- | --- |
| `cpp/` | Native CSR construction, sampling, pybind11 boundary, and tests |
| `python/` | Data orchestration, GNN, baseline, training, evaluation, benchmark |
| `data/` | Local raw inputs and reproducible processed artifacts |
| `results/` | Metrics, benchmark summaries, and charts |
| `docs/` | Architecture, decisions, protocols, and plans |
| `scripts/` | Future thin, reproducible workflow entry points |

Read [AGENTS.md](AGENTS.md) before making changes. Each planned subsystem also
has a local `AGENTS.md` with narrower requirements.

## Owner decisions still required

Implementation must not start until the owner confirms:

1. MovieLens 100K or MovieLens 1M.
2. GraphSAGE or GCN.

The baseline and split policy are also documented as proposals, not settled
choices, in [docs/decisions.md](docs/decisions.md).

## Target deliverables

- A clean CMake build for the C++17/pybind11 module.
- A tested CSR graph and native neighbor sampler.
- A benchmark against an equivalent naive Python sampler.
- A PyTorch Geometric GNN trained with negative sampling and the native sampler.
- A matrix-factorization or node2vec baseline.
- Leakage-safe evaluation with Recall@10 and NDCG@10.
- Reproducible result tables and charts.

See [docs/project-requirements.md](docs/project-requirements.md) for acceptance
criteria and [docs/architecture.md](docs/architecture.md) for component contracts.
