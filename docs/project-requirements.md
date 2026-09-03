# Project Requirements

## 1. Goal

Deliver a GNN-based recommendation engine for MovieLens with a C++17
preprocessing/sampling backend exposed to Python through pybind11. Demonstrate
both recommendation quality and measured native sampling performance.

## 2. Functional requirements

### Data

- Support MovieLens 100K (ADR-001). MovieLens 1M is deferred; do not add 1M
  paths, configs, or downloads.
- Parse users, movies, ratings, and timestamps.
- Normalize source IDs into documented zero-based internal IDs.
- Construct a bipartite graph where every training rating creates a user-movie edge.
- Produce a deterministic, leakage-safe train/validation/test split.
- Store enough metadata to reproduce mappings and splits.

### C++ backend

- Load normalized interactions or the selected MovieLens 100K `u.data` format
  (`sagerec::parse_movielens_100k` for in-memory tab-separated rows).
- Construct and validate compressed sparse row adjacency.
- Implement native neighbor sampling without delegating to a graph library.
- Define behavior for isolated nodes, `k = 0`, `k >= degree`, replacement policy,
  duplicate input edges, invalid IDs, and random seeds.
- Expose construction, graph metadata, and sampling via a pybind11 module named
  `graph_sampler` unless a decision record changes that name.
- Build cleanly with CMake and include dependency-light native tests.

### Python training

- Use PyTorch and PyTorch Geometric.
- Implement GraphSAGE (ADR-002). Do not add a GCN model unless a superseding
  decision accepts it.
- Use the C++ sampler in the mini-batch training data path.
- Train with non-interacted user-movie negatives and exclude known positives.
- Make model, optimizer, sampler fanout, batch size, seed, and paths configurable.
- Save machine-readable run configuration, logs, and metrics.

### Baseline

- Implement the accepted matrix-factorization or node2vec baseline.
- Train and evaluate it on the exact same split and ranking candidates as the GNN.
- Prefer matrix factorization unless node2vec is chosen for a documented reason;
  matrix factorization offers a direct recommendation baseline and simpler parity.

### Evaluation

- Report Recall@10 and NDCG@10 as per-user metrics followed by macro averaging.
- Clearly define user eligibility and recommendation candidate construction.
- Filter training/validation positives from ranked recommendations as appropriate.
- Report GNN and baseline results in one table and at least one chart.
- Record uncertainty across multiple seeds or explicitly label a single-seed smoke result.

### Performance benchmark

- Implement a naive Python reference sampler with matching sampling semantics.
- Verify native/reference correctness properties before timing.
- Include warm-up and multiple measured repetitions.
- Record dataset/graph size, node-query workload, `k`, replacement policy, seed,
  build type, compiler, Python version, CPU, and timing statistic.
- Report native and Python throughput/latency plus computed speedup.

## 3. Non-functional requirements

- Reproducible from documented commands on a clean environment.
- Deterministic where libraries and hardware allow; document remaining nondeterminism.
- No test leakage and no unfiltered false negatives.
- Useful errors for bad data, unavailable extension modules, and invalid config.
- Type hints for Python public APIs and clear ownership/lifetime rules in C++.
- CI-ready commands with nonzero exit status on failure.
- No raw dataset redistribution, secrets, local absolute paths, or large artifacts in Git.

## 4. Required repository shape

```text
cpp/        C++ library, bindings, native tests, benchmark support
python/     package, configurations, training/evaluation tools, Python tests
data/       ignored raw/processed artifacts plus schemas and provenance guidance
results/    lightweight metrics and charts; large runs/checkpoints ignored
docs/       architecture, decisions, protocols, and plans
scripts/    thin reproducible workflow launchers
```

Subdirectories may be refined during implementation, but their responsibilities
must remain clear and their nearest `AGENTS.md` must be updated.

## 5. Acceptance criteria

- A Release CMake build produces the native library, tests, and Python extension.
- Native CSR/sampling tests and Python binding tests pass.
- A tiny synthetic graph exercises the full training/evaluation path deterministically.
- Selected MovieLens data can be prepared from documented commands.
- Training logs prove that mini-batch sampling uses `graph_sampler`.
- C++ versus Python benchmark results include correctness parity and context.
- GNN and baseline are evaluated with the same protocol.
- `results/` contains machine-readable metrics and a generated comparison chart.
- README setup, architecture, experiment, and troubleshooting instructions are current.
