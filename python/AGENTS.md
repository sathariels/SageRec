# Python System Agent Guide

## Responsibility

Own native-extension stubs and binding tests now. Later: dataset preparation
orchestration, GNN and baseline models, training, evaluation, configuration,
benchmark coordination, and result serialization.

## Current slice

- Public type hints live in `graph_sampler.pyi`.
- Binding tests import the compiled `graph_sampler` module only.
- The module exposes `BipartiteCSR` and in-memory `parse_movielens_100k`.
- Do not implement MovieLens download, split, GraphSAGE training, baseline,
  or benchmark code until those phases are opened.
- Do not add MovieLens 1M or GCN modules.

## Boundaries

- The selected GNN is GraphSAGE and will use PyTorch and PyTorch Geometric.
- Training neighborhood expansion must call `graph_sampler`; do not quietly fall
  back to a PyG sampler in primary experiments.
- A naive Python sampler exists only for reference, correctness comparison, and timing.
- Separate modules for data, sampler adapter, models, trainer, evaluator, benchmark,
  and CLI entry points.
- Depend on a narrow sampler protocol so unit tests can inject a deterministic fake.

## Correctness

- Apply the split before constructing the training graph.
- Exclude known positive pairs from negatives and ranked candidates as specified.
- Keep GNN and baseline evaluation paths identical.
- Seed Python, NumPy, PyTorch, data-loader workers, and native sampling explicitly.
- Record nondeterministic backend settings and device information.
- Validate tensor shapes, node type/range, and mapping compatibility early.

## Tests

- Binding construction, exceptions, lifetime, seeded smoke sampling, and
  in-memory MovieLens 100K parser tests (tiny strings, CSR handoff, GraphError).
- Later: split leakage, negative-sample validity, metrics, and tiny e2e smoke.
