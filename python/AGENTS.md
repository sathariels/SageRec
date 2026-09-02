# Python System Agent Guide

## Responsibility

Own dataset preparation orchestration, native sampler integration, GNN and
baseline models, training, evaluation, configuration, benchmark coordination,
result serialization, and Python tests.

## Boundaries

- The selected GNN uses PyTorch and PyTorch Geometric.
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

- Split leakage and reproducibility.
- Negative-sample validity.
- Sampler adapter and native-extension error handling.
- Metric formulas on hand-computed examples.
- Common candidate filtering for GNN and baseline.
- Tiny deterministic end-to-end training/evaluation smoke run.

Do not write Python implementation until the dataset and GNN ADRs are accepted.
