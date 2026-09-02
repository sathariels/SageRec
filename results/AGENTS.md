# Results Agent Guide

## Required evidence

Store lightweight CSV/JSON summaries and generated charts that support reported
claims. Every record must identify commit, dataset manifest, configuration, seed,
dependency versions, hardware/device, timestamp, and protocol version.

- Never hand-edit generated metrics or benchmark results.
- Never report invented or placeholder values as results.
- Clearly distinguish smoke runs from final experiments.
- Compare models with identical eligible users and ranking candidates.
- Include Recall@10 and NDCG@10 for the GNN and accepted baseline.
- Include native/reference timings and speedup with benchmark context.
- Keep checkpoints and bulky per-run logs in ignored subdirectories.
