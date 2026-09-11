# CI Agent Guide

GitHub Actions here must verify the native foundation without side effects.

- Configure a Release CMake build under `cpp/`, build, and run CTest.
- Fail the workflow with a nonzero exit when configure, build, or tests fail.
- Do not download MovieLens, commit datasets, or store secrets. Parser and
  Phase 2 download/prep CTest cases must keep using in-memory or temp-dir
  fixtures. Do not set `SAGEREC_LIVE_MOVIELENS` in CI.
- Do not upload checkpoints, compiled extensions, or generated artifacts as
  source. Cache build directories only if that remains optional and local.
- Do not add MovieLens 1M or GCN jobs. Phase 3 Python tests need NumPy
  (`python3-numpy`); they must keep using in-memory synthetic fixtures.
  Phase 4 mini-batch and GraphSAGE training tests must keep using the
  compiled `graph_sampler` extension and tiny synthetic graphs. Default
  CI may install CPU-only PyTorch for GraphSAGE tests; do not add CUDA
  wheels, PyG NeighborLoader jobs, MovieLens downloads, or a Phase 5
  quality-table job.
