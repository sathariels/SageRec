# CI Agent Guide

GitHub Actions here must verify the native foundation without side effects.

- Configure a Release CMake build under `cpp/`, build, and run CTest.
- Fail the workflow with a nonzero exit when configure, build, or tests fail.
- Do not download MovieLens, commit datasets, or store secrets. Parser CTest
  cases must keep using in-memory fixtures only.
- Do not upload checkpoints, compiled extensions, or generated artifacts as
  source. Cache build directories only if that remains optional and local.
- Do not add MovieLens 1M or GCN jobs.
