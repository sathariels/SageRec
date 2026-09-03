# Native Implementation Agent Guide

This directory contains C++ implementations and the thin pybind11 boundary.

- `bipartite_csr.cpp` owns construction, validation, and sampling.
- `movielens_100k.cpp` owns MovieLens 100K `u.data` parsing and ID mapping.
  It must not construct a CSR or apply a split.
- `bindings.cpp` must stay free of business logic beyond GIL release, seed
  sign checks, and exception translation. Do not bind the parser until a later
  task asks for a Python ingestion surface.
- Validate inputs at public boundaries and keep internal hot paths lean.
- Prefer contiguous storage, RAII, standard-library facilities, and explicit types.
- Avoid global mutable state and implicit thread-local random engines.
- Separate parsing, graph construction, sampling, and binding responsibilities.
- Measure before optimizing and retain correctness tests for every optimization.
- Do not log from tight sampling loops; expose measurements to the benchmark harness.
- Do not add MovieLens 1M parsers, download helpers, or file-path convenience
  APIs here.
