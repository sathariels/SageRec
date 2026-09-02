# Native Implementation Agent Guide

This directory will contain C++ implementations and the thin pybind11 boundary.

- Validate inputs at public boundaries and keep internal hot paths lean.
- Prefer contiguous storage, RAII, standard-library facilities, and explicit types.
- Avoid global mutable state and implicit thread-local random engines.
- Separate parsing, graph construction, sampling, and binding responsibilities.
- Keep binding code free of business logic.
- Measure before optimizing and retain correctness tests for every optimization.
- Do not log from tight sampling loops; expose measurements to the benchmark harness.

Implementation work remains blocked on the two owner decisions in the root handbook.
