# Native Public API Agent Guide

This directory contains stable public C++ headers only.

- Use fixed-width integer types for stored node IDs and offsets.
- Document node-ID layout, ownership, lifetime, complexity, and exception behavior.
- Do not expose pybind11, Python, PyTorch, or implementation-only types.
- Keep includes minimal and prevent platform-specific types from entering the API.
- Every public behavior must have a native test.

No implementation headers should be added here merely for convenience.
