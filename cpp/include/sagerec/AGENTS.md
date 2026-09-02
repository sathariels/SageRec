# SageRec Native Namespace Guide

Future headers in this namespace should cover the smallest coherent native API:
CSR graph construction/inspection and reproducible neighbor sampling. Maintain
the global user/movie ID convention and all CSR invariants from
`docs/architecture.md`. Add a decision record before expanding the public API to
weighted edges, heterogeneous relations, GPU memory, or persistence formats.
