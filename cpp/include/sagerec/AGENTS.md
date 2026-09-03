# SageRec Native Namespace Guide

Public headers in this namespace are the smallest coherent native API:

- `types.hpp`: `NodeId`, `Offset`, `Degree`, `Seed`
- `error.hpp`: `GraphError`
- `bipartite_csr.hpp`: CSR construction/inspection and `sample_neighbors`
- `movielens_100k.hpp`: MovieLens 100K `u.data` parse result and parser

Maintain the global user/movie ID convention and all CSR invariants from
`docs/architecture.md`. Add a decision record before expanding the public API to
weighted edges, heterogeneous relations, GPU memory, persistence formats,
MovieLens 1M, or GCN-specific graph types.
