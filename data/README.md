# Data Layout

`raw/` is reserved for an owner-downloaded MovieLens dataset. `processed/` is
reserved for normalized interactions, ID mappings, split assignments, manifests,
and other reproducible products. Both are ignored except for their guidance files.

ADR-001 accepts MovieLens 100K (commonly tab-separated `u.data`). MovieLens 1M
is deferred; do not add 1M paths, configs, or downloads. Ingestion itself is
not implemented in the first native slice.
