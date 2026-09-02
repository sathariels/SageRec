# Data Layout

`raw/` is reserved for an owner-downloaded MovieLens dataset. `processed/` is
reserved for normalized interactions, ID mappings, split assignments, manifests,
and other reproducible products. Both are ignored except for their guidance files.

MovieLens 100K commonly provides tab-separated `u.data`; MovieLens 1M commonly
provides `::`-separated `ratings.dat`. The final expected input is determined by ADR-001.
