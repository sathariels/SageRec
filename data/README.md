# Data Layout

`raw/` is reserved for an owner-downloaded MovieLens dataset. `processed/` is
reserved for normalized interactions, ID mappings, split assignments, manifests,
and other reproducible products. Both are ignored except for their guidance files.

ADR-001 accepts MovieLens 100K (tab-separated `u.data`: source user id, movie
id, rating, timestamp). MovieLens 1M is deferred; do not add 1M paths, configs,
or downloads.

The native parser `sagerec::parse_movielens_100k` reads in-memory `u.data` text
only. It remaps source IDs to contiguous zero-based local IDs (sorted unique
source IDs of each type), preserves rating and timestamp, and does not apply a
split or write `processed/` artifacts. Download, on-disk preparation, and
manifest generation remain later work. Do not commit MovieLens files.
