# Data Layout

`raw/` holds an owner- or script-downloaded MovieLens 100K archive and extracted
`u.data`. `processed/` holds normalized assignments, train-only pairs, ID maps,
and a filled manifest. Both are gitignored except for guidance files and the
manifest schema.

ADR-001 accepts MovieLens 100K (tab-separated `u.data`: source user id, movie
id, rating, timestamp). MovieLens 1M is deferred; do not add 1M paths, configs,
or downloads.

Official archive: `https://files.grouplens.org/datasets/movielens/ml-100k.zip`
(published MD5 `0e33842e24a9c977be4e0107933c0723`).

The native parser `sagerec::parse_movielens_100k` reads in-memory `u.data` text
only. Phase 2 Python helpers download the zip and feed file/byte contents into
that parser, then reuse ADR-003 `python/sagerec_prep.py` for the split. Train-only
pairs go to `BipartiteCSR`. The manifest schema is `processed/manifest.schema.json`.
Do not commit MovieLens files.
