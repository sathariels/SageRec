# Data Agent Guide

## Policy

This directory describes local data placement and provenance; external MovieLens
data and generated processed artifacts are not committed.

- Treat `raw/` inputs as immutable.
- Make every `processed/` artifact reproducible from source data and versioned config.
- Record dataset edition, source URL, license reference, checksum, parser version,
  row counts, mapping counts, split policy, seed, and schema in a manifest.
- Preserve rating and timestamp even when graph connectivity is unweighted.
- Native 100K parsing is `sagerec::parse_movielens_100k` on in-memory `u.data`
  text. Split assignment stays in `python/sagerec_prep.py` (ADR-003). Phase 2
  download is `python/sagerec_download.py`; on-disk prep is
  `python/sagerec_dataset.py`. Do not add MovieLens 1M.
- The manifest schema is `processed/manifest.schema.json`. On-disk prep fills
  edition, split policy, eligibility/cold-start defaults, counts, source URL,
  checksum, and license.
- Fit mappings and construct adjacency according to the accepted cold-start
  policy (users with fewer than 3 interactions: all rows in train; user
  excluded from ranking eligibility).
- Assert that validation/test positives are absent from the training graph.
- Never copy private paths, credentials, or downloaded archives into Git.
