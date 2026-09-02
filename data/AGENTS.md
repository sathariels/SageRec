# Data Agent Guide

## Policy

This directory describes local data placement and provenance; external MovieLens
data and generated processed artifacts are not committed.

- Treat `raw/` inputs as immutable.
- Make every `processed/` artifact reproducible from source data and versioned config.
- Record dataset edition, source URL, license reference, checksum, parser version,
  row counts, mapping counts, split policy, seed, and schema in a manifest.
- Preserve rating and timestamp even when graph connectivity is unweighted.
- Fit mappings and construct adjacency according to the accepted cold-start policy.
- Assert that validation/test positives are absent from the training graph.
- Never copy private paths, credentials, or downloaded archives into Git.
