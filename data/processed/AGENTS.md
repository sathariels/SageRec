# Processed Data Agent Guide

Only deterministic generated artifacts belong here. Each dataset version must
have a manifest describing input checksums, schema, ID mappings, split method,
seed, counts, and generating command/config. Regeneration must not depend on
timestamps, directory ordering, or unstated random state. Do not commit large outputs.

`manifest.schema.json` is the ADR-003 MovieLens 100K split-manifest contract.
On-disk prepared ratings are not generated in this slice; Python prep stays
in-memory under `python/sagerec_prep.py`.
