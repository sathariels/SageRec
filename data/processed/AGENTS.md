# Processed Data Agent Guide

Only deterministic generated artifacts belong here. Each dataset version must
have a manifest describing input checksums, schema, ID mappings, split method,
seed, counts, and generating command/config. Regeneration must not depend on
timestamps, directory ordering, or unstated random state. Do not commit large outputs.

`manifest.schema.json` is the ADR-003 MovieLens 100K split-manifest contract.
Phase 2 on-disk prep (`python/sagerec_dataset.py`) writes gitignored
`manifest.json`, `assignments.jsonl`, `train_pairs.json`, and `mappings.json`
from a local `u.data` path or bytes. Train-only pairs must stay leakage-safe.
Timing-chart artifacts do not belong here.
