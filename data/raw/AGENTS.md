# Raw Data Agent Guide

Store immutable, externally obtained MovieLens files here during local work.
Do not edit, normalize, or commit them. The expected 100K ratings file is
tab-separated `u.data` (source user id, movie id, rating, timestamp).

Phase 2 download (`python/sagerec_download.py` / `scripts/download_movielens_100k.py`)
fetches the official GroupLens MovieLens 100K zip, verifies the published MD5
when the default URL is used, and extracts `ml-100k/u.data`. The ADR-010
re-rank script may also extract `ml-100k/u.item` from that same zip into
this gitignored directory for genre flags. Native parsing
stays in-memory; it does not read this directory itself.

Record source URL, dataset license reference, archive checksum, and extracted
ratings-file checksum in the processed dataset manifest. Do not add MovieLens 1M.
Never copy archives into Git.
