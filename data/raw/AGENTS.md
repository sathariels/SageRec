# Raw Data Agent Guide

Store immutable, externally obtained MovieLens files here during local work.
Do not edit, normalize, or commit them. The expected 100K ratings file is
tab-separated `u.data` (source user id, movie id, rating, timestamp). Native
parsing is in-memory only and does not read this directory. Record source URL,
retrieval date, dataset license reference, archive checksum, and extracted
ratings-file checksum in the processed dataset manifest. Never auto-download
data without explicit workflow documentation and appropriate network
authorization.
