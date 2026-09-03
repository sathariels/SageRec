# Native Test Agent Guide

Tests here must be deterministic, fast, dependency-light, and runnable through
CTest. Prefer tiny graphs whose CSR arrays can be checked exactly. Every bug fix
requires a regression test. Performance thresholds do not belong in unit tests;
benchmarks record measurements without making fragile machine-dependent assertions.

Cover construction, duplicate-edge dedupe, isolated nodes, `k = 0`, `k >= degree`,
invalid IDs, and seed reproducibility. Parser tests must use tiny in-memory
`u.data` strings only: valid mappings, preserved rating/timestamp, CSR handoff,
empty input, malformed rows/fields/types, nonpositive source IDs, and duplicates.
Test temporary files must use isolated temporary directories and clean themselves
up. Do not download, vendor, or commit MovieLens data.
