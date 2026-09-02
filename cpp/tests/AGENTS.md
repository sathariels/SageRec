# Native Test Agent Guide

Tests here must be deterministic, fast, dependency-light, and runnable through
CTest. Prefer tiny graphs whose CSR arrays can be checked exactly. Every bug fix
requires a regression test. Performance thresholds do not belong in unit tests;
benchmarks record measurements without making fragile machine-dependent assertions.

Test temporary files must use isolated temporary directories and clean themselves up.
