# Workflow Script Agent Guide

Scripts are future thin, reproducible entry points rather than homes for business logic.

- Fail fast and propagate nonzero exit codes.
- Resolve paths from the repository or explicit arguments, never a developer home path.
- Accept configuration, seed, dataset path, output path, and device explicitly.
- Print the effective command/config and versions needed for provenance.
- Delegate implementation to tested native or Python modules.
- Avoid implicit downloads, destructive cleanup, and environment mutation.
- Document each supported script in the root README.

Owner decisions ADR-001 and ADR-002 are recorded. Still do not add download,
training, or benchmark scripts until those phases are explicitly opened.
