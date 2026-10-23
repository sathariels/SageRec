# Documentation Agent Guide

## Purpose

This directory is the source of truth for architecture, project requirements,
decisions, evaluation protocol, and phased delivery plans.

## Requirements

- Keep requirements testable and distinguish required, proposed, and accepted choices.
- ADR-001 (MovieLens 100K), ADR-002 (GraphSAGE), ADR-003 (per-user
  chronological leave-one-out), ADR-004 (matrix factorization), ADR-005
  (uniform sampling without replacement), ADR-006 (PyG SAGEConv on
  native neighborhoods), ADR-007 (multi-seed MF vs GraphSAGE
  leaderboard), ADR-008 (demo CLI serving), ADR-009 (HTTP serving
  on precomputed embeddings), and ADR-010 (two-stage retrieval and
  re-ranking) are accepted. Do not add MovieLens 1M,
  GCN, node2vec, LightGBM, or batch export. ADR-009 supersedes ADR-008's no-HTTP
  clause; do not put sampling or training on the HTTP request path.
  ADR-010 re-ranking is an optional second NumPy pass over precomputed
  features. Default CI does not run the 100K re-rank job.
  Do not switch primary experiments to a PyG NeighborLoader.
- Record consequential decisions with context, alternatives, rationale, and consequences.
- Define metric and benchmark protocols before producing results.
- Keep diagrams aligned with actual component boundaries.
- Link claims to versioned result summaries; never invent placeholder numbers.
- Update the root README when user-facing structure or workflow changes.
