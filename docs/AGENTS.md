# Documentation Agent Guide

## Purpose

This directory is the source of truth for architecture, project requirements,
decisions, evaluation protocol, and phased delivery plans.

## Requirements

- Keep requirements testable and distinguish required, proposed, and accepted choices.
- ADR-001 (MovieLens 100K), ADR-002 (GraphSAGE), ADR-003 (per-user
  chronological leave-one-out), ADR-004 (matrix factorization), ADR-005
  (uniform sampling without replacement), and ADR-006 (PyG SAGEConv on
  native neighborhoods) are accepted. Do not add MovieLens 1M, GCN, or
  node2vec. Do not switch primary experiments to a PyG NeighborLoader.
- Record consequential decisions with context, alternatives, rationale, and consequences.
- Define metric and benchmark protocols before producing results.
- Keep diagrams aligned with actual component boundaries.
- Link claims to versioned result summaries; never invent placeholder numbers.
- Update the root README when user-facing structure or workflow changes.
