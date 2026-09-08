"""Shared ranking-score protocol for the baseline and future GNN.

Metric logic belongs in ``sagerec_metrics``, not in a model. Any scorer that
implements ``score_pairs`` can be evaluated with the same candidate protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class PairScorer(Protocol):
    """Scores aligned user–item pairs. Higher is better for ranking."""

    def score_pairs(
        self,
        user_ids: Sequence[int],
        item_ids: Sequence[int],
    ) -> Sequence[float]:
        """Return one score per ``(user_ids[i], item_ids[i])`` pair."""
        ...
