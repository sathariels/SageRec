"""Shared ranking evaluator: Recall@K and NDCG@K with macro averaging.

Owns candidate filtering, ranking, and metric aggregation. Models implement
``PairScorer`` and must not compute these metrics themselves.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence, Set
from dataclasses import dataclass
from typing import Any, Literal

from sagerec_prep import MIN_INTERACTIONS_FOR_EVAL, SplitName, SplitResult
from sagerec_scoring import PairScorer

DEFAULT_K = 10
EvalSplit = Literal["validation", "test"]


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int, got {type(value).__name__}")
    return value


def _require_k(k: int) -> int:
    k = _require_int(k, "k")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return k


def recall_at_k(
    ranked_items: Sequence[int],
    relevant: Set[int],
    k: int = DEFAULT_K,
) -> float:
    """Fraction of relevant items recovered in the top ``k`` ranks."""
    k = _require_k(k)
    if not relevant:
        raise ValueError("relevant is empty; Recall is undefined")
    top = set(ranked_items[:k])
    hits = sum(1 for item in relevant if item in top)
    return hits / float(len(relevant))


def ndcg_at_k(
    ranked_items: Sequence[int],
    relevant: Set[int],
    k: int = DEFAULT_K,
) -> float:
    """Binary-relevance NDCG@k. IDCG uses ``min(k, |relevant|)`` ideal ranks."""
    k = _require_k(k)
    if not relevant:
        raise ValueError("relevant is empty; NDCG is undefined")
    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def filtered_candidates(
    num_items: int,
    *,
    exclude: Set[int],
) -> list[int]:
    """All item ids in ``[0, num_items)`` except ``exclude``."""
    num_items = _require_int(num_items, "num_items")
    if num_items < 1:
        raise ValueError(f"num_items must be >= 1, got {num_items}")
    return [item for item in range(num_items) if item not in exclude]


def rank_items(
    scorer: PairScorer,
    user_id: int,
    item_ids: Sequence[int],
) -> list[int]:
    """Rank ``item_ids`` by descending score; ties break on smaller item id."""
    user_id = _require_int(user_id, "user_id")
    if not item_ids:
        raise ValueError("item_ids is empty; cannot rank")
    scores = list(scorer.score_pairs([user_id] * len(item_ids), item_ids))
    if len(scores) != len(item_ids):
        raise ValueError(
            "PairScorer.score_pairs must return one score per pair, "
            f"got {len(scores)} scores for {len(item_ids)} items"
        )
    keyed = [(-float(score), int(item), index) for index, (score, item) in enumerate(zip(scores, item_ids))]
    keyed.sort()
    return [item_ids[index] for _, __, index in keyed]


def _items_by_user(split: SplitResult, name: SplitName) -> dict[int, set[int]]:
    by_user: dict[int, set[int]] = defaultdict(set)
    for row in split.interactions_for(name):
        by_user[row.user_id].add(row.movie_id)
    return dict(by_user)


@dataclass(frozen=True)
class RankingReport:
    """Macro-averaged ranking metrics for one evaluation split."""

    k: int
    split: EvalSplit
    n_users: int
    recall_at_k: float
    ndcg_at_k: float
    evaluated_user_ids: tuple[int, ...]
    per_user_recall: tuple[float, ...]
    per_user_ndcg: tuple[float, ...]


def evaluate_ranking(
    scorer: PairScorer,
    split: SplitResult,
    *,
    target_split: EvalSplit = "test",
    k: int = DEFAULT_K,
) -> RankingReport:
    """Evaluate ``scorer`` on ADR-003-eligible users.

    Validation evaluation filters training positives. Test evaluation also
    filters validation positives. The held-out target items stay in the
    candidate set. Cold-start users are omitted.
    """
    k = _require_k(k)
    if target_split not in ("validation", "test"):
        raise ValueError(
            f"target_split must be 'validation' or 'test', got {target_split!r}"
        )
    eligible = split.eligible_user_ids
    if not eligible:
        raise ValueError(
            "no eligible users to evaluate; ADR-003 requires at least "
            f"{MIN_INTERACTIONS_FOR_EVAL} interactions per ranking user"
        )

    train_by_user = _items_by_user(split, "train")
    validation_by_user = _items_by_user(split, "validation")
    target_by_user = _items_by_user(split, target_split)
    filter_validation = target_split == "test"

    user_ids: list[int] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    for user_id in eligible:
        relevant = target_by_user.get(user_id)
        if not relevant:
            raise ValueError(
                f"eligible user {user_id} has no {target_split} positives"
            )
        exclude = set(train_by_user.get(user_id, set()))
        if filter_validation:
            exclude |= validation_by_user.get(user_id, set())
        exclude -= relevant
        candidates = filtered_candidates(split.num_movies, exclude=exclude)
        if not candidates:
            raise ValueError(
                f"user {user_id} has an empty candidate set after filtering"
            )
        ranked = rank_items(scorer, user_id, candidates)
        user_ids.append(user_id)
        recalls.append(recall_at_k(ranked, relevant, k))
        ndcgs.append(ndcg_at_k(ranked, relevant, k))

    n_users = len(user_ids)
    return RankingReport(
        k=k,
        split=target_split,
        n_users=n_users,
        recall_at_k=sum(recalls) / float(n_users),
        ndcg_at_k=sum(ndcgs) / float(n_users),
        evaluated_user_ids=tuple(user_ids),
        per_user_recall=tuple(recalls),
        per_user_ndcg=tuple(ndcgs),
    )
