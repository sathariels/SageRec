"""Training negative sampling that excludes known positives.

The relevant scope is caller-supplied: training uses training positives.
Held-out pairs must not be used as MF training positives; they are not
required in this forbidden set unless the caller passes them.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence, Set
from typing import Any

import numpy as np

Pair = tuple[int, int]


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int, got {type(value).__name__}")
    return value


def items_by_user(pairs: Iterable[Pair]) -> dict[int, set[int]]:
    """Map each user id to the item ids observed in ``pairs``."""
    by_user: dict[int, set[int]] = defaultdict(set)
    for user_id, item_id in pairs:
        by_user[_require_int(user_id, "user_id")].add(
            _require_int(item_id, "item_id")
        )
    return dict(by_user)


def sample_negative_item(
    user_id: int,
    *,
    num_items: int,
    forbidden_items: Set[int],
    rng: np.random.Generator,
    max_tries: int | None = None,
) -> int:
    """Draw one item in ``[0, num_items)`` that is not in ``forbidden_items``.

    Uses rejection sampling, then falls back to the explicit allowed catalog
    if the catalog is small or rejection fails. Fails if the user has no
    eligible negative.
    """
    user_id = _require_int(user_id, "user_id")
    num_items = _require_int(num_items, "num_items")
    if num_items <= 0:
        raise ValueError(f"num_items must be positive, got {num_items}")
    if user_id < 0:
        raise ValueError(f"user_id {user_id} is negative")

    n_forbidden = 0
    for item in forbidden_items:
        item_id = _require_int(item, "forbidden item")
        if 0 <= item_id < num_items:
            n_forbidden += 1
    if n_forbidden >= num_items:
        raise ValueError(
            f"user {user_id} has no eligible negative items: "
            f"{n_forbidden} forbidden in a catalog of {num_items}"
        )

    if max_tries is None:
        max_tries = max(64, 8 * num_items)
    if max_tries < 1:
        raise ValueError(f"max_tries must be positive, got {max_tries}")

    for _ in range(max_tries):
        item_id = int(rng.integers(0, num_items))
        if item_id not in forbidden_items:
            return item_id

    allowed = [item for item in range(num_items) if item not in forbidden_items]
    if not allowed:
        raise ValueError(
            f"user {user_id} has no eligible negative items in a catalog of {num_items}"
        )
    return int(allowed[int(rng.integers(0, len(allowed)))])


def sample_negative_items(
    user_id: int,
    count: int,
    *,
    num_items: int,
    forbidden_items: Set[int],
    rng: np.random.Generator,
) -> list[int]:
    """Draw ``count`` negatives for one user (with replacement across draws)."""
    count = _require_int(count, "count")
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    return [
        sample_negative_item(
            user_id,
            num_items=num_items,
            forbidden_items=forbidden_items,
            rng=rng,
        )
        for _ in range(count)
    ]


def assert_no_positive_overlap(
    negatives: Sequence[Pair],
    positives: Set[Pair] | Mapping[int, Set[int]],
) -> None:
    """Fail if any negative pair is a known positive in the given scope."""
    if isinstance(positives, Mapping):
        positive_pairs = {
            (user_id, item_id)
            for user_id, items in positives.items()
            for item_id in items
        }
    else:
        positive_pairs = set(positives)
    overlap = [pair for pair in negatives if pair in positive_pairs]
    if overlap:
        user_id, item_id = overlap[0]
        raise ValueError(
            "negative pair overlaps a known positive: "
            f"(user_id={user_id}, item_id={item_id})"
        )
