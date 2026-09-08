"""Implicit-feedback matrix factorization baseline (ADR-004).

Fits user and item factors with logistic SGD on training positives plus
uniform negatives. Scores user–item pairs through ``PairScorer``; ranking
metrics live in ``sagerec_metrics``.

v1 uses NumPy only. Do not add node2vec or extra ML libraries here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

import sagerec_negatives as negatives
from sagerec_prep import SplitResult

_EPS = 1e-12


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int, got {type(value).__name__}")
    return value


def _require_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a real number, got {type(value).__name__}")
    return float(value)


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


@dataclass(frozen=True)
class MFConfig:
    """Explicit hyperparameters for the logistic MF baseline."""

    n_factors: int = 8
    n_epochs: int = 16
    learning_rate: float = 0.05
    n_negatives: int = 4
    l2: float = 0.01
    seed: int = 0
    init_std: float = 0.1

    def __post_init__(self) -> None:
        n_factors = _require_int(self.n_factors, "n_factors")
        n_epochs = _require_int(self.n_epochs, "n_epochs")
        n_negatives = _require_int(self.n_negatives, "n_negatives")
        seed = _require_int(self.seed, "seed")
        learning_rate = _require_float(self.learning_rate, "learning_rate")
        l2 = _require_float(self.l2, "l2")
        init_std = _require_float(self.init_std, "init_std")
        if n_factors < 1:
            raise ValueError(f"n_factors must be >= 1, got {n_factors}")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")
        if n_negatives < 1:
            raise ValueError(f"n_negatives must be >= 1, got {n_negatives}")
        if seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if l2 < 0.0:
            raise ValueError(f"l2 must be >= 0, got {l2}")
        if init_std <= 0.0:
            raise ValueError(f"init_std must be > 0, got {init_std}")


class ImplicitMF:
    """Seeded implicit MF. Implements ``PairScorer.score_pairs``."""

    def __init__(
        self,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        user_bias: np.ndarray,
        item_bias: np.ndarray,
        *,
        config: MFConfig,
    ) -> None:
        if user_factors.ndim != 2 or item_factors.ndim != 2:
            raise ValueError("user_factors and item_factors must be 2-dimensional")
        if user_factors.shape[1] != item_factors.shape[1]:
            raise ValueError(
                "user_factors and item_factors must share the factor dimension, "
                f"got {user_factors.shape[1]} and {item_factors.shape[1]}"
            )
        if user_bias.shape != (user_factors.shape[0],):
            raise ValueError(
                "user_bias must have shape (num_users,), "
                f"got {user_bias.shape} vs num_users={user_factors.shape[0]}"
            )
        if item_bias.shape != (item_factors.shape[0],):
            raise ValueError(
                "item_bias must have shape (num_items,), "
                f"got {item_bias.shape} vs num_items={item_factors.shape[0]}"
            )
        self.user_factors = np.asarray(user_factors, dtype=np.float64)
        self.item_factors = np.asarray(item_factors, dtype=np.float64)
        self.user_bias = np.asarray(user_bias, dtype=np.float64)
        self.item_bias = np.asarray(item_bias, dtype=np.float64)
        self.config = config

    @property
    def num_users(self) -> int:
        return int(self.user_factors.shape[0])

    @property
    def num_items(self) -> int:
        return int(self.item_factors.shape[0])

    def score_pairs(
        self,
        user_ids: Sequence[int],
        item_ids: Sequence[int],
    ) -> np.ndarray:
        if len(user_ids) != len(item_ids):
            raise ValueError(
                "user_ids and item_ids must have the same length, "
                f"got {len(user_ids)} and {len(item_ids)}"
            )
        users = np.asarray(user_ids, dtype=np.int64)
        items = np.asarray(item_ids, dtype=np.int64)
        if users.size == 0:
            return np.empty(0, dtype=np.float64)
        if np.any(users < 0) or np.any(users >= self.num_users):
            raise ValueError(
                f"user_ids must lie in [0, {self.num_users}), got min={int(users.min())} "
                f"max={int(users.max())}"
            )
        if np.any(items < 0) or np.any(items >= self.num_items):
            raise ValueError(
                f"item_ids must lie in [0, {self.num_items}), got min={int(items.min())} "
                f"max={int(items.max())}"
            )
        dots = np.einsum(
            "ij,ij->i", self.user_factors[users], self.item_factors[items]
        )
        return dots + self.user_bias[users] + self.item_bias[items]


def _init_model(
    num_users: int,
    num_items: int,
    config: MFConfig,
    rng: np.random.Generator,
) -> ImplicitMF:
    user_factors = rng.normal(0.0, config.init_std, size=(num_users, config.n_factors))
    item_factors = rng.normal(0.0, config.init_std, size=(num_items, config.n_factors))
    return ImplicitMF(
        user_factors,
        item_factors,
        np.zeros(num_users, dtype=np.float64),
        np.zeros(num_items, dtype=np.float64),
        config=config,
    )


def fit_implicit_mf(
    train_pairs: Sequence[tuple[int, int]],
    *,
    num_users: int,
    num_items: int,
    config: MFConfig | None = None,
    forbidden_pairs: Sequence[tuple[int, int]] | None = None,
) -> ImplicitMF:
    """Train logistic MF on ``train_pairs`` only.

    ``forbidden_pairs`` defaults to ``train_pairs`` (the training-positive
    scope). Held-out positives must not appear in ``train_pairs``.
    """
    config = MFConfig() if config is None else config
    num_users = _require_int(num_users, "num_users")
    num_items = _require_int(num_items, "num_items")
    if num_users < 1 or num_items < 1:
        raise ValueError(
            f"num_users and num_items must be >= 1, got {num_users}, {num_items}"
        )
    if not train_pairs:
        raise ValueError("train_pairs is empty; MF needs at least one training positive")

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for index, pair in enumerate(train_pairs):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                f"train_pairs[{index}] must be a (user_id, item_id) tuple"
            )
        user_id = _require_int(pair[0], f"train_pairs[{index}].user_id")
        item_id = _require_int(pair[1], f"train_pairs[{index}].item_id")
        if user_id < 0 or user_id >= num_users:
            raise ValueError(
                f"train_pairs[{index}].user_id {user_id} is outside [0, {num_users})"
            )
        if item_id < 0 or item_id >= num_items:
            raise ValueError(
                f"train_pairs[{index}].item_id {item_id} is outside [0, {num_items})"
            )
        key = (user_id, item_id)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    extra_forbidden = list(forbidden_pairs) if forbidden_pairs is not None else []
    forbidden_by_user = negatives.items_by_user(list(pairs) + extra_forbidden)
    for user_id, item_id in pairs:
        forbidden_by_user.setdefault(user_id, set()).add(item_id)
    forbidden_scope = {
        (user_id, item_id)
        for user_id, items in forbidden_by_user.items()
        for item_id in items
    }

    rng = np.random.default_rng(config.seed)
    model = _init_model(num_users, num_items, config, rng)
    user_factors = model.user_factors
    item_factors = model.item_factors
    user_bias = model.user_bias
    item_bias = model.item_bias
    lr = config.learning_rate
    l2 = config.l2

    def update(user_id: int, item_id: int, label: float) -> None:
        score = float(
            user_factors[user_id] @ item_factors[item_id]
            + user_bias[user_id]
            + item_bias[item_id]
        )
        pred = min(1.0 - _EPS, max(_EPS, _sigmoid(score)))
        error = label - pred
        user_vec = user_factors[user_id].copy()
        item_vec = item_factors[item_id].copy()
        user_factors[user_id] += lr * (error * item_vec - l2 * user_vec)
        item_factors[item_id] += lr * (error * user_vec - l2 * item_vec)
        user_bias[user_id] += lr * (error - l2 * user_bias[user_id])
        item_bias[item_id] += lr * (error - l2 * item_bias[item_id])

    for _epoch in range(config.n_epochs):
        order = rng.permutation(len(pairs))
        for index in order:
            user_id, item_id = pairs[int(index)]
            update(user_id, item_id, 1.0)
            forbidden_items = forbidden_by_user.get(user_id, set())
            for _neg in range(config.n_negatives):
                negative_item = negatives.sample_negative_item(
                    user_id,
                    num_items=num_items,
                    forbidden_items=forbidden_items,
                    rng=rng,
                )
                negatives.assert_no_positive_overlap(
                    [(user_id, negative_item)], forbidden_scope
                )
                update(user_id, negative_item, 0.0)

    return model


def train_implicit_mf(
    split: SplitResult,
    config: MFConfig | None = None,
) -> ImplicitMF:
    """Fit MF on ADR-003 training positives only."""
    train_pairs = split.train_positive_pairs()
    held_out = split.positive_pair_set("validation") | split.positive_pair_set("test")
    overlap = set(train_pairs) & held_out
    if overlap:
        user_id, item_id = next(iter(overlap))
        raise ValueError(
            "held-out positive leaked into MF train pairs: "
            f"(user_id={user_id}, item_id={item_id})"
        )
    return fit_implicit_mf(
        train_pairs,
        num_users=split.num_users,
        num_items=split.num_movies,
        config=config,
    )
