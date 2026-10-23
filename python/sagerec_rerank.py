"""Two-stage re-ranking on top of embedding retrieval (ADR-010).

Stage 1 is the existing dot-product retrieval (ADR-009). Stage 2 is a small
NumPy MLP with a linear skip, trained listwise on the retrieved pool. Features
are the retrieval score, train-only item popularity, train-only genre affinity,
and train-only item recency. Held-out interactions never enter those tables.

This module does not import PyTorch, PyG, or ``graph_sampler``, and it does
not download MovieLens or train on an HTTP request. MovieLens orchestration
lives in ``sagerec_rerank_eval``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import sagerec_metrics as metrics
from sagerec_prep import SplitName, SplitResult

ADR_ID = "ADR-010"
RERANK_SCHEMA = "sagerec_rerank_artifact"
RERANK_SCHEMA_VERSION = 1
DEFAULT_STAGE1_N = 200
DEFAULT_K = 10
DEFAULT_HIDDEN = 8
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_L2 = 1e-4
DEFAULT_SEED = 7
DEFAULT_INIT_STD = 0.1
FEATURE_NAMES = (
    "retrieval_score",
    "item_popularity",
    "genre_affinity",
    "item_recency",
)
N_FEATURES = len(FEATURE_NAMES)
# MovieLens 100K u.item genre columns, in file order, including "unknown".
GENRE_NAMES = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)
TIE_BREAK_POLICY = "score descending, then movie_id ascending"
SCORE_FORMULA = (
    "relu(features @ w1 + b1) @ w2 + b2 + features @ w_linear; "
    "features are standardized with the training mean and std"
)
MAP_DEFINITION = (
    "MAP@10. One relevant item per eligible user. AP@10 is 1/rank when the "
    "held-out item's 1-based rank is <= 10, else 0. MAP is the unweighted "
    "mean over eligible users. Both systems rank inside the same stage-1 "
    "top-N pool, so retrieval-only top-10 matches full-catalog top-10 when "
    "N >= 10 and the tie-break matches."
)
_STD_FLOOR = 1e-12
_FLOAT_DTYPES = (np.float32, np.float64)


class RerankError(ValueError):
    """Actionable failure while building features, training, or re-ranking."""


@dataclass(frozen=True)
class RerankConfig:
    """Explicit stage-2 hyperparameters. Stage 1 width is ``stage1_n``."""

    stage1_n: int = DEFAULT_STAGE1_N
    hidden_dim: int = DEFAULT_HIDDEN
    n_epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    l2: float = DEFAULT_L2
    seed: int = DEFAULT_SEED
    k: int = DEFAULT_K
    init_std: float = DEFAULT_INIT_STD

    def __post_init__(self) -> None:
        stage1_n = _require_int(self.stage1_n, "stage1_n")
        hidden_dim = _require_int(self.hidden_dim, "hidden_dim")
        n_epochs = _require_int(self.n_epochs, "n_epochs")
        seed = _require_int(self.seed, "seed")
        k = _require_int(self.k, "k")
        learning_rate = _require_float(self.learning_rate, "learning_rate")
        l2 = _require_float(self.l2, "l2")
        init_std = _require_float(self.init_std, "init_std")
        if stage1_n < 1:
            raise RerankError(f"stage1_n must be >= 1, got {stage1_n}")
        if k < 1:
            raise RerankError(f"k must be >= 1, got {k}")
        if stage1_n < k:
            raise RerankError(
                f"stage1_n must be >= k so stage 2 can fill top-k, "
                f"got stage1_n={stage1_n} k={k}"
            )
        if hidden_dim < 1:
            raise RerankError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if n_epochs < 1:
            raise RerankError(f"n_epochs must be >= 1, got {n_epochs}")
        if seed < 0:
            raise RerankError(f"seed must be >= 0, got {seed}")
        if learning_rate <= 0.0:
            raise RerankError(f"learning_rate must be > 0, got {learning_rate}")
        if l2 < 0.0:
            raise RerankError(f"l2 must be >= 0, got {l2}")
        if init_std <= 0.0:
            raise RerankError(f"init_std must be > 0, got {init_std}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage1_n": self.stage1_n,
            "hidden_dim": self.hidden_dim,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "seed": self.seed,
            "k": self.k,
            "init_std": self.init_std,
        }


@dataclass(frozen=True)
class FeatureTables:
    """Train-only side information used to build stage-2 features."""

    item_popularity: np.ndarray
    item_recency: np.ndarray
    item_genres: np.ndarray
    user_genre_profile: np.ndarray
    genres_available: bool

    @property
    def num_users(self) -> int:
        return int(self.user_genre_profile.shape[0])

    @property
    def num_items(self) -> int:
        return int(self.item_popularity.shape[0])

    @property
    def n_genres(self) -> int:
        return int(self.item_genres.shape[1])


@dataclass(frozen=True)
class CandidatePool:
    """One user's stage-1 pool. ``positive_index`` is -1 when the label missed."""

    user_id: int
    item_ids: np.ndarray
    retrieval_scores: np.ndarray
    relevant_id: int
    positive_index: int


@dataclass
class RerankBundle:
    """Exported stage-2 model plus the feature tables it was trained against."""

    item_popularity: np.ndarray
    item_recency: np.ndarray
    item_genres: np.ndarray
    user_genre_profile: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w_linear: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    metadata: dict[str, Any]

    @property
    def num_users(self) -> int:
        return int(self.user_genre_profile.shape[0])

    @property
    def num_items(self) -> int:
        return int(self.item_popularity.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.w1.shape[1])


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RerankError(f"{field} must be an int, got {type(value).__name__}")
    return int(value)


def _require_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise RerankError(f"{field} must be a real number, got {type(value).__name__}")
    return float(value)


def load_item_genres(
    path: str | Path,
    num_items: int,
    movie_source_ids: Sequence[int],
) -> np.ndarray:
    """Align MovieLens ``u.item`` genre flags to local movie ids.

    Returns a float64 array of shape ``(num_items, 19)``. Source ids come from
    ``mappings.json`` ``movie_source_ids`` (local rank to source id).
    """
    file_path = Path(path)
    num_items = _require_int(num_items, "num_items")
    if num_items < 1:
        raise RerankError(f"num_items must be >= 1, got {num_items}")
    if len(movie_source_ids) != num_items:
        raise RerankError(
            f"movie_source_ids length {len(movie_source_ids)} does not match "
            f"num_items={num_items}"
        )
    if not file_path.is_file():
        raise RerankError(
            f"u.item path {file_path} does not exist. Genre affinity needs the "
            "MovieLens 100K genre flags. This loader does not download them."
        )
    try:
        text = file_path.read_text(encoding="latin-1")
    except OSError as exc:
        raise RerankError(f"could not read u.item at {file_path}: {exc}") from exc
    return parse_uitem_genres(text, num_items, movie_source_ids)


def parse_uitem_genres(
    text: str,
    num_items: int,
    movie_source_ids: Sequence[int],
) -> np.ndarray:
    """Parse pipe-separated ``u.item`` text into local-id genre rows."""
    num_items = _require_int(num_items, "num_items")
    if len(movie_source_ids) != num_items:
        raise RerankError(
            f"movie_source_ids length {len(movie_source_ids)} does not match "
            f"num_items={num_items}"
        )
    n_genres = len(GENRE_NAMES)
    by_source: dict[int, np.ndarray] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 5 + n_genres:
            raise RerankError(
                f"u.item line {line_number} has {len(parts)} fields; expected at "
                f"least {5 + n_genres} (id, title, dates, url, {n_genres} genres)"
            )
        try:
            source_id = int(parts[0])
        except ValueError as exc:
            raise RerankError(
                f"u.item line {line_number} has a non-integer movie id"
            ) from exc
        if source_id in by_source:
            raise RerankError(
                f"u.item line {line_number} repeats source movie id {source_id}"
            )
        flags = np.empty(n_genres, dtype=np.float64)
        for offset, raw in enumerate(parts[5 : 5 + n_genres]):
            if raw not in ("0", "1"):
                raise RerankError(
                    f"u.item line {line_number} genre {GENRE_NAMES[offset]!r} "
                    f"must be 0 or 1, got {raw!r}"
                )
            flags[offset] = float(raw)
        by_source[source_id] = flags
    rows = np.zeros((num_items, n_genres), dtype=np.float64)
    missing: list[int] = []
    for local_id, source_id in enumerate(movie_source_ids):
        source_int = _require_int(source_id, f"movie_source_ids[{local_id}]")
        flags = by_source.get(source_int)
        if flags is None:
            missing.append(source_int)
            continue
        rows[local_id] = flags
    if missing:
        preview = ", ".join(str(source_id) for source_id in missing[:8])
        raise RerankError(
            f"u.item is missing genre flags for source movie ids {preview}"
        )
    return rows


def build_feature_tables(
    split: SplitResult,
    genres: np.ndarray | None = None,
) -> FeatureTables:
    """Popularity, recency, and genre profiles from **training** interactions.

    Validation and test rows are ignored. Popularity is ``log1p`` of the train
    interaction count. Recency min-max normalizes each item's latest train
    timestamp; items with no train interaction stay 0. When every train item
    shares one timestamp, those items score 1. Genre affinity uses the user's
    mean train genre vector. ``genres is None`` stores a zero genre axis and
    sets ``genres_available`` false.
    """
    num_users = _require_int(split.num_users, "split.num_users")
    num_items = _require_int(split.num_movies, "split.num_movies")
    if genres is None:
        item_genres = np.zeros((num_items, 1), dtype=np.float64)
        genres_available = False
    else:
        item_genres = np.asarray(genres, dtype=np.float64)
        if item_genres.ndim != 2 or item_genres.shape[0] != num_items:
            raise RerankError(
                "genres must have shape (num_movies, n_genres), "
                f"got {item_genres.shape} for num_movies={num_items}"
            )
        if item_genres.shape[1] < 1 or not np.isfinite(item_genres).all():
            raise RerankError("genres must be finite and have at least one column")
        genres_available = True
    n_genres = int(item_genres.shape[1])
    counts = np.zeros(num_items, dtype=np.float64)
    max_ts = np.full(num_items, -1, dtype=np.int64)
    user_sum = np.zeros((num_users, n_genres), dtype=np.float64)
    user_count = np.zeros(num_users, dtype=np.float64)
    for row in split.interactions_for("train"):
        movie_id = int(row.movie_id)
        user_id = int(row.user_id)
        if movie_id < 0 or movie_id >= num_items or user_id < 0 or user_id >= num_users:
            raise RerankError(
                f"train interaction ({user_id}, {movie_id}) is outside "
                f"users [0, {num_users}) movies [0, {num_items})"
            )
        counts[movie_id] += 1.0
        timestamp = int(row.timestamp)
        if timestamp > int(max_ts[movie_id]):
            max_ts[movie_id] = timestamp
        user_sum[user_id] += item_genres[movie_id]
        user_count[user_id] += 1.0
    popularity = np.log1p(counts)
    recency = np.zeros(num_items, dtype=np.float64)
    observed = max_ts >= 0
    if bool(observed.any()):
        observed_ts = max_ts[observed].astype(np.float64)
        lo = float(observed_ts.min())
        hi = float(observed_ts.max())
        if hi > lo:
            recency[observed] = (observed_ts - lo) / (hi - lo)
        else:
            recency[observed] = 1.0
    profile = np.zeros((num_users, n_genres), dtype=np.float64)
    nonzero = user_count > 0.0
    profile[nonzero] = user_sum[nonzero] / user_count[nonzero, None]
    return FeatureTables(
        item_popularity=popularity,
        item_recency=recency,
        item_genres=np.asarray(item_genres, dtype=np.float64),
        user_genre_profile=profile,
        genres_available=genres_available,
    )


def candidate_features(
    tables: FeatureTables,
    user_id: int,
    item_ids: Sequence[int] | np.ndarray,
    retrieval_scores: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Build the fixed 4-column feature matrix for one retrieved pool.

    Column order is ``FEATURE_NAMES``: retrieval score, item popularity,
    genre affinity (user profile dot item genres), item recency.
    """
    user_id = _require_int(user_id, "user_id")
    if user_id < 0 or user_id >= tables.num_users:
        raise RerankError(
            f"user_id {user_id} is outside [0, {tables.num_users})"
        )
    items = np.asarray(list(item_ids), dtype=np.int64)
    scores = np.asarray(list(retrieval_scores), dtype=np.float64)
    if items.ndim != 1 or scores.ndim != 1 or items.shape != scores.shape:
        raise RerankError(
            "item_ids and retrieval_scores must be 1-d and the same length, "
            f"got {items.shape} and {scores.shape}"
        )
    if items.size and (int(items.min()) < 0 or int(items.max()) >= tables.num_items):
        raise RerankError(
            f"item_ids must lie in [0, {tables.num_items})"
        )
    if scores.size and not np.isfinite(scores).all():
        raise RerankError("retrieval_scores must be finite")
    if items.size == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float64)
    affinity = tables.item_genres[items] @ tables.user_genre_profile[user_id]
    return np.column_stack(
        (
            scores,
            tables.item_popularity[items],
            np.asarray(affinity, dtype=np.float64),
            tables.item_recency[items],
        )
    )


def _items_by_user(split: SplitResult, name: SplitName) -> dict[int, set[int]]:
    by_user: dict[int, set[int]] = defaultdict(set)
    for row in split.interactions_for(name):
        by_user[int(row.user_id)].add(int(row.movie_id))
    return dict(by_user)


def top_n_item_ids(
    scores: np.ndarray,
    exclude: set[int],
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Highest-score unseen items. Ties break on the smaller item id."""
    n = _require_int(n, "n")
    if n < 1:
        raise RerankError(f"n must be >= 1, got {n}")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise RerankError(f"scores must be 1-d, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise RerankError("retrieval scores must be finite before masking")
    masked = values.copy()
    if exclude:
        drop = np.fromiter(exclude, dtype=np.int64, count=len(exclude))
        if drop.size and (int(drop.min()) < 0 or int(drop.max()) >= masked.shape[0]):
            raise RerankError(
                f"exclude ids must lie in [0, {masked.shape[0]})"
            )
        masked[drop] = -np.inf
    n_unseen = int(np.isfinite(masked).sum())
    take = min(n, n_unseen)
    if take <= 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, np.zeros(0, dtype=np.float64)
    order = np.lexsort((np.arange(masked.shape[0], dtype=np.int64), -masked))
    chosen = order[:take]
    return chosen.astype(np.int64, copy=False), values[chosen]


def build_candidate_pools(
    split: SplitResult,
    score_user: Callable[[int], np.ndarray],
    *,
    label_split: SplitName,
    extra_exclude_splits: Sequence[SplitName] = (),
    stage1_n: int,
) -> list[CandidatePool]:
    """Stage-1 pools for eligible users.

    ``label_split`` is the single relevant item. ``extra_exclude_splits`` are
    removed from the pool (training drops the test item; test eval drops the
    validation item). Train positives are always removed. The relevant item is
    kept even if a caller also listed its split in ``extra_exclude_splits``.
    """
    if label_split not in ("validation", "test"):
        raise RerankError(
            f"label_split must be 'validation' or 'test', got {label_split!r}"
        )
    stage1_n = _require_int(stage1_n, "stage1_n")
    if stage1_n < 1:
        raise RerankError(f"stage1_n must be >= 1, got {stage1_n}")
    for name in extra_exclude_splits:
        if name not in ("train", "validation", "test"):
            raise RerankError(f"unknown extra exclude split {name!r}")
    if not split.eligible_user_ids:
        raise RerankError("no eligible users; ADR-003 ranking needs at least 3 interactions")
    train_by_user = _items_by_user(split, "train")
    extra_by_split = [_items_by_user(split, name) for name in extra_exclude_splits]
    label_by_user = _items_by_user(split, label_split)
    pools: list[CandidatePool] = []
    for user_id in split.eligible_user_ids:
        relevant_ids = label_by_user.get(int(user_id), set())
        if len(relevant_ids) != 1:
            raise RerankError(
                f"eligible user {user_id} must have exactly one {label_split} "
                f"item, got {sorted(relevant_ids)}"
            )
        relevant_id = next(iter(relevant_ids))
        exclude = set(train_by_user.get(int(user_id), set()))
        for extra in extra_by_split:
            exclude |= extra.get(int(user_id), set())
        exclude.discard(relevant_id)
        scores = np.asarray(score_user(int(user_id)), dtype=np.float64)
        if scores.shape != (split.num_movies,):
            raise RerankError(
                f"score_user({user_id}) must have shape ({split.num_movies},), "
                f"got {scores.shape}"
            )
        item_ids, retrieval_scores = top_n_item_ids(scores, exclude, stage1_n)
        matches = np.flatnonzero(item_ids == relevant_id)
        positive_index = int(matches[0]) if matches.size else -1
        pools.append(
            CandidatePool(
                user_id=int(user_id),
                item_ids=item_ids,
                retrieval_scores=retrieval_scores,
                relevant_id=int(relevant_id),
                positive_index=positive_index,
            )
        )
    return pools


def _softmax_nll(logits: np.ndarray, positive_index: int) -> tuple[float, np.ndarray]:
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    total = float(exp.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RerankError("listwise softmax received non-finite logits")
    loss = float(-(shifted[positive_index] - np.log(total)))
    grad = exp / total
    grad[positive_index] -= 1.0
    return loss, grad


def _init_mlp(config: RerankConfig, rng: np.random.Generator) -> dict[str, Any]:
    hidden = config.hidden_dim
    return {
        "w1": rng.normal(0.0, config.init_std, size=(N_FEATURES, hidden)),
        "b1": np.full(hidden, 0.1, dtype=np.float64),
        "w2": rng.normal(0.0, config.init_std, size=(hidden,)),
        "b2": 0.0,
        "w_linear": rng.normal(0.0, config.init_std, size=(N_FEATURES,)),
    }


def _forward(
    features: np.ndarray,
    weights: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = features @ weights["w1"] + weights["b1"]
    hidden = np.maximum(pre, 0.0)
    logits = hidden @ weights["w2"] + float(weights["b2"]) + features @ weights["w_linear"]
    return pre, hidden, np.asarray(logits, dtype=np.float64)


def train_listwise_mlp(
    feature_matrices: Sequence[np.ndarray],
    positive_indices: Sequence[int],
    config: RerankConfig,
) -> tuple[dict[str, Any], list[float]]:
    """Listwise softmax SGD. Each matrix is one user's standardized pool.

    Softmax gradients already sum to zero, so they are not divided by the
    pool length. L2 is spread across the epoch (one pass applies ``l2``
    once in total). Returns weights and the mean loss per epoch.
    """
    if len(feature_matrices) != len(positive_indices) or not feature_matrices:
        raise RerankError(
            "train_listwise_mlp needs one non-empty feature matrix and one "
            f"positive index per user, got {len(feature_matrices)} matrices "
            f"and {len(positive_indices)} labels"
        )
    prepared: list[tuple[np.ndarray, int]] = []
    for index, (matrix, positive_index) in enumerate(zip(feature_matrices, positive_indices)):
        features = np.asarray(matrix, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != N_FEATURES:
            raise RerankError(
                f"feature matrix {index} must have shape (pool, {N_FEATURES}), "
                f"got {features.shape}"
            )
        if not np.isfinite(features).all():
            raise RerankError(f"feature matrix {index} must be finite")
        label = _require_int(positive_index, f"positive_indices[{index}]")
        if label < 0 or label >= features.shape[0]:
            raise RerankError(
                f"positive_indices[{index}]={label} is outside the pool of "
                f"length {features.shape[0]}"
            )
        prepared.append((features, label))
    rng = np.random.default_rng(config.seed)
    weights = _init_mlp(config, rng)
    losses: list[float] = []
    lr = config.learning_rate
    l2_step = config.l2 / float(len(prepared))
    for _epoch in range(config.n_epochs):
        order = rng.permutation(len(prepared))
        total = 0.0
        for example_index in order:
            features, label = prepared[int(example_index)]
            pre, hidden, logits = _forward(features, weights)
            loss, grad_logits = _softmax_nll(logits, label)
            if not np.isfinite(loss):
                raise RerankError("listwise loss became non-finite; lower the learning rate")
            grad_w2 = hidden.T @ grad_logits
            grad_b2 = float(grad_logits.sum())
            d_hidden = grad_logits[:, None] * weights["w2"][None, :]
            d_hidden *= pre > 0.0
            grad_w1 = features.T @ d_hidden
            grad_b1 = d_hidden.sum(axis=0)
            grad_linear = features.T @ grad_logits
            weights["w1"] = weights["w1"] - lr * (grad_w1 + l2_step * weights["w1"])
            weights["b1"] = weights["b1"] - lr * grad_b1
            weights["w2"] = weights["w2"] - lr * (grad_w2 + l2_step * weights["w2"])
            weights["b2"] = float(weights["b2"]) - lr * grad_b2
            weights["w_linear"] = weights["w_linear"] - lr * (
                grad_linear + l2_step * weights["w_linear"]
            )
            total += loss
        losses.append(total / float(len(prepared)))
    for key, value in weights.items():
        array = np.asarray(value, dtype=np.float64)
        if not np.isfinite(array).all():
            raise RerankError(f"trained weight {key} is non-finite")
    return weights, losses


def _standardize_fit(matrices: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    flat = np.concatenate([np.asarray(matrix, dtype=np.float64) for matrix in matrices], axis=0)
    if flat.ndim != 2 or flat.shape[1] != N_FEATURES or flat.shape[0] < 1:
        raise RerankError(f"cannot fit feature norm on shape {flat.shape}")
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < _STD_FLOOR, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def _apply_norm(
    matrix: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (np.asarray(matrix, dtype=np.float64) - mean) / std


def _bundle_from_fit(
    tables: FeatureTables,
    weights: Mapping[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    config: RerankConfig,
    *,
    stage1_model: str,
    train_stats: Mapping[str, Any],
) -> RerankBundle:
    if not isinstance(stage1_model, str) or not stage1_model:
        raise RerankError("stage1_model must be a non-empty string")
    metadata: dict[str, Any] = {
        "schema": RERANK_SCHEMA,
        "schema_version": RERANK_SCHEMA_VERSION,
        "adr": ADR_ID,
        "stage1_model": stage1_model,
        "num_users": tables.num_users,
        "num_items": tables.num_items,
        "n_genres": tables.n_genres,
        "hidden_dim": config.hidden_dim,
        "feature_names": list(FEATURE_NAMES),
        "genre_names": list(GENRE_NAMES) if tables.n_genres == len(GENRE_NAMES) else [
            f"genre_{index}" for index in range(tables.n_genres)
        ],
        "genres_available": bool(tables.genres_available),
        "stage1_n": config.stage1_n,
        "score_formula": SCORE_FORMULA,
        "tie_break": TIE_BREAK_POLICY,
        "hyperparams": config.as_dict(),
        "train": dict(train_stats),
    }
    bundle = RerankBundle(
        item_popularity=np.asarray(tables.item_popularity, dtype=np.float64),
        item_recency=np.asarray(tables.item_recency, dtype=np.float64),
        item_genres=np.asarray(tables.item_genres, dtype=np.float64),
        user_genre_profile=np.asarray(tables.user_genre_profile, dtype=np.float64),
        w1=np.asarray(weights["w1"], dtype=np.float64),
        b1=np.asarray(weights["b1"], dtype=np.float64),
        w2=np.asarray(weights["w2"], dtype=np.float64),
        b2=np.asarray([float(weights["b2"])], dtype=np.float64),
        w_linear=np.asarray(weights["w_linear"], dtype=np.float64),
        feature_mean=np.asarray(mean, dtype=np.float64),
        feature_std=np.asarray(std, dtype=np.float64),
        metadata=metadata,
    )
    _validate_bundle(bundle)
    return bundle


def fit_reranker(
    split: SplitResult,
    score_user: Callable[[int], np.ndarray],
    tables: FeatureTables,
    config: RerankConfig | None = None,
    *,
    stage1_model: str,
) -> tuple[RerankBundle, dict[str, Any]]:
    """Fit stage 2 on validation labels inside the stage-1 pool.

    Training pools exclude train positives and test positives. Users whose
    validation item was not retrieved are skipped and counted. Feature mean
    and std are fit on the training pools only.
    """
    config = RerankConfig() if config is None else config
    if tables.num_users != split.num_users or tables.num_items != split.num_movies:
        raise RerankError(
            "feature tables do not match the split "
            f"(tables {tables.num_users}x{tables.num_items}, "
            f"split {split.num_users}x{split.num_movies})"
        )
    pools = build_candidate_pools(
        split,
        score_user,
        label_split="validation",
        extra_exclude_splits=("test",),
        stage1_n=config.stage1_n,
    )
    labeled = [pool for pool in pools if pool.positive_index >= 0]
    n_eligible = len(pools)
    if not labeled:
        raise RerankError(
            "no eligible user had the validation item inside the stage-1 "
            f"top-{config.stage1_n}. Increase stage1_n or check the retriever. "
            "Refusing to train a re-ranker with no positives."
        )
    raw = [
        candidate_features(tables, pool.user_id, pool.item_ids, pool.retrieval_scores)
        for pool in labeled
    ]
    mean, std = _standardize_fit(raw)
    standardized = [_apply_norm(matrix, mean, std) for matrix in raw]
    weights, losses = train_listwise_mlp(
        standardized,
        [pool.positive_index for pool in labeled],
        config,
    )
    stats = {
        "n_eligible_users": n_eligible,
        "n_train_users": len(labeled),
        "validation_hit_rate_at_n": len(labeled) / float(n_eligible),
        "loss_first_epoch": float(losses[0]),
        "loss_final_epoch": float(losses[-1]),
        "n_epochs": config.n_epochs,
        "label_split": "validation",
        "excluded_from_training_pools": ["train", "test"],
    }
    bundle = _bundle_from_fit(
        tables,
        weights,
        mean,
        std,
        config,
        stage1_model=stage1_model,
        train_stats=stats,
    )
    return bundle, stats


def _logits(bundle: RerankBundle, features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return np.zeros(0, dtype=np.float64)
    normed = _apply_norm(features, bundle.feature_mean, bundle.feature_std)
    _pre, _hidden, logits = _forward(
        normed,
        {
            "w1": bundle.w1,
            "b1": bundle.b1,
            "w2": bundle.w2,
            "b2": float(bundle.b2[0]),
            "w_linear": bundle.w_linear,
        },
    )
    if not np.isfinite(logits).all():
        raise RerankError("re-ranker produced non-finite scores")
    return logits


def order_candidates(
    bundle: RerankBundle,
    user_id: int,
    item_ids: Sequence[int],
    retrieval_scores: Sequence[float],
) -> list[tuple[int, float]]:
    """Re-rank a stage-1 pool. Score descending, then smaller ``item_id``."""
    user_id = _require_int(user_id, "user_id")
    if user_id < 0 or user_id >= bundle.num_users:
        raise RerankError(
            f"user_id {user_id} is outside the re-rank artifact [0, {bundle.num_users})"
        )
    items = np.asarray(list(item_ids), dtype=np.int64)
    if items.size == 0:
        return []
    tables = FeatureTables(
        item_popularity=bundle.item_popularity,
        item_recency=bundle.item_recency,
        item_genres=bundle.item_genres,
        user_genre_profile=bundle.user_genre_profile,
        genres_available=bool(bundle.metadata.get("genres_available")),
    )
    features = candidate_features(tables, user_id, items, retrieval_scores)
    logits = _logits(bundle, features)
    order = np.lexsort((items, -logits))
    return [(int(items[index]), float(logits[index])) for index in order]


def _metric_block(
    ranked_by_user: Sequence[Sequence[int]],
    relevant_by_user: Sequence[int],
    k: int,
) -> dict[str, float]:
    recalls: list[float] = []
    ndcgs: list[float] = []
    aps: list[float] = []
    for ranked, relevant_id in zip(ranked_by_user, relevant_by_user):
        relevant = {int(relevant_id)}
        recalls.append(metrics.recall_at_k(ranked, relevant, k))
        ndcgs.append(metrics.ndcg_at_k(ranked, relevant, k))
        aps.append(metrics.average_precision_at_k(ranked, relevant, k))
    n_users = float(len(relevant_by_user))
    return {
        "recall_at_k": sum(recalls) / n_users,
        "ndcg_at_k": sum(ndcgs) / n_users,
        "map_at_k": sum(aps) / n_users,
    }


def _rank_rerank(bundle: RerankBundle, pool: CandidatePool) -> list[int]:
    ordered = order_candidates(
        bundle,
        pool.user_id,
        pool.item_ids.tolist(),
        pool.retrieval_scores.tolist(),
    )
    return [item_id for item_id, _score in ordered]


def evaluate_two_stage(
    split: SplitResult,
    score_user: Callable[[int], np.ndarray],
    tables: FeatureTables,
    config: RerankConfig | None = None,
    *,
    stage1_model: str,
) -> dict[str, Any]:
    """Train on validation pools and score the test split both ways.

    Test pools exclude train and validation positives. Retrieval-only keeps
    the stage-1 order. Retrieval plus re-rank reorders that same pool.
    """
    config = RerankConfig() if config is None else config
    bundle, train_stats = fit_reranker(
        split,
        score_user,
        tables,
        config,
        stage1_model=stage1_model,
    )
    test_pools = build_candidate_pools(
        split,
        score_user,
        label_split="test",
        extra_exclude_splits=("validation",),
        stage1_n=config.stage1_n,
    )
    retrieval_ranked = [pool.item_ids.tolist() for pool in test_pools]
    rerank_ranked = [_rank_rerank(bundle, pool) for pool in test_pools]
    relevant = [pool.relevant_id for pool in test_pools]
    retrieval_metrics = _metric_block(retrieval_ranked, relevant, config.k)
    rerank_metrics = _metric_block(rerank_ranked, relevant, config.k)
    hits = sum(1 for pool in test_pools if pool.positive_index >= 0)
    n_users = len(test_pools)
    delta = {
        key: rerank_metrics[key] - retrieval_metrics[key]
        for key in ("recall_at_k", "ndcg_at_k", "map_at_k")
    }
    return {
        "k": config.k,
        "stage1_n": config.stage1_n,
        "n_users": n_users,
        "stage1_hit_rate_at_n": hits / float(n_users),
        "retrieval_only": retrieval_metrics,
        "retrieval_plus_rerank": rerank_metrics,
        "delta_retrieval_plus_rerank_minus_retrieval_only": delta,
        "rerank_train": train_stats,
        "bundle": bundle,
        "config": config.as_dict(),
        "stage1_model": stage1_model,
        "genres_available": bool(tables.genres_available),
        "feature_names": list(FEATURE_NAMES),
    }


def _validate_bundle(bundle: RerankBundle) -> None:
    meta = bundle.metadata
    if not isinstance(meta, dict):
        raise RerankError(f"metadata must be a dict, got {type(meta).__name__}")
    if meta.get("schema") != RERANK_SCHEMA:
        raise RerankError(
            f"metadata.schema is {meta.get('schema')!r}, expected {RERANK_SCHEMA!r}"
        )
    if meta.get("schema_version") != RERANK_SCHEMA_VERSION:
        raise RerankError(
            f"metadata.schema_version is {meta.get('schema_version')!r}, "
            f"expected {RERANK_SCHEMA_VERSION}"
        )
    stage1_model = meta.get("stage1_model")
    if not isinstance(stage1_model, str) or not stage1_model:
        raise RerankError("metadata.stage1_model must be a non-empty string")
    if list(meta.get("feature_names") or []) != list(FEATURE_NAMES):
        raise RerankError(
            f"metadata.feature_names must be {list(FEATURE_NAMES)}, "
            f"got {meta.get('feature_names')!r}"
        )
    popularity = np.asarray(bundle.item_popularity)
    recency = np.asarray(bundle.item_recency)
    genres = np.asarray(bundle.item_genres)
    profile = np.asarray(bundle.user_genre_profile)
    if popularity.ndim != 1 or recency.shape != popularity.shape:
        raise RerankError(
            "item_popularity and item_recency must be 1-d and the same length, "
            f"got {popularity.shape} and {recency.shape}"
        )
    num_items = int(popularity.shape[0])
    num_users = int(profile.shape[0])
    if num_users < 1 or num_items < 1:
        raise RerankError(
            f"re-rank tables need at least one user and item, got {num_users} x {num_items}"
        )
    if genres.ndim != 2 or genres.shape[0] != num_items or profile.shape != (num_users, genres.shape[1]):
        raise RerankError(
            "item_genres and user_genre_profile must share a genre axis, "
            f"got genres {genres.shape} profile {profile.shape}"
        )
    if genres.shape[1] < 1:
        raise RerankError("item_genres must have at least one column")
    w1 = np.asarray(bundle.w1)
    b1 = np.asarray(bundle.b1)
    w2 = np.asarray(bundle.w2)
    b2 = np.asarray(bundle.b2)
    linear = np.asarray(bundle.w_linear)
    mean = np.asarray(bundle.feature_mean)
    std = np.asarray(bundle.feature_std)
    if w1.shape != (N_FEATURES, w1.shape[1] if w1.ndim == 2 else 0):
        raise RerankError(f"w1 must have shape ({N_FEATURES}, hidden), got {w1.shape}")
    hidden = int(w1.shape[1])
    if hidden < 1 or b1.shape != (hidden,) or w2.shape != (hidden,):
        raise RerankError(
            f"hidden weights must match hidden={hidden}, got b1 {b1.shape} w2 {w2.shape}"
        )
    if b2.shape != (1,) or linear.shape != (N_FEATURES,) or mean.shape != (N_FEATURES,):
        raise RerankError(
            "b2, w_linear, and feature_mean must match the feature width, "
            f"got b2 {b2.shape} w_linear {linear.shape} mean {mean.shape}"
        )
    if std.shape != (N_FEATURES,) or np.any(std <= 0.0):
        raise RerankError(
            f"feature_std must be positive with shape ({N_FEATURES},), got {std}"
        )
    arrays = (popularity, recency, genres, profile, w1, b1, w2, b2, linear, mean, std)
    if any(array.dtype != np.float64 or not np.isfinite(array).all() for array in arrays):
        raise RerankError("re-rank arrays must be finite float64")
    for field, expected in (
        ("num_users", num_users),
        ("num_items", num_items),
        ("n_genres", int(genres.shape[1])),
        ("hidden_dim", hidden),
    ):
        if meta.get(field) != expected:
            raise RerankError(
                f"metadata.{field} is {meta.get(field)!r}, expected {expected}"
            )


def assert_rerank_compatible(
    bundle: RerankBundle,
    *,
    num_users: int,
    num_items: int,
    stage1_model: str,
) -> None:
    """Fail when a serving catalog and a re-rank artifact disagree."""
    _validate_bundle(bundle)
    if bundle.num_users != num_users or bundle.num_items != num_items:
        raise RerankError(
            "re-rank artifact catalog is "
            f"{bundle.num_users} users x {bundle.num_items} items, but the "
            f"serving artifact is {num_users} x {num_items}. Export both from "
            "the same checkpoint."
        )
    trained_for = bundle.metadata["stage1_model"]
    if trained_for != stage1_model:
        raise RerankError(
            f"re-rank artifact stage1_model is {trained_for!r} but the serving "
            f"artifact model is {stage1_model!r}. Train the re-ranker on the "
            "same embedding model, or omit SAGEREC_RERANK_ARTIFACT. "
            "Re-ranking is optional."
        )


def save_rerank_artifact(path: str | Path, bundle: RerankBundle) -> Path:
    """Write ``bundle`` to a compressed ``.npz`` artifact."""
    dest = Path(path)
    if dest.suffix != ".npz":
        raise RerankError(
            f"re-rank artifact path {dest} must end in .npz "
            f"(schema {RERANK_SCHEMA!r})"
        )
    _validate_bundle(bundle)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle.metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    np.savez_compressed(
        dest,
        item_popularity=np.asarray(bundle.item_popularity, dtype=np.float64),
        item_recency=np.asarray(bundle.item_recency, dtype=np.float64),
        item_genres=np.asarray(bundle.item_genres, dtype=np.float64),
        user_genre_profile=np.asarray(bundle.user_genre_profile, dtype=np.float64),
        w1=np.asarray(bundle.w1, dtype=np.float64),
        b1=np.asarray(bundle.b1, dtype=np.float64),
        w2=np.asarray(bundle.w2, dtype=np.float64),
        b2=np.asarray(bundle.b2, dtype=np.float64),
        w_linear=np.asarray(bundle.w_linear, dtype=np.float64),
        feature_mean=np.asarray(bundle.feature_mean, dtype=np.float64),
        feature_std=np.asarray(bundle.feature_std, dtype=np.float64),
        metadata_json=np.frombuffer(payload, dtype=np.uint8).copy(),
    )
    return dest


def load_rerank_artifact(path: str | Path) -> RerankBundle:
    """Load a schema-v1 re-rank artifact. Missing or corrupt files fail."""
    file_path = Path(path)
    if not file_path.exists():
        raise RerankError(
            f"re-rank artifact {file_path} does not exist. Train one with "
            "scripts/run_rerank_eval.py. The HTTP service does not train a "
            "re-ranker and does not download MovieLens."
        )
    if not file_path.is_file():
        raise RerankError(f"re-rank artifact {file_path} is not a file")
    required = (
        "item_popularity",
        "item_recency",
        "item_genres",
        "user_genre_profile",
        "w1",
        "b1",
        "w2",
        "b2",
        "w_linear",
        "feature_mean",
        "feature_std",
        "metadata_json",
    )
    try:
        with np.load(file_path, allow_pickle=False) as data:
            missing = [key for key in required if key not in data.files]
            if missing:
                raise RerankError(
                    f"re-rank artifact {file_path} is missing arrays {missing}. "
                    f"Expected schema {RERANK_SCHEMA!r} version {RERANK_SCHEMA_VERSION}."
                )
            metadata = json.loads(data["metadata_json"].tobytes().decode("utf-8"))
            bundle = RerankBundle(
                item_popularity=np.array(data["item_popularity"], dtype=np.float64, copy=True),
                item_recency=np.array(data["item_recency"], dtype=np.float64, copy=True),
                item_genres=np.array(data["item_genres"], dtype=np.float64, copy=True),
                user_genre_profile=np.array(data["user_genre_profile"], dtype=np.float64, copy=True),
                w1=np.array(data["w1"], dtype=np.float64, copy=True),
                b1=np.array(data["b1"], dtype=np.float64, copy=True),
                w2=np.array(data["w2"], dtype=np.float64, copy=True),
                b2=np.array(data["b2"], dtype=np.float64, copy=True),
                w_linear=np.array(data["w_linear"], dtype=np.float64, copy=True),
                feature_mean=np.array(data["feature_mean"], dtype=np.float64, copy=True),
                feature_std=np.array(data["feature_std"], dtype=np.float64, copy=True),
                metadata=metadata,
            )
    except RerankError:
        raise
    except Exception as exc:
        raise RerankError(
            f"re-rank artifact {file_path} is corrupt or not a SageRec re-rank "
            f"artifact: {exc}. Expected schema {RERANK_SCHEMA!r} version "
            f"{RERANK_SCHEMA_VERSION}."
        ) from exc
    try:
        _validate_bundle(bundle)
    except RerankError as exc:
        raise RerankError(f"re-rank artifact {file_path} failed validation: {exc}") from exc
    return bundle


def build_synthetic_rerank_bundle(
    *,
    num_users: int,
    num_items: int,
    seed: int = DEFAULT_SEED,
    stage1_model: str,
    stage1_n: int = DEFAULT_STAGE1_N,
    hidden_dim: int = DEFAULT_HIDDEN,
    n_genres: int = len(GENRE_NAMES),
) -> RerankBundle:
    """Seeded noise MLP and feature tables for latency measurement.

    This is not a MovieLens-trained re-ranker and not a ranking-quality
    result. The forward pass still multiplies real arrays so the timed path
    does the same work as a loaded model.
    """
    num_users = _require_int(num_users, "num_users")
    num_items = _require_int(num_items, "num_items")
    seed = _require_int(seed, "seed")
    stage1_n = _require_int(stage1_n, "stage1_n")
    hidden_dim = _require_int(hidden_dim, "hidden_dim")
    n_genres = _require_int(n_genres, "n_genres")
    if not isinstance(stage1_model, str) or not stage1_model:
        raise RerankError("stage1_model must be a non-empty string")
    if min(num_users, num_items, hidden_dim, n_genres, stage1_n) < 1:
        raise RerankError(
            "synthetic re-rank sizes must be >= 1, got "
            f"users={num_users} items={num_items} hidden={hidden_dim} "
            f"genres={n_genres} stage1_n={stage1_n}"
        )
    if seed < 0:
        raise RerankError(f"seed must be >= 0, got {seed}")
    rng = np.random.default_rng(seed)
    tables = FeatureTables(
        item_popularity=np.log1p(rng.integers(0, 50, size=num_items)).astype(np.float64),
        item_recency=rng.random(num_items),
        item_genres=(rng.random((num_items, n_genres)) < 0.2).astype(np.float64),
        user_genre_profile=rng.random((num_users, n_genres)),
        genres_available=False,
    )
    config = RerankConfig(
        stage1_n=stage1_n,
        hidden_dim=hidden_dim,
        n_epochs=1,
        learning_rate=DEFAULT_LEARNING_RATE,
        l2=0.0,
        seed=seed,
        k=min(DEFAULT_K, stage1_n),
        init_std=DEFAULT_INIT_STD,
    )
    weights = _init_mlp(config, rng)
    mean = np.zeros(N_FEATURES, dtype=np.float64)
    std = np.ones(N_FEATURES, dtype=np.float64)
    bundle = _bundle_from_fit(
        tables,
        weights,
        mean,
        std,
        config,
        stage1_model=stage1_model,
        train_stats={
            "kind": "synthetic",
            "note": (
                "Seeded synthetic MLP for ADR-010 latency. Feature tables are "
                "noise, not MovieLens train counts, genres, or timestamps."
            ),
        },
    )
    bundle.metadata["catalog_kind"] = "synthetic"
    bundle.metadata["seed"] = seed
    bundle.metadata["note"] = (
        "Synthetic re-rank artifact for HTTP latency. Not the MovieLens "
        "trained model and not a ranking-quality result."
    )
    _validate_bundle(bundle)
    return bundle


def zero_linear_bundle(
    tables: FeatureTables,
    *,
    stage1_model: str,
    popularity_weight: float = 0.0,
    stage1_n: int = DEFAULT_STAGE1_N,
) -> RerankBundle:
    """Hand-built bundle whose score is ``popularity_weight * popularity``.

    Other features and the hidden layer are zero. Used by tests that need a
    known re-rank order without training.
    """
    hidden = 1
    weights = {
        "w1": np.zeros((N_FEATURES, hidden), dtype=np.float64),
        "b1": np.zeros(hidden, dtype=np.float64),
        "w2": np.zeros(hidden, dtype=np.float64),
        "b2": 0.0,
        "w_linear": np.asarray(
            [0.0, float(popularity_weight), 0.0, 0.0],
            dtype=np.float64,
        ),
    }
    config = RerankConfig(
        stage1_n=max(stage1_n, DEFAULT_K),
        hidden_dim=hidden,
        n_epochs=1,
        seed=0,
        k=DEFAULT_K,
    )
    return _bundle_from_fit(
        tables,
        weights,
        np.zeros(N_FEATURES, dtype=np.float64),
        np.ones(N_FEATURES, dtype=np.float64),
        config,
        stage1_model=stage1_model,
        train_stats={"kind": "hand_built"},
    )
