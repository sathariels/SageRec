"""Precomputed serving index and exact NumPy top-K (ADR-009).

The on-disk artifact stores user and item embeddings, optional biases, the
train-positive items each user has already seen, and item titles. A recommend
call is a dot product plus a mask. FAISS is not required; exact NumPy search
keeps the ADR-008 tie-break (score descending, then smaller ``item_id``).

This module does not import PyTorch, PyTorch Geometric, or ``graph_sampler``,
and it does not download MovieLens. GraphSAGE export materializes embeddings
before anything here is loaded.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SERVING_SCHEMA = "sagerec_serving_artifact"
SERVING_SCHEMA_VERSION = 1
DEFAULT_K = 10
TIE_BREAK_POLICY = "score descending, then movie_id ascending"
SCORE_FORMULA = "dot(user_embedding, item_embedding) + user_bias + item_bias"
TOPK_IMPLEMENTATION = "numpy_exact"
ADR_ID = "ADR-009"
_FLOAT_DTYPES = (np.float32, np.float64)
_REQUIRED_ARRAYS = (
    "user_embeddings",
    "item_embeddings",
    "user_bias",
    "item_bias",
    "seen_offsets",
    "seen_movies",
    "item_titles",
    "metadata_json",
)


class ServingError(ValueError):
    """Actionable failure while reading or ranking a serving artifact."""


class UnknownUserError(ServingError):
    """``user_id`` is an int outside ``[0, num_users)``."""


@dataclass
class ScoredItem:
    """One recommendation row. ``item_id`` is the zero-based movie id."""

    item_id: int
    title: str
    score: float


@dataclass
class ServingIndex:
    """In-memory serving catalog loaded from a ``.npz`` artifact."""

    user_embeddings: np.ndarray
    item_embeddings: np.ndarray
    user_bias: np.ndarray
    item_bias: np.ndarray
    seen_offsets: np.ndarray
    seen_movies: np.ndarray
    item_titles: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def num_users(self) -> int:
        return int(self.user_embeddings.shape[0])

    @property
    def num_items(self) -> int:
        return int(self.item_embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.user_embeddings.shape[1])

    def seen_item_ids(self, user_id: int) -> np.ndarray:
        """Train-positive movie ids excluded for ``user_id``."""
        if user_id < 0 or user_id >= self.num_users:
            raise UnknownUserError(
                f"unknown user_id {user_id}; expected a zero-based id in "
                f"[0, {self.num_users})"
            )
        start = int(self.seen_offsets[user_id])
        end = int(self.seen_offsets[user_id + 1])
        return self.seen_movies[start:end]


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ServingError(f"{field} must be an int, got {type(value).__name__}")
    return int(value)


def build_seen_csr(
    num_users: int,
    num_items: int,
    pairs: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pack train-positive ``(user_id, movie_id)`` pairs into a CSR mask.

    Duplicate pairs collapse. Each user's movie ids are sorted ascending.
    """
    num_users = _require_int(num_users, "num_users")
    num_items = _require_int(num_items, "num_items")
    if num_users < 1 or num_items < 1:
        raise ServingError(
            f"num_users and num_items must be >= 1, got {num_users}, {num_items}"
        )
    buckets: list[set[int]] = [set() for _ in range(num_users)]
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ServingError(
                f"seen pairs[{index}] must be a (user_id, movie_id) pair"
            )
        try:
            user_id = _require_int(int(pair[0]), f"seen pairs[{index}].user_id")
            movie_id = _require_int(int(pair[1]), f"seen pairs[{index}].movie_id")
        except (TypeError, ValueError) as exc:
            raise ServingError(
                f"seen pairs[{index}] must contain integer ids"
            ) from exc
        if user_id < 0 or user_id >= num_users:
            raise ServingError(
                f"seen pairs[{index}].user_id {user_id} is outside [0, {num_users})"
            )
        if movie_id < 0 or movie_id >= num_items:
            raise ServingError(
                f"seen pairs[{index}].movie_id {movie_id} is outside [0, {num_items})"
            )
        buckets[user_id].add(movie_id)
    offsets = [0]
    movies: list[int] = []
    for bucket in buckets:
        movies.extend(sorted(bucket))
        offsets.append(len(movies))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(movies, dtype=np.int64),
    )


def _validate_index(index: ServingIndex) -> None:
    user = np.asarray(index.user_embeddings)
    item = np.asarray(index.item_embeddings)
    user_bias = np.asarray(index.user_bias)
    item_bias = np.asarray(index.item_bias)
    offsets = np.asarray(index.seen_offsets)
    movies = np.asarray(index.seen_movies)
    if user.ndim != 2 or item.ndim != 2:
        raise ServingError(
            "user_embeddings and item_embeddings must be 2-dimensional, "
            f"got {user.shape} and {item.shape}"
        )
    if user.shape[0] < 1 or item.shape[0] < 1 or user.shape[1] < 1:
        raise ServingError(
            "serving embeddings must have at least one user, one item, and "
            f"one dimension, got users={user.shape} items={item.shape}"
        )
    if user.shape[1] != item.shape[1]:
        raise ServingError(
            "user and item embeddings must share a dimension, "
            f"got {user.shape[1]} and {item.shape[1]}"
        )
    if user.dtype != item.dtype or user.dtype not in _FLOAT_DTYPES:
        raise ServingError(
            "user and item embeddings must be float32 or float64 and share a "
            f"dtype, got {user.dtype} and {item.dtype}"
        )
    if user_bias.shape != (user.shape[0],) or user_bias.dtype != user.dtype:
        raise ServingError(
            "user_bias must have shape (num_users,) and the embedding dtype, "
            f"got shape {user_bias.shape} dtype {user_bias.dtype}"
        )
    if item_bias.shape != (item.shape[0],) or item_bias.dtype != user.dtype:
        raise ServingError(
            "item_bias must have shape (num_items,) and the embedding dtype, "
            f"got shape {item_bias.shape} dtype {item_bias.dtype}"
        )
    if not (
        np.isfinite(user).all()
        and np.isfinite(item).all()
        and np.isfinite(user_bias).all()
        and np.isfinite(item_bias).all()
    ):
        raise ServingError("serving embeddings and biases must be finite")
    if offsets.dtype != np.int64 or offsets.shape != (user.shape[0] + 1,):
        raise ServingError(
            "seen_offsets must be int64 with shape (num_users + 1,), "
            f"got dtype {offsets.dtype} shape {offsets.shape}"
        )
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(movies.shape[0]):
        raise ServingError(
            "seen_offsets must start at 0 and end at len(seen_movies), "
            f"got {int(offsets[0])} .. {int(offsets[-1])} vs "
            f"{int(movies.shape[0])} movies"
        )
    if movies.dtype != np.int64 or movies.ndim != 1:
        raise ServingError(
            f"seen_movies must be a 1-d int64 array, got dtype {movies.dtype} "
            f"ndim {movies.ndim}"
        )
    if np.any(np.diff(offsets) < 0):
        raise ServingError("seen_offsets must be non-decreasing")
    num_items = int(item.shape[0])
    for user_id in range(int(user.shape[0])):
        start = int(offsets[user_id])
        end = int(offsets[user_id + 1])
        chunk = movies[start:end]
        if chunk.size and (int(chunk.min()) < 0 or int(chunk.max()) >= num_items):
            raise ServingError(
                f"seen movies for user {user_id} fall outside [0, {num_items})"
            )
        if chunk.size and np.any(np.diff(chunk) <= 0):
            raise ServingError(
                f"seen movies for user {user_id} must be strictly increasing"
            )
    titles = tuple(index.item_titles)
    if len(titles) != num_items:
        raise ServingError(
            f"item_titles length {len(titles)} does not match num_items {num_items}"
        )
    for item_id, title in enumerate(titles):
        if not isinstance(title, str) or not title:
            raise ServingError(
                f"item_titles[{item_id}] must be a non-empty string"
            )
    meta = index.metadata
    if not isinstance(meta, dict):
        raise ServingError(
            f"metadata must be a dict, got {type(meta).__name__}"
        )
    if meta.get("schema") != SERVING_SCHEMA:
        raise ServingError(
            f"metadata.schema is {meta.get('schema')!r}, expected {SERVING_SCHEMA!r}"
        )
    if meta.get("schema_version") != SERVING_SCHEMA_VERSION:
        raise ServingError(
            f"metadata.schema_version is {meta.get('schema_version')!r}, "
            f"expected {SERVING_SCHEMA_VERSION}"
        )
    model = meta.get("model")
    if not isinstance(model, str) or not model:
        raise ServingError("metadata.model must be a non-empty string")
    for field, expected in (
        ("num_users", int(user.shape[0])),
        ("num_items", num_items),
        ("embedding_dim", int(user.shape[1])),
    ):
        if meta.get(field) != expected:
            raise ServingError(
                f"metadata.{field} is {meta.get(field)!r}, expected {expected}"
            )


def save_serving_artifact(path: str | Path, index: ServingIndex) -> Path:
    """Write ``index`` to a compressed ``.npz`` artifact."""
    dest = Path(path)
    if dest.suffix != ".npz":
        raise ServingError(
            f"serving artifact path {dest} must end in .npz "
            f"(schema {SERVING_SCHEMA!r})"
        )
    _validate_index(index)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        index.metadata, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    np.savez_compressed(
        dest,
        user_embeddings=np.asarray(index.user_embeddings),
        item_embeddings=np.asarray(index.item_embeddings),
        user_bias=np.asarray(index.user_bias),
        item_bias=np.asarray(index.item_bias),
        seen_offsets=np.asarray(index.seen_offsets, dtype=np.int64),
        seen_movies=np.asarray(index.seen_movies, dtype=np.int64),
        item_titles=np.asarray(list(index.item_titles)),
        metadata_json=np.frombuffer(payload, dtype=np.uint8).copy(),
    )
    return dest


def _path_display(path: Path) -> str:
    return str(path)


def load_serving_artifact(path: str | Path) -> ServingIndex:
    """Load a schema-v1 serving artifact. Missing or corrupt files fail."""
    file_path = Path(path)
    if not file_path.exists():
        raise ServingError(
            f"serving artifact {_path_display(file_path)!r} does not exist. "
            "Export one with scripts/export_embeddings.py from a "
            "sagerec_checkpoint. The HTTP service does not download MovieLens "
            "or train a model."
        )
    if not file_path.is_file():
        raise ServingError(
            f"serving artifact {_path_display(file_path)!r} is not a file"
        )
    try:
        with np.load(file_path, allow_pickle=False) as data:
            missing = [key for key in _REQUIRED_ARRAYS if key not in data.files]
            if missing:
                raise ServingError(
                    f"serving artifact {_path_display(file_path)!r} is missing "
                    f"arrays {missing}. Expected schema {SERVING_SCHEMA!r} "
                    f"version {SERVING_SCHEMA_VERSION}."
                )
            metadata = json.loads(data["metadata_json"].tobytes().decode("utf-8"))
            index = ServingIndex(
                user_embeddings=np.array(data["user_embeddings"], copy=True),
                item_embeddings=np.array(data["item_embeddings"], copy=True),
                user_bias=np.array(data["user_bias"], copy=True),
                item_bias=np.array(data["item_bias"], copy=True),
                seen_offsets=np.array(data["seen_offsets"], dtype=np.int64, copy=True),
                seen_movies=np.array(data["seen_movies"], dtype=np.int64, copy=True),
                item_titles=tuple(str(title) for title in data["item_titles"].tolist()),
                metadata=metadata,
            )
    except ServingError:
        raise
    except Exception as exc:
        raise ServingError(
            f"serving artifact {_path_display(file_path)!r} is corrupt or not a "
            f"SageRec serving artifact: {exc}. Expected schema {SERVING_SCHEMA!r} "
            f"version {SERVING_SCHEMA_VERSION} written by save_serving_artifact."
        ) from exc
    try:
        _validate_index(index)
    except ServingError as exc:
        raise ServingError(
            f"serving artifact {_path_display(file_path)!r} failed validation: {exc}"
        ) from exc
    return index


def recommend_topk(
    index: ServingIndex,
    user_id: int,
    k: int = DEFAULT_K,
) -> list[ScoredItem]:
    """Exact top-K over items the user has not already seen.

    Score is ``SCORE_FORMULA``. Already-seen train positives are removed
    before ranking. Ties break on smaller ``item_id`` (``TIE_BREAK_POLICY``).
    FAISS is not used.
    """
    user_id = _require_int(user_id, "user_id")
    k = _require_int(k, "k")
    if k < 1:
        raise ServingError(f"k must be >= 1, got {k}")
    if user_id < 0 or user_id >= index.num_users:
        raise UnknownUserError(
            f"unknown user_id {user_id}; expected a zero-based id in "
            f"[0, {index.num_users})"
        )
    dtype = index.user_embeddings.dtype
    user_vec = np.asarray(index.user_embeddings[user_id], dtype=dtype)
    scores = np.asarray(index.item_embeddings @ user_vec, dtype=dtype).copy()
    scores += np.asarray(index.user_bias[user_id], dtype=dtype)
    scores += np.asarray(index.item_bias, dtype=dtype)
    seen = index.seen_item_ids(user_id)
    if seen.size:
        scores[seen] = -np.inf
    n_unseen = int(np.isfinite(scores).sum())
    take = min(k, n_unseen)
    if take <= 0:
        return []
    order = np.lexsort((np.arange(index.num_items, dtype=np.int64), -scores))
    chosen = order[:take]
    return [
        ScoredItem(
            item_id=int(item_id),
            title=index.item_titles[int(item_id)],
            score=float(scores[int(item_id)]),
        )
        for item_id in chosen
    ]


def _metadata(
    *,
    model: str,
    num_users: int,
    num_items: int,
    embedding_dim: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SERVING_SCHEMA,
        "schema_version": SERVING_SCHEMA_VERSION,
        "adr": ADR_ID,
        "model": model,
        "num_users": num_users,
        "num_items": num_items,
        "embedding_dim": embedding_dim,
        "score_formula": SCORE_FORMULA,
        "tie_break": TIE_BREAK_POLICY,
        "topk": TOPK_IMPLEMENTATION,
        "faiss": "not_required",
        "seen_policy": (
            "exclude train-positive movie ids stored in seen_offsets/seen_movies"
        ),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def make_serving_index(
    *,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    seen_pairs: Sequence[tuple[int, int]],
    item_titles: Sequence[str] | None = None,
    user_bias: np.ndarray | None = None,
    item_bias: np.ndarray | None = None,
    model: str,
    extra_metadata: Mapping[str, Any] | None = None,
) -> ServingIndex:
    """Validate arrays and pack a ``ServingIndex`` (not yet written)."""
    user = np.asarray(user_embeddings)
    item = np.asarray(item_embeddings)
    if user.ndim != 2 or item.ndim != 2:
        raise ServingError("embeddings must be 2-dimensional")
    num_users = int(user.shape[0])
    num_items = int(item.shape[0])
    if user_bias is None:
        user_bias_arr = np.zeros(num_users, dtype=user.dtype)
    else:
        user_bias_arr = np.asarray(user_bias)
    if item_bias is None:
        item_bias_arr = np.zeros(num_items, dtype=item.dtype)
    else:
        item_bias_arr = np.asarray(item_bias)
    if item_titles is None:
        titles = tuple(f"movie_{item_id}" for item_id in range(num_items))
    else:
        titles = tuple(item_titles)
    offsets, movies = build_seen_csr(num_users, num_items, seen_pairs)
    index = ServingIndex(
        user_embeddings=user,
        item_embeddings=item,
        user_bias=user_bias_arr,
        item_bias=item_bias_arr,
        seen_offsets=offsets,
        seen_movies=movies,
        item_titles=titles,
        metadata=_metadata(
            model=model,
            num_users=num_users,
            num_items=num_items,
            embedding_dim=int(user.shape[1]) if user.ndim == 2 else 0,
            extra=extra_metadata,
        ),
    )
    _validate_index(index)
    return index


def build_synthetic_serving_index(
    *,
    num_users: int = 943,
    num_items: int = 1682,
    embedding_dim: int = 16,
    seen_per_user: int = 20,
    seed: int = 7,
) -> ServingIndex:
    """Deterministic catalog for serving-latency measurement.

    User and item counts default to the public MovieLens 100K cardinality so
    the dot product is a similar size. Embeddings are standard-normal draws
    from ``seed``. This is not a trained checkpoint and not a ranking-quality
    result. No ratings file is read.
    """
    num_users = _require_int(num_users, "num_users")
    num_items = _require_int(num_items, "num_items")
    embedding_dim = _require_int(embedding_dim, "embedding_dim")
    seen_per_user = _require_int(seen_per_user, "seen_per_user")
    seed = _require_int(seed, "seed")
    if embedding_dim < 1:
        raise ServingError(f"embedding_dim must be >= 1, got {embedding_dim}")
    if seen_per_user < 0 or seen_per_user >= num_items:
        raise ServingError(
            f"seen_per_user must be in [0, num_items), got {seen_per_user} "
            f"for num_items={num_items}"
        )
    if seed < 0:
        raise ServingError(f"seed must be >= 0, got {seed}")
    rng = np.random.default_rng(seed)
    user_embeddings = rng.standard_normal((num_users, embedding_dim)).astype(np.float32)
    item_embeddings = rng.standard_normal((num_items, embedding_dim)).astype(np.float32)
    pairs: list[tuple[int, int]] = []
    for user_id in range(num_users):
        chosen = rng.choice(num_items, size=seen_per_user, replace=False)
        pairs.extend((user_id, int(movie_id)) for movie_id in chosen)
    return make_serving_index(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        seen_pairs=pairs,
        model="synthetic",
        extra_metadata={
            "seed": seed,
            "seen_per_user": seen_per_user,
            "catalog_kind": "synthetic",
            "note": (
                "Synthetic serving catalog for ADR-009 latency measurement. "
                "Embeddings are Generator(seed) standard normal, not a trained "
                "GraphSAGE or MF checkpoint. Default user/item counts match the "
                "public MovieLens 100K cardinality so the dot-product size is "
                "similar. This file is not a ranking-quality result."
            ),
        },
    )
