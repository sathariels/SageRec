"""In-memory MovieLens 100K split and prep helpers (ADR-003).

Takes already-normalized interactions (parser output or equivalent structs)
and assigns train / validation / test with per-user chronological leave-one-out.
Does not download data, read filesystem paths, or build a CSR. Callers pass
``train_positive_pairs()`` into ``BipartiteCSR`` (training positives only).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

SplitName = Literal["train", "validation", "test"]

DATASET_EDITION_100K = "100k"
SPLIT_POLICY_ID = "per_user_chronological_leave_one_out"
SPLIT_POLICY_NAME = "Per-user chronological leave-one-out"
SPLIT_POLICY_VERSION = "adr-003-v1"
MIN_INTERACTIONS_FOR_EVAL = 3
COLD_START_POLICY = "all_to_train_exclude_from_eval"
TIE_BREAK = "timestamp_asc_user_id_asc_movie_id_asc"


class SplitCountBlock(TypedDict):
    total: int
    train: int
    validation: int
    test: int


class ManifestCounts(TypedDict):
    users: int
    movies: int
    interactions: SplitCountBlock
    eligible_users: int
    cold_start_users: int


class DatasetManifest(TypedDict):
    dataset_edition: str
    split_policy_id: str
    split_policy_name: str
    split_policy_version: str
    min_interactions_for_eval: int
    cold_start_policy: str
    tie_break: str
    seed: None
    counts: ManifestCounts
    source_url: str | None
    checksum: str | None
    license: str | None


@dataclass(frozen=True)
class NormalizedInteraction:
    """One normalized user-movie interaction used as split input."""

    user_id: int
    movie_id: int
    timestamp: int
    rating: int = 0


@dataclass(frozen=True)
class SplitAssignment:
    """A normalized interaction plus its ADR-003 split label."""

    user_id: int
    movie_id: int
    timestamp: int
    rating: int
    split: SplitName


@dataclass(frozen=True)
class SplitResult:
    """Deterministic leave-one-out assignment for one in-memory 100K corpus."""

    assignments: tuple[SplitAssignment, ...]
    eligible_user_ids: tuple[int, ...]
    cold_start_user_ids: tuple[int, ...]
    num_users: int
    num_movies: int
    dataset_edition: str = DATASET_EDITION_100K

    def interactions_for(self, split: SplitName) -> tuple[SplitAssignment, ...]:
        if split not in ("train", "validation", "test"):
            raise ValueError(
                f"split must be 'train', 'validation', or 'test', got {split!r}"
            )
        return tuple(item for item in self.assignments if item.split == split)

    def train_positive_pairs(self) -> list[tuple[int, int]]:
        """Train-only ``(user_id, movie_id)`` pairs for ``BipartiteCSR``."""
        return [(item.user_id, item.movie_id) for item in self.interactions_for("train")]

    def positive_pair_set(self, split: SplitName) -> set[tuple[int, int]]:
        return {(item.user_id, item.movie_id) for item in self.interactions_for(split)}


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int, got {type(value).__name__}")
    return value


def _coerce_interaction(item: Any, index: int) -> NormalizedInteraction:
    if isinstance(item, NormalizedInteraction):
        user_id = item.user_id
        movie_id = item.movie_id
        timestamp = item.timestamp
        rating = item.rating
    elif isinstance(item, Mapping):
        try:
            user_id = item["user_id"]
            movie_id = item["movie_id"]
            timestamp = item["timestamp"]
        except KeyError as exc:
            raise ValueError(
                f"interaction[{index}] is missing required field {exc.args[0]!r}"
            ) from exc
        rating = item.get("rating", 0)
    else:
        try:
            user_id = item.user_id
            movie_id = item.movie_id
            timestamp = item.timestamp
        except AttributeError as exc:
            raise ValueError(
                f"interaction[{index}] must provide user_id, movie_id, and timestamp"
            ) from exc
        rating = getattr(item, "rating", 0)

    return NormalizedInteraction(
        user_id=_require_int(user_id, f"interaction[{index}].user_id"),
        movie_id=_require_int(movie_id, f"interaction[{index}].movie_id"),
        timestamp=_require_int(timestamp, f"interaction[{index}].timestamp"),
        rating=_require_int(rating, f"interaction[{index}].rating"),
    )


def _chronological_key(item: NormalizedInteraction) -> tuple[int, int, int]:
    """Documented ADR-003 order: timestamp, then local user_id, then movie_id."""
    return (item.timestamp, item.user_id, item.movie_id)


def split_interactions(
    interactions: Sequence[Any],
    *,
    num_users: int | None = None,
    num_movies: int | None = None,
    dataset_edition: str = DATASET_EDITION_100K,
) -> SplitResult:
    """Assign train/validation/test labels using accepted ADR-003.

    Eligible users (at least ``MIN_INTERACTIONS_FOR_EVAL`` interactions) hold
    out the latest interaction as test and the second-latest as validation
    after sorting by ``(timestamp, user_id, movie_id)``. Cold-start users keep
    every interaction in train and are excluded from ranking eligibility.

    Local IDs are a strictly increasing function of MovieLens source IDs after
    ``parse_movielens_100k``, so this tie-break matches
    ``(timestamp, source_user_id, source_movie_id)``. The parser rejects
    duplicate source user-movie pairs, and this function rejects duplicate
    local pairs. Assignments keep input order; ranking uses the sort key.
    """
    if dataset_edition != DATASET_EDITION_100K:
        raise ValueError(
            "ADR-001 accepts MovieLens 100K only; "
            f"dataset_edition must be {DATASET_EDITION_100K!r}, got {dataset_edition!r}"
        )
    if not interactions:
        raise ValueError(
            "interactions is empty; expected at least one normalized user-movie row"
        )

    records = [_coerce_interaction(item, index) for index, item in enumerate(interactions)]

    seen_pairs: set[tuple[int, int]] = set()
    max_user = -1
    max_movie = -1
    by_user: dict[int, list[NormalizedInteraction]] = {}
    for index, record in enumerate(records):
        if record.user_id < 0:
            raise ValueError(
                f"interaction[{index}].user_id {record.user_id} is negative; "
                "expected a zero-based local user id"
            )
        if record.movie_id < 0:
            raise ValueError(
                f"interaction[{index}].movie_id {record.movie_id} is negative; "
                "expected a zero-based local movie id"
            )
        pair = (record.user_id, record.movie_id)
        if pair in seen_pairs:
            raise ValueError(
                f"duplicate local pair (user_id={record.user_id}, movie_id={record.movie_id}) "
                f"at interaction[{index}]"
            )
        seen_pairs.add(pair)
        by_user.setdefault(record.user_id, []).append(record)
        if record.user_id > max_user:
            max_user = record.user_id
        if record.movie_id > max_movie:
            max_movie = record.movie_id

    inferred_users = max_user + 1
    inferred_movies = max_movie + 1
    resolved_users = inferred_users if num_users is None else _require_int(num_users, "num_users")
    resolved_movies = (
        inferred_movies if num_movies is None else _require_int(num_movies, "num_movies")
    )
    if resolved_users < inferred_users:
        raise ValueError(
            f"num_users {resolved_users} is smaller than required local range "
            f"[0, {inferred_users})"
        )
    if resolved_movies < inferred_movies:
        raise ValueError(
            f"num_movies {resolved_movies} is smaller than required local range "
            f"[0, {inferred_movies})"
        )

    split_by_pair: dict[tuple[int, int], SplitName] = {}
    eligible: list[int] = []
    cold_start: list[int] = []
    for user_id in sorted(by_user):
        ranked = sorted(by_user[user_id], key=_chronological_key)
        if len(ranked) < MIN_INTERACTIONS_FOR_EVAL:
            cold_start.append(user_id)
            for record in ranked:
                split_by_pair[(record.user_id, record.movie_id)] = "train"
            continue
        eligible.append(user_id)
        for record in ranked[:-2]:
            split_by_pair[(record.user_id, record.movie_id)] = "train"
        validation = ranked[-2]
        test = ranked[-1]
        split_by_pair[(validation.user_id, validation.movie_id)] = "validation"
        split_by_pair[(test.user_id, test.movie_id)] = "test"

    assignments = tuple(
        SplitAssignment(
            user_id=record.user_id,
            movie_id=record.movie_id,
            timestamp=record.timestamp,
            rating=record.rating,
            split=split_by_pair[(record.user_id, record.movie_id)],
        )
        for record in records
    )
    return SplitResult(
        assignments=assignments,
        eligible_user_ids=tuple(eligible),
        cold_start_user_ids=tuple(cold_start),
        num_users=resolved_users,
        num_movies=resolved_movies,
        dataset_edition=dataset_edition,
    )


def train_positive_pairs(result: SplitResult) -> list[tuple[int, int]]:
    """Train-only pairs suitable for ``BipartiteCSR`` / ``graph_sampler``."""
    return result.train_positive_pairs()


def build_manifest(
    result: SplitResult,
    *,
    source_url: str | None = None,
    checksum: str | None = None,
    license: str | None = None,
) -> DatasetManifest:
    """JSON-ready provenance record. Checksum/URL are placeholders unless supplied."""
    interaction_counts: SplitCountBlock = {
        "total": len(result.assignments),
        "train": len(result.interactions_for("train")),
        "validation": len(result.interactions_for("validation")),
        "test": len(result.interactions_for("test")),
    }
    counts: ManifestCounts = {
        "users": result.num_users,
        "movies": result.num_movies,
        "interactions": interaction_counts,
        "eligible_users": len(result.eligible_user_ids),
        "cold_start_users": len(result.cold_start_user_ids),
    }
    return {
        "dataset_edition": result.dataset_edition,
        "split_policy_id": SPLIT_POLICY_ID,
        "split_policy_name": SPLIT_POLICY_NAME,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "min_interactions_for_eval": MIN_INTERACTIONS_FOR_EVAL,
        "cold_start_policy": COLD_START_POLICY,
        "tie_break": TIE_BREAK,
        "seed": None,
        "counts": counts,
        "source_url": source_url,
        "checksum": checksum,
        "license": license,
    }
