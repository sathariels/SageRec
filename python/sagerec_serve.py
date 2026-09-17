"""Demo serving for SageRec checkpoints (ADR-008).

Loads a GraphSAGE or PairScorer-compatible implicit-MF checkpoint, scores
caller-supplied candidate movie IDs for one user, and ranks top-K. This is a
local CLI helper, not a batch exporter and not an HTTP API.

GraphSAGE neighborhoods at serve time still come from native
``graph_sampler`` via ``sagerec_minibatch.NativeMinibatchSampler`` (ADR-005,
ADR-006). This module does not import or call a PyG ``NeighborLoader`` /
``ClusterLoader``. It does not download MovieLens.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

import sagerec_baseline as baseline
import sagerec_dataset as dataset
import sagerec_graphsage as graphsage
import sagerec_minibatch as minibatch
from sagerec_scoring import PairScorer

CHECKPOINT_SCHEMA = "sagerec_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
MODEL_GRAPHSAGE = "graphsage"
MODEL_IMPLICIT_MF = "implicit_mf"
SUPPORTED_MODELS = (MODEL_GRAPHSAGE, MODEL_IMPLICIT_MF)
DEFAULT_K = 10
TIE_BREAK_POLICY = "score descending, then movie_id ascending"

_GRAPHSAGE_HYPERPARAM_KEYS = {
    "embedding_dim",
    "hidden_dim",
    "n_layers",
    "fanouts",
    "n_epochs",
    "batch_size",
    "learning_rate",
    "n_negatives",
    "l2",
    "seed",
    "init_std",
    "optimizer",
}
_MF_HYPERPARAM_KEYS = {
    "n_factors",
    "n_epochs",
    "learning_rate",
    "n_negatives",
    "l2",
    "seed",
    "init_std",
}


class ServeError(ValueError):
    """Actionable failure while loading a checkpoint or ranking candidates."""


@dataclass(frozen=True)
class RankedCandidate:
    """One printed recommendation row."""

    rank: int
    movie_id: int
    score: float


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ServeError(f"{field} must be an int, got {type(value).__name__}")
    return value


def _path_display(path: Path) -> str:
    return str(path)


def _require_file(path: Path, *, what: str) -> Path:
    if not path.exists():
        raise ServeError(
            f"{what} {_path_display(path)!r} does not exist. "
            "Pass a real filesystem path; the demo CLI does not download data "
            "or search the network."
        )
    if not path.is_file():
        raise ServeError(
            f"{what} {_path_display(path)!r} is not a file. "
            "Expected a checkpoint or train-pairs JSON path."
        )
    return path


def parse_candidate_ids(text: str) -> list[int]:
    """Parse comma- or whitespace-separated zero-based movie IDs."""
    if not isinstance(text, str):
        raise ServeError(
            f"--candidates must be a string, got {type(text).__name__}"
        )
    parts = [part for part in text.replace(",", " ").split() if part]
    if not parts:
        raise ServeError(
            "--candidates is empty; pass one or more zero-based movie IDs "
            "(for example --candidates 0,3,7)"
        )
    movie_ids: list[int] = []
    seen: set[int] = set()
    for part in parts:
        try:
            movie_id = int(part, 10)
        except ValueError as exc:
            raise ServeError(
                f"invalid movie id {part!r} in --candidates; expected integers"
            ) from exc
        if movie_id in seen:
            raise ServeError(
                f"duplicate movie id {movie_id} in --candidates; "
                "each candidate must appear once"
            )
        seen.add(movie_id)
        movie_ids.append(movie_id)
    return movie_ids


def train_pairs_from_native_sampler(
    sampler: minibatch.NativeMinibatchSampler,
) -> list[tuple[int, int]]:
    """Recover local ``(user_id, movie_id)`` edges from a train-only CSR."""
    if not isinstance(sampler, minibatch.NativeMinibatchSampler):
        raise ServeError(
            "GraphSAGE serve requires sagerec_minibatch.NativeMinibatchSampler, "
            f"got {type(sampler).__name__}. Do not substitute a PyG sampler."
        )
    offsets = sampler.offsets()
    neighbors = sampler.neighbors()
    n_users = sampler.num_users
    pairs: list[tuple[int, int]] = []
    for user_id in range(n_users):
        start = offsets[user_id]
        end = offsets[user_id + 1]
        for global_id in neighbors[start:end]:
            node = int(global_id)
            if node < n_users:
                raise ServeError(
                    f"user {user_id} neighborhood contains user node {node}; "
                    "expected movie global IDs only in the train-only CSR"
                )
            pairs.append((user_id, node - n_users))
    if not pairs:
        raise ServeError(
            "train-only CSR has no user-movie edges; cannot rebuild GraphSAGE "
            "serve-time neighborhoods"
        )
    return pairs


def load_train_pairs_json(path: str | Path) -> list[tuple[int, int]]:
    """Load ``[[user_id, movie_id], ...]`` from a JSON file."""
    file_path = _require_file(Path(path), what="train-pairs JSON")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ServeError(
            f"train-pairs JSON at {_path_display(file_path)!r} is not valid JSON: "
            f"{exc}. Expected a list of [user_id, movie_id] pairs."
        ) from exc
    return _coerce_train_pairs(payload, origin=_path_display(file_path))


def load_train_pairs_from_processed(
    processed_dir: str | Path,
) -> list[tuple[int, int]]:
    """Load ``train_pairs.json`` from an existing processed directory.

    Missing artifacts fail. This never downloads MovieLens.
    """
    dest = Path(processed_dir)
    if not dest.exists():
        raise ServeError(
            f"processed directory {_path_display(dest)!r} does not exist. "
            "GraphSAGE serve needs train-only pairs to rebuild the native CSR. "
            "Pass --train-pairs, --processed-dir, or a checkpoint that embeds "
            "train_pairs. The demo CLI does not download MovieLens."
        )
    if not dest.is_dir():
        raise ServeError(
            f"processed path {_path_display(dest)!r} is not a directory"
        )
    return load_train_pairs_json(dest / dataset.TRAIN_PAIRS_NAME)


def _coerce_train_pairs(payload: Any, *, origin: str) -> list[tuple[int, int]]:
    if not isinstance(payload, list) or not payload:
        raise ServeError(
            f"train_pairs from {origin} must be a non-empty list of "
            "[user_id, movie_id] pairs"
        )
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for index, row in enumerate(payload):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ServeError(
                f"train_pairs[{index}] from {origin} must be a "
                "[user_id, movie_id] pair"
            )
        try:
            user_id = _require_int(int(row[0]), f"train_pairs[{index}].user_id")
            movie_id = _require_int(int(row[1]), f"train_pairs[{index}].movie_id")
        except (TypeError, ValueError) as exc:
            raise ServeError(
                f"train_pairs[{index}] from {origin} must contain integer IDs"
            ) from exc
        key = (user_id, movie_id)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def _pairs_as_lists(pairs: Sequence[tuple[int, int]]) -> list[list[int]]:
    return [[int(user_id), int(movie_id)] for user_id, movie_id in pairs]


def _checkpoint_note(model_name: str) -> str:
    if model_name == MODEL_GRAPHSAGE:
        return (
            "SageRec GraphSAGE checkpoint (ADR-008). Neighborhoods at serve "
            "time come from native graph_sampler via sagerec_minibatch on "
            "train-only pairs; not a PyG NeighborLoader. This file is not a "
            "MovieLens 100K quality result."
        )
    return (
        "SageRec implicit-MF checkpoint (ADR-008). Pair scoring uses stored "
        "factors only; no native sampler is required. This file is not a "
        "MovieLens 100K quality result."
    )


def graphsage_checkpoint_payload(
    model: graphsage.GraphSAGERecommender,
    *,
    train_pairs: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Build a schema-v1 GraphSAGE checkpoint mapping (not yet written)."""
    if not isinstance(model, graphsage.GraphSAGERecommender):
        raise ServeError(
            "save_checkpoint expected sagerec_graphsage.GraphSAGERecommender, "
            f"got {type(model).__name__}"
        )
    pairs = (
        list(train_pairs)
        if train_pairs is not None
        else train_pairs_from_native_sampler(model.sampler)
    )
    if not pairs:
        raise ServeError("train_pairs is empty; GraphSAGE serve needs the train-only CSR")
    state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": MODEL_GRAPHSAGE,
        "num_users": int(model.num_users),
        "num_movies": int(model.num_items),
        "seed": int(model.config.seed),
        "hyperparams": model.config.as_dict(),
        "state_dict": state,
        "train_pairs": _pairs_as_lists(pairs),
        "conv_stack": (
            "torch_geometric.nn.SAGEConv (mean) on native NeighborhoodBatch "
            "edge_index; not a PyG NeighborLoader"
        ),
        "sampler": (
            "graph_sampler.BipartiteCSR.sample_neighbors via "
            "sagerec_minibatch.NativeMinibatchSampler (ADR-005)"
        ),
        "tie_break": TIE_BREAK_POLICY,
        "note": _checkpoint_note(MODEL_GRAPHSAGE),
    }


def mf_checkpoint_payload(model: baseline.ImplicitMF) -> dict[str, Any]:
    """Build a schema-v1 implicit-MF checkpoint mapping (not yet written)."""
    if not isinstance(model, baseline.ImplicitMF):
        raise ServeError(
            "save_checkpoint expected sagerec_baseline.ImplicitMF, "
            f"got {type(model).__name__}"
        )
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": MODEL_IMPLICIT_MF,
        "num_users": int(model.num_users),
        "num_movies": int(model.num_items),
        "seed": int(model.config.seed),
        "hyperparams": model.config.as_dict(),
        "state_dict": {
            "user_factors": np.asarray(model.user_factors, dtype=np.float64),
            "item_factors": np.asarray(model.item_factors, dtype=np.float64),
            "user_bias": np.asarray(model.user_bias, dtype=np.float64),
            "item_bias": np.asarray(model.item_bias, dtype=np.float64),
        },
        "tie_break": TIE_BREAK_POLICY,
        "note": _checkpoint_note(MODEL_IMPLICIT_MF),
    }


def save_checkpoint(
    path: str | Path,
    model: PairScorer,
    *,
    train_pairs: Sequence[tuple[int, int]] | None = None,
) -> Path:
    """Write a schema-v1 SageRec checkpoint.

    GraphSAGE checkpoints store ``train_pairs`` so serve can rebuild the
    train-only native CSR. Implicit MF stores factors only.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, graphsage.GraphSAGERecommender):
        payload = graphsage_checkpoint_payload(model, train_pairs=train_pairs)
    elif isinstance(model, baseline.ImplicitMF):
        if train_pairs is not None:
            raise ServeError(
                "implicit-MF checkpoints do not store train_pairs; omit "
                "train_pairs when saving ImplicitMF"
            )
        payload = mf_checkpoint_payload(model)
    else:
        raise ServeError(
            "save_checkpoint supports GraphSAGERecommender or ImplicitMF "
            f"(PairScorer), got {type(model).__name__}"
        )
    graphsage.torch.save(payload, dest)
    return dest


def _load_raw_payload(path: Path) -> Any:
    _require_file(path, what="checkpoint")
    try:
        return graphsage.torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} is corrupt or not a SageRec "
            f"checkpoint: {exc}. Expected schema {CHECKPOINT_SCHEMA!r} version "
            f"{CHECKPOINT_SCHEMA_VERSION} written by sagerec_serve.save_checkpoint."
        ) from exc


def _require_payload_dict(payload: Any, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} is corrupt or not a SageRec "
            f"checkpoint: expected a dict payload, got {type(payload).__name__}. "
            f"Expected schema {CHECKPOINT_SCHEMA!r} version "
            f"{CHECKPOINT_SCHEMA_VERSION}."
        )
    return payload


def _require_schema(payload: Mapping[str, Any], path: Path) -> None:
    schema = payload.get("schema")
    version = payload.get("schema_version")
    if schema != CHECKPOINT_SCHEMA:
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} is corrupt or not a SageRec "
            f"checkpoint: schema is {schema!r}, expected {CHECKPOINT_SCHEMA!r}."
        )
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} has unsupported "
            f"schema_version {version!r}; this CLI reads version "
            f"{CHECKPOINT_SCHEMA_VERSION} only."
        )


def _require_counts(payload: Mapping[str, Any], path: Path) -> tuple[int, int]:
    try:
        num_users = _require_int(int(payload["num_users"]), "num_users")
        num_movies = _require_int(int(payload["num_movies"]), "num_movies")
    except (KeyError, TypeError, ValueError) as exc:
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} is missing valid "
            "num_users / num_movies integers"
        ) from exc
    if num_users < 1 or num_movies < 1:
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} has invalid graph size "
            f"num_users={num_users}, num_movies={num_movies}"
        )
    return num_users, num_movies


def _hyperparams(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    hp = payload.get("hyperparams")
    if not isinstance(hp, dict):
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} is missing a hyperparams object"
        )
    return dict(hp)


def _graphsage_config_from_payload(
    payload: Mapping[str, Any], path: Path
) -> graphsage.GraphSAGEConfig:
    hp = _hyperparams(payload, path)
    if "fanouts" in hp:
        fanouts = hp["fanouts"]
        if isinstance(fanouts, list):
            hp["fanouts"] = tuple(fanouts)
    if "seed" not in hp and "seed" in payload:
        hp["seed"] = payload["seed"]
    unknown = set(hp) - _GRAPHSAGE_HYPERPARAM_KEYS
    if unknown:
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} has unknown GraphSAGE "
            f"hyperparams {sorted(unknown)}"
        )
    missing = _GRAPHSAGE_HYPERPARAM_KEYS - set(hp)
    if missing:
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} is missing GraphSAGE "
            f"hyperparams {sorted(missing)}"
        )
    try:
        return graphsage.GraphSAGEConfig(**hp)
    except (TypeError, ValueError) as exc:
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} has invalid GraphSAGE "
            f"hyperparams: {exc}"
        ) from exc


def _mf_config_from_payload(
    payload: Mapping[str, Any], path: Path
) -> baseline.MFConfig:
    hp = _hyperparams(payload, path)
    if "seed" not in hp:
        hp["seed"] = payload.get("seed", 0)
    unknown = set(hp) - _MF_HYPERPARAM_KEYS
    if unknown:
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} has unknown MF hyperparams "
            f"{sorted(unknown)}"
        )
    try:
        return baseline.MFConfig(**hp)
    except (TypeError, ValueError) as exc:
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} has invalid MF hyperparams: {exc}"
        ) from exc


def _resolve_train_pairs(
    payload: Mapping[str, Any],
    path: Path,
    *,
    train_pairs: Sequence[tuple[int, int]] | None,
    train_pairs_path: str | Path | None,
    processed_dir: str | Path | None,
) -> list[tuple[int, int]]:
    if train_pairs is not None:
        return _coerce_train_pairs(list(train_pairs), origin="caller")
    if train_pairs_path is not None:
        return load_train_pairs_json(train_pairs_path)
    if processed_dir is not None:
        return load_train_pairs_from_processed(processed_dir)
    embedded = payload.get("train_pairs")
    if embedded is None:
        raise ServeError(
            f"GraphSAGE checkpoint at {_path_display(path)!r} has no train_pairs. "
            "Serve-time neighborhoods need the train-only CSR. Pass --train-pairs "
            "or --processed-dir, or save the checkpoint with train_pairs included. "
            "The demo CLI does not download MovieLens."
        )
    return _coerce_train_pairs(embedded, origin=_path_display(path))


def _load_graphsage(
    payload: Mapping[str, Any],
    path: Path,
    *,
    train_pairs: Sequence[tuple[int, int]] | None,
    train_pairs_path: str | Path | None,
    processed_dir: str | Path | None,
) -> graphsage.GraphSAGERecommender:
    num_users, num_movies = _require_counts(payload, path)
    config = _graphsage_config_from_payload(payload, path)
    pairs = _resolve_train_pairs(
        payload,
        path,
        train_pairs=train_pairs,
        train_pairs_path=train_pairs_path,
        processed_dir=processed_dir,
    )
    for index, (user_id, movie_id) in enumerate(pairs):
        if user_id < 0 or user_id >= num_users:
            raise ServeError(
                f"train_pairs[{index}].user_id {user_id} is outside "
                f"[0, {num_users}) from the checkpoint"
            )
        if movie_id < 0 or movie_id >= num_movies:
            raise ServeError(
                f"train_pairs[{index}].movie_id {movie_id} is outside "
                f"[0, {num_movies}) from the checkpoint"
            )
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} is missing a GraphSAGE "
            "state_dict"
        )
    sampler = minibatch.NativeMinibatchSampler.from_train_pairs(
        num_users, num_movies, pairs
    )
    model = graphsage.GraphSAGERecommender(sampler, config)
    try:
        model.load_state_dict(state)
    except Exception as exc:
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} state_dict does not match "
            f"the GraphSAGE architecture from hyperparams: {exc}"
        ) from exc
    model.eval()
    return model


def _load_mf(payload: Mapping[str, Any], path: Path) -> baseline.ImplicitMF:
    num_users, num_movies = _require_counts(payload, path)
    config = _mf_config_from_payload(payload, path)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} is missing an MF state_dict"
        )
    try:
        user_factors = np.asarray(state["user_factors"], dtype=np.float64)
        item_factors = np.asarray(state["item_factors"], dtype=np.float64)
        user_bias = np.asarray(state["user_bias"], dtype=np.float64)
        item_bias = np.asarray(state["item_bias"], dtype=np.float64)
    except KeyError as exc:
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} MF state_dict is missing "
            f"{exc.args[0]}"
        ) from exc
    if user_factors.shape != (num_users, config.n_factors):
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} user_factors shape "
            f"{user_factors.shape} does not match "
            f"(num_users={num_users}, n_factors={config.n_factors})"
        )
    if item_factors.shape != (num_movies, config.n_factors):
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} item_factors shape "
            f"{item_factors.shape} does not match "
            f"(num_movies={num_movies}, n_factors={config.n_factors})"
        )
    try:
        return baseline.ImplicitMF(
            user_factors,
            item_factors,
            user_bias,
            item_bias,
            config=config,
        )
    except ValueError as exc:
        raise ServeError(
            f"checkpoint at {_path_display(path)!r} has invalid MF factors: {exc}"
        ) from exc


def load_checkpoint(
    path: str | Path,
    *,
    train_pairs: Sequence[tuple[int, int]] | None = None,
    train_pairs_path: str | Path | None = None,
    processed_dir: str | Path | None = None,
) -> PairScorer:
    """Load a GraphSAGE or implicit-MF checkpoint as a ``PairScorer``."""
    file_path = Path(path)
    payload = _require_payload_dict(_load_raw_payload(file_path), file_path)
    _require_schema(payload, file_path)
    model_name = payload.get("model")
    if model_name == MODEL_GRAPHSAGE:
        return _load_graphsage(
            payload,
            file_path,
            train_pairs=train_pairs,
            train_pairs_path=train_pairs_path,
            processed_dir=processed_dir,
        )
    if model_name == MODEL_IMPLICIT_MF:
        return _load_mf(payload, file_path)
    raise ServeError(
        f"checkpoint at {_path_display(file_path)!r} has unsupported model "
        f"{model_name!r}; expected one of {list(SUPPORTED_MODELS)}"
    )


def rank_candidates(
    scorer: PairScorer,
    user_id: int,
    movie_ids: Sequence[int],
    *,
    k: int = DEFAULT_K,
) -> list[RankedCandidate]:
    """Score candidates and return top-K.

    Ranking is score descending; ties break on smaller ``movie_id``
    (``TIE_BREAK_POLICY``). Scores must be finite.
    """
    if not isinstance(scorer, PairScorer):
        raise ServeError(
            "scorer must implement PairScorer.score_pairs, "
            f"got {type(scorer).__name__}"
        )
    user_id = _require_int(user_id, "user_id")
    k = _require_int(k, "k")
    if k < 1:
        raise ServeError(f"k must be >= 1, got {k}")
    if not movie_ids:
        raise ServeError("candidates is empty; pass one or more movie IDs")
    items = [int(movie_id) for movie_id in movie_ids]
    if len(set(items)) != len(items):
        raise ServeError("candidates contain duplicate movie IDs")
    try:
        raw_scores = scorer.score_pairs([user_id] * len(items), items)
    except ValueError as exc:
        raise ServeError(str(exc)) from exc
    scores = [float(score) for score in raw_scores]
    if len(scores) != len(items):
        raise ServeError(
            "PairScorer.score_pairs must return one score per candidate, "
            f"got {len(scores)} scores for {len(items)} movies"
        )
    for movie_id, score in zip(items, scores):
        if not math.isfinite(score):
            raise ServeError(
                f"non-finite score {score!r} for user_id={user_id} "
                f"movie_id={movie_id}"
            )
    ordered = sorted(
        zip(items, scores),
        key=lambda row: (-row[1], row[0]),
    )
    top = ordered[:k]
    return [
        RankedCandidate(rank=rank, movie_id=movie_id, score=score)
        for rank, (movie_id, score) in enumerate(top, start=1)
    ]


def format_recommendations(rows: Sequence[RankedCandidate]) -> str:
    """TSV table: rank, movie_id, score."""
    lines = ["rank\tmovie_id\tscore"]
    for row in rows:
        lines.append(f"{row.rank}\t{row.movie_id}\t{row.score:.10g}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a SageRec GraphSAGE or implicit-MF checkpoint, score "
            "candidate movie IDs for one user, and print top-K. ADR-008 demo "
            "CLI: not an HTTP API and not a batch exporter. Does not download "
            "MovieLens. GraphSAGE serve-time neighborhoods need train-only "
            "pairs (embedded in the checkpoint, --train-pairs, or "
            "--processed-dir)."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Schema-v1 sagerec_checkpoint path written by save_checkpoint.",
    )
    parser.add_argument(
        "--user",
        required=True,
        type=int,
        help="Zero-based local user ID to score.",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Comma-separated zero-based movie IDs to score.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Top-K to print (default {DEFAULT_K}).",
    )
    parser.add_argument(
        "--train-pairs",
        type=Path,
        default=None,
        help=(
            "Optional JSON list of [user_id, movie_id] training positives. "
            "Used when a GraphSAGE checkpoint omits train_pairs."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help=(
            "Optional existing data/processed directory (reads train_pairs.json). "
            "Never downloads MovieLens."
        ),
    )
    return parser


def recommend_from_checkpoint(
    checkpoint: str | Path,
    user_id: int,
    candidates: Sequence[int],
    *,
    k: int = DEFAULT_K,
    train_pairs: Sequence[tuple[int, int]] | None = None,
    train_pairs_path: str | Path | None = None,
    processed_dir: str | Path | None = None,
) -> list[RankedCandidate]:
    """Load a checkpoint and return ranked top-K candidates."""
    scorer = load_checkpoint(
        checkpoint,
        train_pairs=train_pairs,
        train_pairs_path=train_pairs_path,
        processed_dir=processed_dir,
    )
    return rank_candidates(scorer, user_id, candidates, k=k)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """CLI entry. Returns 0 on success and a nonzero status on failure."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        return 0 if code is None else int(code)
    try:
        movie_ids = parse_candidate_ids(args.candidates)
        rows = recommend_from_checkpoint(
            args.checkpoint,
            args.user,
            movie_ids,
            k=args.k,
            train_pairs_path=args.train_pairs,
            processed_dir=args.processed_dir,
        )
    except ServeError as exc:
        print(f"error: {exc}", file=err)
        return 1
    out.write(format_recommendations(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
