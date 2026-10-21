"""Offline export of serving embeddings from a SageRec checkpoint (ADR-009).

GraphSAGE embeddings come from
``GraphSAGERecommender.eval_embedding_matrices``, which calls
``materialize_eval_embeddings`` (native ``graph_sampler`` neighborhoods via
``sagerec_minibatch``, one pass at ``config.seed``). Implicit MF exports the
factors and biases ``ImplicitMF.score_pairs`` already uses. The HTTP service
loads the artifact and does not sample or train.

This module does not download MovieLens and does not start an HTTP server.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

import numpy as np

import sagerec_baseline as baseline
import sagerec_dataset as dataset
import sagerec_graphsage as graphsage
import sagerec_serve as serve
import sagerec_serving as serving

MODEL_SYNTHETIC = "synthetic"


class ExportError(ValueError):
    """Actionable failure while exporting a serving artifact."""


def _movie_source_ids(processed_dir: Path) -> list[int] | None:
    mappings_path = processed_dir / dataset.MAPPINGS_NAME
    if not mappings_path.is_file():
        return None
    try:
        payload = json.loads(mappings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportError(
            f"mappings file {mappings_path} is not valid JSON: {exc}"
        ) from exc
    source_ids = payload.get("movie_source_ids")
    if not isinstance(source_ids, list) or not source_ids:
        raise ExportError(
            f"mappings file {mappings_path} is missing a non-empty "
            "movie_source_ids list"
        )
    ids: list[int] = []
    for index, value in enumerate(source_ids):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ExportError(
                f"movie_source_ids[{index}] in {mappings_path} must be an int"
            )
        ids.append(int(value))
    return ids


def _titles_from_json(path: Path, num_items: int) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportError(
            f"titles JSON at {path} is not valid JSON: {exc}. Expected a list "
            "of titles aligned to local movie ids, or an object mapping "
            "local id strings to titles."
        ) from exc
    if isinstance(payload, list):
        if len(payload) != num_items:
            raise ExportError(
                f"titles JSON at {path} has length {len(payload)}, expected "
                f"num_items={num_items}"
            )
        titles = payload
    elif isinstance(payload, dict):
        titles = []
        for item_id in range(num_items):
            if str(item_id) not in payload:
                raise ExportError(
                    f"titles JSON at {path} is missing local movie id {item_id}"
                )
            titles.append(payload[str(item_id)])
    else:
        raise ExportError(
            f"titles JSON at {path} must be a list or object, "
            f"got {type(payload).__name__}"
        )
    cleaned: list[str] = []
    for item_id, title in enumerate(titles):
        if not isinstance(title, str) or not title:
            raise ExportError(
                f"title for local movie id {item_id} must be a non-empty string"
            )
        cleaned.append(title)
    return cleaned


def _titles_from_uitem(
    path: Path,
    num_items: int,
    movie_source_ids: Sequence[int] | None,
) -> list[str]:
    """Read MovieLens ``u.item`` (pipe-separated, latin-1) into local titles.

    ``movie_source_ids[local_id]`` is the source movie id. When it is omitted,
    source ids must be exactly ``1 .. num_items`` (local id = source id - 1).
    """
    try:
        text = path.read_text(encoding="latin-1")
    except OSError as exc:
        raise ExportError(f"could not read titles file {path}: {exc}") from exc
    by_source: dict[int, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2:
            raise ExportError(
                f"titles file {path} line {line_number} is not pipe-separated "
                "u.item (movie id, title, ...)"
            )
        try:
            source_id = int(parts[0])
        except ValueError as exc:
            raise ExportError(
                f"titles file {path} line {line_number} has a non-integer movie id"
            ) from exc
        title = parts[1].strip()
        if not title:
            raise ExportError(
                f"titles file {path} line {line_number} has an empty title"
            )
        by_source[source_id] = title
    if movie_source_ids is None:
        expected = set(range(1, num_items + 1))
        if set(by_source) != expected:
            raise ExportError(
                f"titles file {path} source ids are not exactly 1..{num_items}. "
                "Pass --processed-dir so mappings.json movie_source_ids can "
                "align source ids to local movie ids."
            )
        source_ids = list(range(1, num_items + 1))
    else:
        if len(movie_source_ids) != num_items:
            raise ExportError(
                f"movie_source_ids length {len(movie_source_ids)} does not match "
                f"num_items={num_items}"
            )
        source_ids = [int(source_id) for source_id in movie_source_ids]
    missing = [source_id for source_id in source_ids if source_id not in by_source]
    if missing:
        preview = ", ".join(str(source_id) for source_id in missing[:8])
        raise ExportError(
            f"titles file {path} is missing titles for source movie ids {preview}"
        )
    return [by_source[source_id] for source_id in source_ids]


def load_item_titles(
    path: str | Path,
    num_items: int,
    movie_source_ids: Sequence[int] | None = None,
) -> list[str]:
    """Load local-id titles from JSON or a MovieLens ``u.item`` file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ExportError(
            f"titles path {file_path} does not exist or is not a file. "
            "Pass a JSON list/object or a MovieLens u.item file. "
            "Export does not download MovieLens."
        )
    if file_path.suffix.lower() == ".json":
        return _titles_from_json(file_path, num_items)
    return _titles_from_uitem(file_path, num_items, movie_source_ids)


def _resolve_mf_pairs(
    *,
    num_users: int,
    num_items: int,
    train_pairs: Sequence[tuple[int, int]] | None,
    train_pairs_path: str | Path | None,
    processed_dir: str | Path | None,
) -> list[tuple[int, int]]:
    if train_pairs is not None:
        raw: Any = list(train_pairs)
        origin = "caller"
    elif train_pairs_path is not None:
        raw = serve.load_train_pairs_json(train_pairs_path)
        origin = str(train_pairs_path)
    elif processed_dir is not None:
        raw = serve.load_train_pairs_from_processed(processed_dir)
        origin = str(processed_dir)
    else:
        raise ExportError(
            "implicit-MF checkpoints do not store train pairs. Pass "
            "--train-pairs or --processed-dir so the serving artifact can "
            "exclude already-seen items. Export does not download MovieLens."
        )
    try:
        pairs = serve._coerce_train_pairs(raw, origin=origin)
    except serve.ServeError as exc:
        raise ExportError(str(exc)) from exc
    for index, (user_id, movie_id) in enumerate(pairs):
        if user_id < 0 or user_id >= num_users:
            raise ExportError(
                f"train pair {index} user_id {user_id} is outside [0, {num_users})"
            )
        if movie_id < 0 or movie_id >= num_items:
            raise ExportError(
                f"train pair {index} movie_id {movie_id} is outside [0, {num_items})"
            )
    return pairs


def export_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    *,
    train_pairs: Sequence[tuple[int, int]] | None = None,
    train_pairs_path: str | Path | None = None,
    processed_dir: str | Path | None = None,
    titles_path: str | Path | None = None,
) -> Path:
    """Write a serving artifact from a schema-v1 SageRec checkpoint.

    GraphSAGE uses the checkpoint's train-only pairs (or the override passed
    through to ``load_checkpoint``) both to rebuild the native CSR and to
    mask already-seen items. Implicit MF requires ``train_pairs``,
    ``train_pairs_path``, or ``processed_dir`` for that mask. Titles default
    to ``movie_{local_id}`` when ``titles_path`` is omitted.
    """
    try:
        model = serve.load_checkpoint(
            checkpoint,
            train_pairs=train_pairs,
            train_pairs_path=train_pairs_path,
            processed_dir=processed_dir,
        )
    except serve.ServeError as exc:
        raise ExportError(str(exc)) from exc

    source_ids: list[int] | None = None
    if processed_dir is not None:
        source_ids = _movie_source_ids(Path(processed_dir))

    checkpoint_name = Path(checkpoint).name
    if isinstance(model, graphsage.GraphSAGERecommender):
        user_embeddings, item_embeddings = model.eval_embedding_matrices()
        try:
            pairs = serve.train_pairs_from_native_sampler(model.sampler)
        except serve.ServeError as exc:
            raise ExportError(str(exc)) from exc
        model_name = serve.MODEL_GRAPHSAGE
        seed = int(model.config.seed)
        hyperparams = model.config.as_dict()
        user_bias = None
        item_bias = None
        note = (
            "GraphSAGE user/item rows from eval_embedding_matrices "
            "(materialize_eval_embeddings at config.seed). Neighborhoods were "
            "sampled once at export time by graph_sampler via sagerec_minibatch, "
            "not a PyG NeighborLoader. Biases are zero so the serving dot "
            "matches the inner-product scorer. Not a MovieLens quality result."
        )
    elif isinstance(model, baseline.ImplicitMF):
        user_embeddings = np.array(model.user_factors, dtype=np.float64, copy=True)
        item_embeddings = np.array(model.item_factors, dtype=np.float64, copy=True)
        user_bias = np.array(model.user_bias, dtype=np.float64, copy=True)
        item_bias = np.array(model.item_bias, dtype=np.float64, copy=True)
        try:
            pairs = _resolve_mf_pairs(
                num_users=model.num_users,
                num_items=model.num_items,
                train_pairs=train_pairs,
                train_pairs_path=train_pairs_path,
                processed_dir=processed_dir,
            )
        except serve.ServeError as exc:
            raise ExportError(str(exc)) from exc
        model_name = serve.MODEL_IMPLICIT_MF
        seed = int(model.config.seed)
        hyperparams = model.config.as_dict()
        note = (
            "Implicit MF factors and biases copied from the checkpoint. "
            "Serving scores use the same dot-plus-bias formula as "
            "ImplicitMF.score_pairs. Seen items are the supplied train "
            "positives. Not a MovieLens quality result."
        )
    else:
        raise ExportError(
            "export supports GraphSAGERecommender or ImplicitMF checkpoints, "
            f"got {type(model).__name__}"
        )

    num_items = int(item_embeddings.shape[0])
    if titles_path is None:
        titles: Sequence[str] | None = None
    else:
        titles = load_item_titles(titles_path, num_items, source_ids)

    index = serving.make_serving_index(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        user_bias=user_bias,
        item_bias=item_bias,
        seen_pairs=pairs,
        item_titles=titles,
        model=model_name,
        extra_metadata={
            "seed": seed,
            "checkpoint_file": checkpoint_name,
            "hyperparams": hyperparams,
            "catalog_kind": "checkpoint",
            "note": note,
        },
    )
    try:
        return serving.save_serving_artifact(output, index)
    except serving.ServingError as exc:
        raise ExportError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export precomputed user and item embeddings from a SageRec "
            "checkpoint to a sagerec_serving_artifact .npz (ADR-009). "
            "GraphSAGE materializes native-sampled eval embeddings once. "
            "Implicit MF copies factors and biases. Does not download "
            "MovieLens and does not start HTTP."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Schema-v1 sagerec_checkpoint written by sagerec_serve.save_checkpoint.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination .npz serving artifact.",
    )
    parser.add_argument(
        "--train-pairs",
        type=Path,
        default=None,
        help=(
            "JSON list of [user_id, movie_id] training positives. Required for "
            "implicit MF. Optional override for GraphSAGE when the checkpoint "
            "omits train_pairs."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help=(
            "Existing data/processed directory (train_pairs.json, and "
            "mappings.json when aligning u.item titles). Never downloads."
        ),
    )
    parser.add_argument(
        "--titles",
        type=Path,
        default=None,
        help=(
            "Optional JSON titles or MovieLens u.item. Defaults to "
            "movie_{local_id}."
        ),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """CLI entry. Returns 0 on success and 1 on export failure."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        return 0 if code is None else int(code)
    try:
        dest = export_checkpoint(
            args.checkpoint,
            args.output,
            train_pairs_path=args.train_pairs,
            processed_dir=args.processed_dir,
            titles_path=args.titles,
        )
    except ExportError as exc:
        print(f"error: {exc}", file=err)
        return 1
    index = serving.load_serving_artifact(dest)
    out.write(
        f"wrote {dest} model={index.metadata['model']} "
        f"users={index.num_users} items={index.num_items} "
        f"dim={index.embedding_dim}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
