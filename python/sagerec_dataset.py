"""On-disk MovieLens 100K preparation (Phase 2).

Reads ``u.data`` from a path or bytes, parses with
``graph_sampler.parse_movielens_100k``, splits with ADR-003
``sagerec_prep``, and writes gitignored artifacts plus a filled manifest.
Does not download and does not implement MovieLens 1M.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import graph_sampler
import sagerec_download as download
import sagerec_prep as prep

ASSIGNMENTS_NAME = "assignments.jsonl"
TRAIN_PAIRS_NAME = "train_pairs.json"
MAPPINGS_NAME = "mappings.json"
MANIFEST_NAME = "manifest.json"


class MovieLensPrepError(ValueError):
    """Actionable failure while reading or writing prepared 100K artifacts."""


@dataclass(frozen=True)
class PreparedDataset:
    """In-memory split plus the paths written under ``processed_dir``."""

    split: prep.SplitResult
    manifest: prep.DatasetManifest
    processed_dir: Path
    manifest_path: Path
    assignments_path: Path
    train_pairs_path: Path
    mappings_path: Path
    source_sha256: str


def read_udata_bytes(source: str | Path | bytes) -> bytes:
    """Load raw ``u.data`` bytes from a filesystem path or in-memory bytes."""
    if isinstance(source, bytes):
        if not source.strip():
            raise MovieLensPrepError(
                "u.data bytes are empty; expected tab-separated MovieLens 100K rows"
            )
        return source
    path = Path(source)
    if not path.is_file():
        raise MovieLensPrepError(
            f"u.data path {str(path)!r} does not exist or is not a file. "
            "Download the official 100K archive first "
            f"({download.MOVIELENS_100K_URL}) or pass the ratings bytes."
        )
    data = path.read_bytes()
    if not data.strip():
        raise MovieLensPrepError(
            f"u.data at {path} is empty; expected tab-separated MovieLens 100K rows"
        )
    return data


def _decode_udata(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _assert_train_only_pairs(split: prep.SplitResult) -> list[tuple[int, int]]:
    train_pairs = split.train_positive_pairs()
    held_out = split.positive_pair_set("validation") | split.positive_pair_set("test")
    overlap = set(train_pairs) & held_out
    if overlap:
        user_id, movie_id = next(iter(overlap))
        raise MovieLensPrepError(
            "held-out positive leaked into train pairs: "
            f"(user_id={user_id}, movie_id={movie_id})"
        )
    return train_pairs


def parse_and_split_udata(source: str | Path | bytes) -> tuple[prep.SplitResult, Any, str]:
    """Parse ``u.data`` and apply ADR-003. Returns split, parser result, sha256."""
    raw = read_udata_bytes(source)
    digest = download.sha256_hex(raw)
    text = _decode_udata(raw)
    try:
        parsed = graph_sampler.parse_movielens_100k(text)
    except graph_sampler.GraphError as exc:
        raise MovieLensPrepError(
            f"graph_sampler.parse_movielens_100k rejected the ratings text: {exc}"
        ) from exc
    split = prep.split_interactions(
        parsed.interactions,
        num_users=parsed.num_users,
        num_movies=parsed.num_movies,
        dataset_edition=prep.DATASET_EDITION_100K,
    )
    _assert_train_only_pairs(split)
    return split, parsed, digest


def write_processed_artifacts(
    split: prep.SplitResult,
    parsed: Any,
    processed_dir: str | Path,
    *,
    source_url: str | None = None,
    checksum: str | None = None,
    license: str | None = None,
) -> PreparedDataset:
    """Write split artifacts and a schema-shaped manifest under ``processed_dir``."""
    dest = Path(processed_dir)
    dest.mkdir(parents=True, exist_ok=True)
    train_pairs = _assert_train_only_pairs(split)
    manifest = prep.build_manifest(
        split,
        source_url=source_url,
        checksum=checksum,
        license=license,
    )
    assignments_path = dest / ASSIGNMENTS_NAME
    train_pairs_path = dest / TRAIN_PAIRS_NAME
    mappings_path = dest / MAPPINGS_NAME
    manifest_path = dest / MANIFEST_NAME

    with assignments_path.open("w", encoding="utf-8") as handle:
        for row in split.assignments:
            handle.write(
                json.dumps(
                    {
                        "user_id": row.user_id,
                        "movie_id": row.movie_id,
                        "rating": row.rating,
                        "timestamp": row.timestamp,
                        "split": row.split,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

    train_pairs_path.write_text(
        json.dumps(train_pairs, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    mappings_path.write_text(
        json.dumps(
            {
                "user_source_ids": list(parsed.user_source_ids),
                "movie_source_ids": list(parsed.movie_source_ids),
                "eligible_user_ids": list(split.eligible_user_ids),
                "cold_start_user_ids": list(split.cold_start_user_ids),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    written_pairs = [tuple(pair) for pair in json.loads(train_pairs_path.read_text())]
    held_out = split.positive_pair_set("validation") | split.positive_pair_set("test")
    if set(written_pairs) & held_out:
        raise MovieLensPrepError(
            "refusing to keep processed artifacts: written train_pairs overlap held-out positives"
        )

    return PreparedDataset(
        split=split,
        manifest=manifest,
        processed_dir=dest,
        manifest_path=manifest_path,
        assignments_path=assignments_path,
        train_pairs_path=train_pairs_path,
        mappings_path=mappings_path,
        source_sha256=checksum.split(":", 1)[-1] if checksum else "",
    )


def prepare_movielens_100k(
    source: str | Path | bytes,
    processed_dir: str | Path,
    *,
    source_url: str | None = None,
    checksum: str | None = None,
    license: str | None = None,
) -> PreparedDataset:
    """Ingest ``u.data``, apply ADR-003, and write ``processed/`` artifacts."""
    split, parsed, digest = parse_and_split_udata(source)
    filled_checksum = checksum if checksum is not None else download.format_checksum(
        "sha256", digest
    )
    return write_processed_artifacts(
        split,
        parsed,
        processed_dir,
        source_url=source_url,
        checksum=filled_checksum,
        license=license,
    )


def load_manifest(processed_dir: str | Path) -> prep.DatasetManifest:
    path = Path(processed_dir) / MANIFEST_NAME
    if not path.is_file():
        raise MovieLensPrepError(
            f"processed manifest not found at {path}. "
            "Run prepare_movielens_100k first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_edition") != prep.DATASET_EDITION_100K:
        raise MovieLensPrepError(
            f"{path} dataset_edition must be {prep.DATASET_EDITION_100K!r}, "
            f"got {payload.get('dataset_edition')!r}"
        )
    return payload


def load_split_from_processed(processed_dir: str | Path) -> prep.SplitResult:
    """Rebuild an ADR-003 ``SplitResult`` from written assignments + mappings."""
    dest = Path(processed_dir)
    assignments_path = dest / ASSIGNMENTS_NAME
    mappings_path = dest / MAPPINGS_NAME
    if not assignments_path.is_file() or not mappings_path.is_file():
        raise MovieLensPrepError(
            f"processed split artifacts missing under {dest} "
            f"(need {ASSIGNMENTS_NAME} and {MAPPINGS_NAME})"
        )
    mappings = json.loads(mappings_path.read_text(encoding="utf-8"))
    assignments: list[prep.SplitAssignment] = []
    with assignments_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                assignments.append(
                    prep.SplitAssignment(
                        user_id=int(row["user_id"]),
                        movie_id=int(row["movie_id"]),
                        timestamp=int(row["timestamp"]),
                        rating=int(row["rating"]),
                        split=row["split"],
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise MovieLensPrepError(
                    f"{assignments_path}:{line_no} is not a valid assignment row"
                ) from exc
    if not assignments:
        raise MovieLensPrepError(f"{assignments_path} contains no assignment rows")
    return prep.SplitResult(
        assignments=tuple(assignments),
        eligible_user_ids=tuple(int(x) for x in mappings["eligible_user_ids"]),
        cold_start_user_ids=tuple(int(x) for x in mappings["cold_start_user_ids"]),
        num_users=len(mappings["user_source_ids"]),
        num_movies=len(mappings["movie_source_ids"]),
        dataset_edition=prep.DATASET_EDITION_100K,
    )
