"""MovieLens 100K retrieval versus retrieval-plus-re-rank evaluation (ADR-010).

Stage 1 is implicit MF trained on ADR-003 train positives (the same dot-product
retrieval ADR-009 serves). Stage 2 is the NumPy MLP in ``sagerec_rerank``,
fit on validation labels inside the stage-1 pool and scored on the test split.
Genre flags come from ``u.item``. This module does not start HTTP and does
not overwrite Phase 5 or Phase 7 result files.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sagerec_baseline as baseline
import sagerec_dataset as dataset
import sagerec_download as download
import sagerec_prep as prep
import sagerec_rerank as rerank
import sagerec_serving as serving

PROTOCOL_VERSION = "rerank-eval-v1"
ADR_ID = rerank.ADR_ID
METRIC_KEYS = ("recall_at_10", "ndcg_at_10", "map")

REQUIRED_RERANK_EVAL_KEYS = (
    "benchmark",
    "protocol_version",
    "adr",
    "dataset_edition",
    "source_url",
    "checksum",
    "license",
    "split_policy_id",
    "split_policy_version",
    "seed",
    "stage1",
    "stage2",
    "protocol",
    "n_users",
    "stage1_hit_rate_at_n",
    "retrieval_only",
    "retrieval_plus_rerank",
    "delta_retrieval_plus_rerank_minus_retrieval_only",
    "rerank_train",
    "eligibility",
    "cpu",
    "platform",
    "python",
    "numpy",
    "git_commit",
    "git_dirty",
    "generated_at",
    "note",
)


class RerankEvalError(RuntimeError):
    """The MovieLens re-rank evaluation could not be measured."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cpu_info() -> dict[str, Any]:
    model: str | None = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {
        "model": model or (platform.processor() or "unknown"),
        "logical_cpus": os.cpu_count(),
        "machine": platform.machine(),
    }


def git_commit(root: Path) -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def git_dirty(root: Path) -> bool:
    try:
        value = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(value.strip())


def _finite(value: float, field: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise RerankEvalError(f"{field} is not finite: {value!r}")
    return number


def _at_10(block: Mapping[str, float], *, k: int) -> dict[str, float]:
    if k != 10:
        raise RerankEvalError(
            f"stored rerank-eval-v1 requires k=10, got k={k}. "
            "Do not relabel another cutoff as Recall@10."
        )
    return {
        "recall_at_10": _finite(block["recall_at_k"], "recall_at_k"),
        "ndcg_at_10": _finite(block["ndcg_at_k"], "ndcg_at_k"),
        "map": _finite(block["map_at_k"], "map_at_k"),
    }


def resolve_uitem(raw_dir: Path, uitem_path: Path | None) -> tuple[str, str]:
    """Return ``(u.item text, origin)`` without downloading a new archive.

    Looks at ``--uitem``, then ``raw_dir/ml-100k/u.item``, then extracts
    ``u.item`` from an already-present ``raw_dir/ml-100k.zip``.
    """
    if uitem_path is not None:
        if not uitem_path.is_file():
            raise RerankEvalError(
                f"--uitem {uitem_path} does not exist. Pass the MovieLens "
                "100K u.item file. This script does not download it on its own."
            )
        return uitem_path.read_text(encoding="latin-1"), str(uitem_path)
    extracted = raw_dir / "ml-100k" / "u.item"
    if extracted.is_file():
        return extracted.read_text(encoding="latin-1"), str(extracted)
    archive_path = raw_dir / "ml-100k.zip"
    if not archive_path.is_file():
        raise RerankEvalError(
            "MovieLens u.item was not found. Pass --uitem, place u.item at "
            f"{extracted}, or keep the official zip at {archive_path}. "
            "Genre affinity needs those flags. Genres are not invented."
        )
    payload, member = download.extract_uitem(archive_path.read_bytes())
    extracted.parent.mkdir(parents=True, exist_ok=True)
    extracted.write_bytes(payload)
    return payload.decode("latin-1"), f"{archive_path}#{member}"


def _movie_source_ids(processed_dir: Path, num_items: int) -> list[int]:
    path = processed_dir / dataset.MAPPINGS_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RerankEvalError(
            f"could not read movie source ids from {path}: {exc}"
        ) from exc
    source_ids = payload.get("movie_source_ids")
    if not isinstance(source_ids, list) or len(source_ids) != num_items:
        raise RerankEvalError(
            f"{path} movie_source_ids must be a list of length {num_items}"
        )
    return [int(source_id) for source_id in source_ids]


def mf_score_matrix(model: baseline.ImplicitMF) -> np.ndarray:
    """Full user-item score matrix, matching ``ImplicitMF.score_pairs``."""
    scores = model.user_factors @ model.item_factors.T
    scores += model.user_bias[:, None]
    scores += model.item_bias[None, :]
    probe_items = [0, model.num_items // 2, model.num_items - 1]
    probe_users = [0]
    for user_id in probe_users:
        live = np.asarray(
            model.score_pairs([user_id] * len(probe_items), probe_items),
            dtype=np.float64,
        )
        if not np.allclose(scores[user_id, probe_items], live, rtol=1e-8, atol=1e-8):
            raise RerankEvalError(
                "stage-1 matrix scores do not match ImplicitMF.score_pairs"
            )
    return scores


def build_rerank_eval_payload(
    *,
    manifest: Mapping[str, Any],
    measured: Mapping[str, Any],
    mf_config: baseline.MFConfig,
    uitem_origin: str,
    root: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Machine-readable comparison. Does not invent metrics."""
    root = repo_root() if root is None else root
    retrieval = _at_10(measured["retrieval_only"], k=int(measured["k"]))
    reranked = _at_10(measured["retrieval_plus_rerank"], k=int(measured["k"]))
    delta_raw = measured["delta_retrieval_plus_rerank_minus_retrieval_only"]
    delta = {
        "recall_at_10": _finite(delta_raw["recall_at_k"], "delta.recall_at_k"),
        "ndcg_at_10": _finite(delta_raw["ndcg_at_k"], "delta.ndcg_at_k"),
        "map": _finite(delta_raw["map_at_k"], "delta.map_at_k"),
    }
    for key in METRIC_KEYS:
        if abs((reranked[key] - retrieval[key]) - delta[key]) > 1e-12:
            raise RerankEvalError(f"delta.{key} does not match the two metric blocks")
    dirty = git_dirty(root)
    note = (
        "Single-seed MovieLens 100K two-stage comparison (ADR-010). "
        "Stage 1 is implicit MF (ADR-004) dot-product retrieval, the same "
        "score ADR-009 serves, trained on ADR-003 train positives only. "
        "Stage 2 is a NumPy MLP with a linear skip, fit listwise on "
        "validation items that stage 1 retrieved, and scored on the test "
        "split. Both systems rank inside that stage-1 top-N pool. "
        "MAP is MAP@10. Features use train interactions only. Genres come "
        "from u.item. This is not a GraphSAGE result, not the Phase 5 or "
        "Phase 7 leaderboard, and not a SOTA claim. Re-ranking cannot "
        "recover a test item that stage 1 did not retrieve."
    )
    if dirty:
        note += (
            " git_dirty was true at generation, so git_commit is the HEAD "
            "the process saw and may not contain this results file."
        )
    counts = manifest["counts"]
    return {
        "benchmark": "sagerec_two_stage_rerank",
        "protocol_version": PROTOCOL_VERSION,
        "adr": ADR_ID,
        "dataset_edition": manifest["dataset_edition"],
        "source_url": manifest.get("source_url"),
        "checksum": manifest.get("checksum"),
        "license": manifest.get("license"),
        "split_policy_id": manifest["split_policy_id"],
        "split_policy_version": manifest["split_policy_version"],
        "seed": int(mf_config.seed),
        "stage1": {
            "model": "implicit_mf",
            "score_formula": serving.SCORE_FORMULA,
            "tie_break": serving.TIE_BREAK_POLICY,
            "hyperparams": mf_config.as_dict(),
            "stage1_n": int(measured["stage1_n"]),
            "candidate_filter": (
                "test eval excludes train and validation positives; "
                "the test positive stays in the pool"
            ),
        },
        "stage2": {
            "model": "numpy_mlp_listwise",
            "score_formula": rerank.SCORE_FORMULA,
            "tie_break": rerank.TIE_BREAK_POLICY,
            "hyperparams": dict(measured["config"]),
            "feature_names": list(measured["feature_names"]),
            "genres_available": bool(measured["genres_available"]),
            "genre_names": list(rerank.GENRE_NAMES),
            "uitem_origin": uitem_origin,
            "label_split": "validation",
            "eval_split": "test",
            "training_pool_excludes": ["train", "test"],
        },
        "protocol": {
            "k": 10,
            "stage1_n": int(measured["stage1_n"]),
            "metrics": list(METRIC_KEYS),
            "map": rerank.MAP_DEFINITION,
            "eligible_users": "ADR-003 users with at least 3 interactions",
            "same_pool": True,
        },
        "n_users": int(measured["n_users"]),
        "stage1_hit_rate_at_n": _finite(
            measured["stage1_hit_rate_at_n"], "stage1_hit_rate_at_n"
        ),
        "retrieval_only": retrieval,
        "retrieval_plus_rerank": reranked,
        "delta_retrieval_plus_rerank_minus_retrieval_only": delta,
        "rerank_train": {
            key: measured["rerank_train"][key]
            for key in (
                "n_eligible_users",
                "n_train_users",
                "validation_hit_rate_at_n",
                "loss_first_epoch",
                "loss_final_epoch",
                "n_epochs",
                "label_split",
                "excluded_from_training_pools",
            )
        },
        "eligibility": {
            "users": counts["users"],
            "movies": counts["movies"],
            "interactions": counts["interactions"],
            "eligible_users": counts["eligible_users"],
            "cold_start_users": counts["cold_start_users"],
            "evaluated_users": int(measured["n_users"]),
        },
        "cpu": cpu_info(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "git_commit": git_commit(root),
        "git_dirty": dirty,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "note": note,
    }


def render_rerank_eval_markdown(payload: Mapping[str, Any]) -> str:
    """Markdown table rendered from a rerank-eval payload (no extra rounding)."""
    missing = [key for key in REQUIRED_RERANK_EVAL_KEYS if key not in payload]
    if missing:
        raise RerankEvalError(f"rerank eval payload is missing keys {missing}")
    retrieval = payload["retrieval_only"]
    reranked = payload["retrieval_plus_rerank"]
    delta = payload["delta_retrieval_plus_rerank_minus_retrieval_only"]
    for block_name, block in (
        ("retrieval_only", retrieval),
        ("retrieval_plus_rerank", reranked),
        ("delta", delta),
    ):
        for key in METRIC_KEYS:
            if key not in block:
                raise RerankEvalError(f"{block_name} is missing {key}")

    def fmt(value: float) -> str:
        return f"{float(value):.6f}"

    stage1 = payload["stage1"]
    stage2 = payload["stage2"]
    cpu = payload["cpu"]
    lines = [
        "# SageRec two-stage re-rank evaluation",
        "",
        "MovieLens 100K, ADR-003 leave-one-out, test split. Stage 1 is implicit "
        "MF retrieval. Stage 2 re-ranks that top-N pool. Not a GraphSAGE result "
        "and not the Phase 5 or Phase 7 leaderboard.",
        "",
        f"- Protocol: `{payload['protocol_version']}` ({payload['adr']})",
        (
            f"- Seed: {payload['seed']}; stage-1 N: {stage1['stage1_n']}; "
            f"users: {payload['n_users']}; "
            f"stage-1 hit rate @N: {fmt(payload['stage1_hit_rate_at_n'])}"
        ),
        (
            f"- Features: {', '.join(stage2['feature_names'])}; "
            f"genres_available={stage2['genres_available']}"
        ),
        (
            f"- Python: {payload['python']}; NumPy: {payload['numpy']}; "
            f"CPU: {cpu['model']} ({cpu['logical_cpus']} logical)"
        ),
        f"- Commit: `{payload['git_commit']}` (dirty={payload['git_dirty']})",
        "",
        "| System | Recall@10 | NDCG@10 | MAP@10 |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| retrieval only | {fmt(retrieval['recall_at_10'])} | "
            f"{fmt(retrieval['ndcg_at_10'])} | {fmt(retrieval['map'])} |"
        ),
        (
            f"| retrieval + re-rank | {fmt(reranked['recall_at_10'])} | "
            f"{fmt(reranked['ndcg_at_10'])} | {fmt(reranked['map'])} |"
        ),
        (
            f"| delta (re-rank − retrieval) | {fmt(delta['recall_at_10'])} | "
            f"{fmt(delta['ndcg_at_10'])} | {fmt(delta['map'])} |"
        ),
        "",
        payload["note"],
        "",
    ]
    return "\n".join(lines)


def write_rerank_eval_results(
    payload: Mapping[str, Any],
    json_path: str | Path,
    md_path: str | Path,
) -> tuple[Path, Path]:
    missing = [key for key in REQUIRED_RERANK_EVAL_KEYS if key not in payload]
    if missing:
        raise RerankEvalError(f"refusing to write rerank eval missing {missing}")
    json_dest = Path(json_path)
    md_dest = Path(md_path)
    json_dest.parent.mkdir(parents=True, exist_ok=True)
    md_dest.parent.mkdir(parents=True, exist_ok=True)
    json_dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_dest.write_text(render_rerank_eval_markdown(payload), encoding="utf-8")
    return json_dest, md_dest


def run_movielens_rerank_eval(
    *,
    processed_dir: str | Path,
    raw_dir: str | Path,
    uitem_path: str | Path | None = None,
    seed: int = rerank.DEFAULT_SEED,
    stage1_n: int = rerank.DEFAULT_STAGE1_N,
    hidden_dim: int = rerank.DEFAULT_HIDDEN,
    n_epochs: int = rerank.DEFAULT_EPOCHS,
    learning_rate: float = rerank.DEFAULT_LEARNING_RATE,
    l2: float = rerank.DEFAULT_L2,
    json_out: str | Path | None = None,
    md_out: str | Path | None = None,
    rerank_out: str | Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Train MF, fit the re-ranker, and write the measured comparison."""
    root = repo_root() if root is None else Path(root)
    processed = Path(processed_dir)
    raw = Path(raw_dir)
    if json_out is None:
        json_out = root / "results" / "rerank_eval.json"
    if md_out is None:
        md_out = root / "results" / "rerank_eval.md"
    split, manifest = dataset.ensure_movielens_100k_processed(processed, raw)
    if manifest.get("dataset_edition") != prep.DATASET_EDITION_100K:
        raise RerankEvalError(
            f"expected dataset edition {prep.DATASET_EDITION_100K}, "
            f"got {manifest.get('dataset_edition')!r}"
        )
    if manifest.get("split_policy_id") != prep.SPLIT_POLICY_ID:
        raise RerankEvalError(
            f"expected split {prep.SPLIT_POLICY_ID}, got {manifest.get('split_policy_id')!r}"
        )
    uitem_text, uitem_origin = resolve_uitem(raw, Path(uitem_path) if uitem_path else None)
    genres = rerank.parse_uitem_genres(
        uitem_text,
        split.num_movies,
        _movie_source_ids(processed, split.num_movies),
    )
    mf_config = baseline.movielens_100k_config(seed=seed)
    model = baseline.train_implicit_mf(split, mf_config)
    score_matrix = mf_score_matrix(model)
    tables = rerank.build_feature_tables(split, genres)
    config = rerank.RerankConfig(
        stage1_n=stage1_n,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
        k=10,
    )

    def score_user(user_id: int) -> np.ndarray:
        return score_matrix[int(user_id)]

    measured = rerank.evaluate_two_stage(
        split,
        score_user,
        tables,
        config,
        stage1_model="implicit_mf",
    )
    payload = build_rerank_eval_payload(
        manifest=manifest,
        measured=measured,
        mf_config=mf_config,
        uitem_origin=uitem_origin,
        root=root,
    )
    write_rerank_eval_results(payload, json_out, md_out)
    if rerank_out is not None:
        rerank.save_rerank_artifact(rerank_out, measured["bundle"])
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MovieLens 100K retrieval-only vs retrieval plus re-rank "
            "and write results/rerank_eval.json. Uses ADR-003 prep and implicit "
            "MF as stage 1. Does not overwrite Phase 5 or Phase 7 files."
        )
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--uitem", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=rerank.DEFAULT_SEED)
    parser.add_argument("--stage1-n", type=int, default=rerank.DEFAULT_STAGE1_N)
    parser.add_argument("--hidden-dim", type=int, default=rerank.DEFAULT_HIDDEN)
    parser.add_argument("--epochs", type=int, default=rerank.DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=rerank.DEFAULT_LEARNING_RATE)
    parser.add_argument("--l2", type=float, default=rerank.DEFAULT_L2)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument(
        "--rerank-out",
        type=Path,
        default=None,
        help="Optional gitignored .npz re-rank artifact (for example results/serving/rerank.npz).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        return 0 if code is None else int(code)
    try:
        payload = run_movielens_rerank_eval(
            processed_dir=args.processed_dir,
            raw_dir=args.raw_dir,
            uitem_path=args.uitem,
            seed=args.seed,
            stage1_n=args.stage1_n,
            hidden_dim=args.hidden_dim,
            n_epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            json_out=args.json_out,
            md_out=args.md_out,
            rerank_out=args.rerank_out,
        )
    except (
        RerankEvalError,
        rerank.RerankError,
        ValueError,
        dataset.MovieLensPrepError,
        download.MovieLensDownloadError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    retrieval = payload["retrieval_only"]
    reranked = payload["retrieval_plus_rerank"]
    print(
        "retrieval_only recall@10={r_rec:.6f} ndcg@10={r_ndcg:.6f} map@10={r_map:.6f} "
        "rerank recall@10={k_rec:.6f} ndcg@10={k_ndcg:.6f} map@10={k_map:.6f} "
        "hit_at_n={hit:.6f}".format(
            r_rec=retrieval["recall_at_10"],
            r_ndcg=retrieval["ndcg_at_10"],
            r_map=retrieval["map"],
            k_rec=reranked["recall_at_10"],
            k_ndcg=reranked["ndcg_at_10"],
            k_map=reranked["map"],
            hit=payload["stage1_hit_rate_at_n"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
