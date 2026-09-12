"""GNN-versus-MF comparison writer (Phase 5).

Reads stored ranking-result JSON files and writes a machine-readable
comparison, a markdown table, and an SVG bar chart. Does not invent
metrics: values must already exist in the MF and GraphSAGE result files.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_RESULT_KEYS = (
    "model",
    "dataset_edition",
    "split_policy_id",
    "split_policy_version",
    "min_interactions_for_eval",
    "cold_start_policy",
    "seed",
    "hyperparams",
    "metrics",
    "eligibility",
    "note",
)
REQUIRED_METRIC_KEYS = ("k", "split", "recall_at_k", "ndcg_at_k", "n_evaluated_users")
PROTOCOL_FIELDS = (
    "dataset_edition",
    "split_policy_id",
    "split_policy_version",
    "min_interactions_for_eval",
    "cold_start_policy",
    "seed",
)
MF_MODEL = "implicit_mf"
GRAPHSAGE_MODEL = "graphsage"


class ComparisonError(ValueError):
    """Stored ranking results are missing fields or are not comparable."""


def stored_result_ref(path: str | Path, repo_root: str | Path | None = None) -> str:
    """POSIX path for provenance: relative to ``repo_root`` when possible."""
    resolved = Path(path).resolve()
    if repo_root is not None:
        try:
            return resolved.relative_to(Path(repo_root).resolve()).as_posix()
        except ValueError:
            pass
    parts = resolved.parts
    if "results" in parts:
        start = parts.index("results")
        return "/".join(parts[start:])
    return Path(path).name


def load_ranking_result(path: str | Path) -> dict[str, Any]:
    """Load one ranking-result JSON and validate the shared schema."""
    result_path = Path(path)
    if not result_path.is_file():
        raise ComparisonError(
            f"ranking result not found at {result_path}. "
            "Train and evaluate first; do not invent metrics."
        )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ComparisonError(f"{result_path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComparisonError(f"{result_path} must contain a JSON object")
    missing = [key for key in REQUIRED_RESULT_KEYS if key not in payload]
    if missing:
        raise ComparisonError(f"{result_path} is missing required keys: {missing}")
    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise ComparisonError(f"{result_path} metrics must be an object")
    missing_metrics = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    if missing_metrics:
        raise ComparisonError(
            f"{result_path} metrics is missing required keys: {missing_metrics}"
        )
    eligibility = payload["eligibility"]
    if not isinstance(eligibility, dict) or "evaluated_users" not in eligibility:
        raise ComparisonError(
            f"{result_path} eligibility must include evaluated_users"
        )
    _require_metric_number(metrics["recall_at_k"], f"{result_path} metrics.recall_at_k")
    _require_metric_number(metrics["ndcg_at_k"], f"{result_path} metrics.ndcg_at_k")
    return payload


def _require_metric_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{field} must be a real number, got {type(value).__name__}")
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise ComparisonError(f"{field} must lie in [0, 1], got {number}")
    return number


def assert_comparable(
    mf: Mapping[str, Any],
    gnn: Mapping[str, Any],
    *,
    mf_path: str | Path,
    gnn_path: str | Path,
) -> None:
    """Fail unless MF and GraphSAGE share the same eval protocol."""
    if mf.get("model") != MF_MODEL:
        raise ComparisonError(
            f"{mf_path} model must be {MF_MODEL!r}, got {mf.get('model')!r}"
        )
    if gnn.get("model") != GRAPHSAGE_MODEL:
        raise ComparisonError(
            f"{gnn_path} model must be {GRAPHSAGE_MODEL!r}, got {gnn.get('model')!r}"
        )
    for field in PROTOCOL_FIELDS:
        if mf.get(field) != gnn.get(field):
            raise ComparisonError(
                f"{field} mismatch: MF ({mf_path}) has {mf.get(field)!r}, "
                f"GraphSAGE ({gnn_path}) has {gnn.get(field)!r}. "
                "Compare only with the same split, seed, and protocol."
            )
    for key in ("k", "split", "n_evaluated_users"):
        if mf["metrics"].get(key) != gnn["metrics"].get(key):
            raise ComparisonError(
                f"metrics.{key} mismatch: MF {mf['metrics'].get(key)!r} vs "
                f"GraphSAGE {gnn['metrics'].get(key)!r}"
            )
    mf_users = mf["eligibility"].get("evaluated_users")
    gnn_users = gnn["eligibility"].get("evaluated_users")
    if mf_users != gnn_users:
        raise ComparisonError(
            f"eligibility.evaluated_users mismatch: MF {mf_users!r} vs "
            f"GraphSAGE {gnn_users!r}"
        )
    if mf_users != mf["metrics"]["n_evaluated_users"]:
        raise ComparisonError(
            "MF eligibility.evaluated_users must equal metrics.n_evaluated_users"
        )
    if gnn_users != gnn["metrics"]["n_evaluated_users"]:
        raise ComparisonError(
            "GraphSAGE eligibility.evaluated_users must equal metrics.n_evaluated_users"
        )
    for side, payload in (("MF", mf), ("GraphSAGE", gnn)):
        eligible = payload["eligibility"].get("eligible_users")
        if eligible is not None and eligible != payload["metrics"]["n_evaluated_users"]:
            raise ComparisonError(
                f"{side} eligible_users {eligible!r} does not match "
                f"n_evaluated_users {payload['metrics']['n_evaluated_users']!r}"
            )


def build_comparison(
    mf: Mapping[str, Any],
    gnn: Mapping[str, Any],
    *,
    mf_path: str | Path,
    gnn_path: str | Path,
    chart_path: str | Path | None = None,
    table_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the machine-readable comparison object from stored results."""
    assert_comparable(mf, gnn, mf_path=mf_path, gnn_path=gnn_path)
    mf_ref = stored_result_ref(mf_path, repo_root)
    gnn_ref = stored_result_ref(gnn_path, repo_root)
    mf_epochs = mf["hyperparams"].get("n_epochs")
    gnn_epochs = gnn["hyperparams"].get("n_epochs")
    payload: dict[str, Any] = {
        "comparison": "gnn_vs_mf",
        "dataset_edition": mf["dataset_edition"],
        "split_policy_id": mf["split_policy_id"],
        "split_policy_version": mf["split_policy_version"],
        "min_interactions_for_eval": mf["min_interactions_for_eval"],
        "cold_start_policy": mf["cold_start_policy"],
        "split": mf["metrics"]["split"],
        "k": mf["metrics"]["k"],
        "seed": mf["seed"],
        "n_evaluated_users": mf["metrics"]["n_evaluated_users"],
        "protocol_note": (
            "Same ADR-003 split, eligible users, candidate protocol, and "
            "Recall@10 / NDCG@10. Single-seed comparison, not a multi-seed "
            "leaderboard. Fairness is the shared eval protocol, not identical "
            "wall-clock or epoch count "
            f"(MF n_epochs={mf_epochs}, GraphSAGE n_epochs={gnn_epochs})."
        ),
        "models": {
            MF_MODEL: {
                "result_path": mf_ref,
                "recall_at_k": mf["metrics"]["recall_at_k"],
                "ndcg_at_k": mf["metrics"]["ndcg_at_k"],
                "n_epochs": mf_epochs,
                "hyperparams": dict(mf["hyperparams"]),
            },
            GRAPHSAGE_MODEL: {
                "result_path": gnn_ref,
                "recall_at_k": gnn["metrics"]["recall_at_k"],
                "ndcg_at_k": gnn["metrics"]["ndcg_at_k"],
                "n_epochs": gnn_epochs,
                "hyperparams": dict(gnn["hyperparams"]),
                "sampler": gnn.get("sampler"),
            },
        },
        "sources": {
            MF_MODEL: mf_ref,
            GRAPHSAGE_MODEL: gnn_ref,
        },
        "note": (
            "Values are copied from the stored result files. "
            "Do not treat this as a published multi-seed leaderboard."
        ),
    }
    if chart_path is not None:
        payload["chart_path"] = stored_result_ref(chart_path, repo_root)
    if table_path is not None:
        payload["table_path"] = stored_result_ref(table_path, repo_root)
    return payload


def render_comparison_markdown(comparison: Mapping[str, Any]) -> str:
    """Human-readable table for the stored MF vs GraphSAGE metrics."""
    mf = comparison["models"][MF_MODEL]
    gnn = comparison["models"][GRAPHSAGE_MODEL]
    k = comparison["k"]
    return (
        "# MovieLens 100K GraphSAGE vs implicit MF\n"
        "\n"
        f"{comparison['protocol_note']}\n"
        "\n"
        f"| Model | Seed | Epochs | Recall@{k} | NDCG@{k} | "
        "Evaluated users | Result file |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
        f"| implicit MF | {comparison['seed']} | {mf['n_epochs']} | "
        f"{mf['recall_at_k']:.6f} | {mf['ndcg_at_k']:.6f} | "
        f"{comparison['n_evaluated_users']} | `{mf['result_path']}` |\n"
        f"| GraphSAGE | {comparison['seed']} | {gnn['n_epochs']} | "
        f"{gnn['recall_at_k']:.6f} | {gnn['ndcg_at_k']:.6f} | "
        f"{comparison['n_evaluated_users']} | `{gnn['result_path']}` |\n"
        "\n"
        "GraphSAGE neighborhoods come from `graph_sampler` via "
        "`sagerec_minibatch.NativeMinibatchSampler` (ADR-005 uniform without "
        "replacement). This comparison does not use a PyG NeighborLoader.\n"
        "\n"
        f"{comparison['note']}\n"
    )


def render_comparison_svg(comparison: Mapping[str, Any]) -> str:
    """Simple two-metric grouped bar chart as SVG (no extra dependencies)."""
    mf = comparison["models"][MF_MODEL]
    gnn = comparison["models"][GRAPHSAGE_MODEL]
    values = [
        float(mf["recall_at_k"]),
        float(gnn["recall_at_k"]),
        float(mf["ndcg_at_k"]),
        float(gnn["ndcg_at_k"]),
    ]
    ymax = max(values + [0.01]) * 1.35
    width, height = 720, 420
    left, right, top, bottom = 70, 30, 50, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    groups = [
        (f"Recall@{comparison['k']}", values[0], values[1]),
        (f"NDCG@{comparison['k']}", values[2], values[3]),
    ]
    group_w = plot_w / len(groups)
    bar_w = group_w * 0.28
    mf_color = "#4C78A8"
    gnn_color = "#F58518"

    def x_for(group_index: int, bar_index: int) -> float:
        center = left + group_w * (group_index + 0.5)
        offset = -bar_w * 0.65 if bar_index == 0 else bar_w * 0.65
        return center + offset - bar_w / 2

    def y_for(value: float) -> tuple[float, float]:
        bar_h = (value / ymax) * plot_h
        return top + plot_h - bar_h, bar_h

    bars: list[str] = []
    for group_index, (label, mf_value, gnn_value) in enumerate(groups):
        for bar_index, (value, color) in enumerate(
            ((mf_value, mf_color), (gnn_value, gnn_color))
        ):
            x = x_for(group_index, bar_index)
            y, bar_h = y_for(value)
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{bar_h:.1f}" fill="{color}"/>'
            )
            bars.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" '
                f'text-anchor="middle" font-size="12" font-family="sans-serif">'
                f"{value:.4f}</text>"
            )
        group_center = left + group_w * (group_index + 0.5)
        bars.append(
            f'<text x="{group_center:.1f}" y="{height - 28:.1f}" '
            f'text-anchor="middle" font-size="14" font-family="sans-serif">'
            f"{label}</text>"
        )

    axis = (
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="#333" stroke-width="1.5"/>'
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" '
        f'y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>'
    )
    legend = (
        f'<rect x="{width - 210}" y="14" width="14" height="14" fill="{mf_color}"/>'
        f'<text x="{width - 190}" y="26" font-size="13" font-family="sans-serif">'
        f"implicit MF</text>"
        f'<rect x="{width - 110}" y="14" width="14" height="14" fill="{gnn_color}"/>'
        f'<text x="{width - 90}" y="26" font-size="13" font-family="sans-serif">'
        f"GraphSAGE</text>"
    )
    title = (
        '<text x="360" y="36" text-anchor="middle" font-size="16" '
        'font-family="sans-serif">'
        f"MovieLens 100K single-seed {comparison['split']} ranking "
        f"(seed {comparison['seed']})</text>"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        f"<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n"
        f"{title}\n{axis}\n{''.join(bars)}\n{legend}\n"
        "</svg>\n"
    )


@dataclass(frozen=True)
class ComparisonArtifacts:
    """Paths written by ``write_comparison_artifacts``."""

    payload: dict[str, Any]
    json_path: Path
    markdown_path: Path
    chart_path: Path


def write_comparison_artifacts(
    mf_path: str | Path,
    gnn_path: str | Path,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    chart_path: str | Path,
    repo_root: str | Path | None = None,
) -> ComparisonArtifacts:
    """Load stored results and write JSON, markdown, and SVG artifacts."""
    mf = load_ranking_result(mf_path)
    gnn = load_ranking_result(gnn_path)
    json_out = Path(json_path)
    md_out = Path(markdown_path)
    chart_out = Path(chart_path)
    payload = build_comparison(
        mf,
        gnn,
        mf_path=mf_path,
        gnn_path=gnn_path,
        chart_path=chart_out,
        table_path=md_out,
        repo_root=repo_root,
    )
    markdown = render_comparison_markdown(payload)
    svg = render_comparison_svg(payload)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    chart_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(markdown, encoding="utf-8")
    chart_out.write_text(svg, encoding="utf-8")
    return ComparisonArtifacts(
        payload=payload,
        json_path=json_out,
        markdown_path=md_out,
        chart_path=chart_out,
    )
