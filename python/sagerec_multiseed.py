"""Multi-seed MF vs GraphSAGE aggregation (ADR-007).

Reads per-seed ranking-result JSON (same schema as Phase 5) and writes a
leaderboard with per-seed rows plus mean / sample-std / median. Does not
invent metrics. Does not overwrite historical single-seed Phase 5 files.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sagerec_compare as compare

CONTINUITY_SEED = 7
DEFAULT_SEEDS = (7, 11, 13, 17, 19)
PROTOCOL_VERSION = "adr-007-v1"
COMPARISON_ID = "multiseed_gnn_vs_mf"
STD_DEFINITION = "sample standard deviation (n-1)"
PROTOCOL_FIELDS_ACROSS_SEEDS = (
    "dataset_edition",
    "split_policy_id",
    "split_policy_version",
    "min_interactions_for_eval",
    "cold_start_policy",
)
HISTORICAL_SINGLE_SEED = {
    "implicit_mf": "results/mf_movielens_100k.json",
    "graphsage": "results/graphsage_movielens_100k.json",
    "comparison": "results/gnn_vs_mf_movielens_100k.json",
    "note": (
        "Phase 5 single-seed provenance (seed 7). GraphSAGE there used "
        "in-repo mean layers; this multi-seed table is a separate SAGEConv "
        "measurement and does not retcon those files."
    ),
}


class MultiseedError(ValueError):
    """Seed list or per-seed ranking results are not aggregatable."""


def normalize_seeds(seeds: Sequence[int] | None = None) -> tuple[int, ...]:
    """Validate a modest seed list that includes the Phase 5 continuity seed."""
    if seeds is None:
        values = list(DEFAULT_SEEDS)
    else:
        values = []
        for index, seed in enumerate(seeds):
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise MultiseedError(
                    f"seeds[{index}] must be an int, got {type(seed).__name__}"
                )
            if seed < 0:
                raise MultiseedError(f"seeds[{index}] must be >= 0, got {seed}")
            values.append(seed)
    if len(values) < 2:
        raise MultiseedError(
            f"need at least two seeds to report {STD_DEFINITION}, got {len(values)}"
        )
    if len(set(values)) != len(values):
        raise MultiseedError(f"seed list must be unique, got {values}")
    if CONTINUITY_SEED not in values:
        raise MultiseedError(
            f"seed list must include continuity seed {CONTINUITY_SEED} "
            f"to match Phase 5 single-seed runs, got {values}"
        )
    return tuple(values)


def metric_moments(values: Sequence[float]) -> dict[str, Any]:
    """Mean, sample std, and median for one metric across seeds."""
    numbers: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MultiseedError(
                f"metric value[{index}] must be a real number, "
                f"got {type(value).__name__}"
            )
        number = float(value)
        if number < 0.0 or number > 1.0:
            raise MultiseedError(
                f"metric value[{index}] must lie in [0, 1], got {number}"
            )
        numbers.append(number)
    if len(numbers) < 2:
        raise MultiseedError(
            f"need at least two values to report {STD_DEFINITION}, got {len(numbers)}"
        )
    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers),
        "median": float(statistics.median(numbers)),
        "min": min(numbers),
        "max": max(numbers),
        "values": numbers,
    }


def _hyperparams_identity(hyperparams: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(hyperparams, Mapping):
        return {}
    return {key: value for key, value in hyperparams.items() if key != "seed"}


def _require_mapping(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise MultiseedError(f"{field} must be an object")
    return value


def assert_shared_protocol_across_seeds(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    seeds: Sequence[int],
) -> None:
    """Fail unless every seed pair shares the ADR-003 eval protocol."""
    if len(pairs) != len(seeds):
        raise MultiseedError(
            f"expected {len(seeds)} seed pairs, got {len(pairs)}"
        )
    if not pairs:
        raise MultiseedError("no per-seed result pairs to aggregate")
    first_mf, first_gnn = pairs[0]
    for index, (mf, gnn) in enumerate(pairs):
        seed = seeds[index]
        try:
            compare.assert_comparable(
                mf,
                gnn,
                mf_path=f"mf seed {seed}",
                gnn_path=f"graphsage seed {seed}",
            )
        except compare.ComparisonError as exc:
            raise MultiseedError(str(exc)) from exc
        if mf.get("seed") != seed or gnn.get("seed") != seed:
            raise MultiseedError(
                f"seed mismatch at index {index}: expected {seed}, "
                f"MF has {mf.get('seed')!r}, GraphSAGE has {gnn.get('seed')!r}"
            )
        for field in PROTOCOL_FIELDS_ACROSS_SEEDS:
            if mf.get(field) != first_mf.get(field):
                raise MultiseedError(
                    f"{field} mismatch across seeds: seed {seeds[0]} has "
                    f"{first_mf.get(field)!r}, seed {seed} has {mf.get(field)!r}"
                )
            if gnn.get(field) != first_gnn.get(field):
                raise MultiseedError(
                    f"{field} mismatch across seeds: seed {seeds[0]} has "
                    f"{first_gnn.get(field)!r}, seed {seed} has {gnn.get(field)!r}"
                )
        for key in ("k", "split", "n_evaluated_users"):
            if mf["metrics"].get(key) != first_mf["metrics"].get(key):
                raise MultiseedError(
                    f"metrics.{key} mismatch across seeds: "
                    f"{first_mf['metrics'].get(key)!r} vs {mf['metrics'].get(key)!r}"
                )
        mf_hp = _hyperparams_identity(_require_mapping(mf, "hyperparams"))
        gnn_hp = _hyperparams_identity(_require_mapping(gnn, "hyperparams"))
        if mf_hp != _hyperparams_identity(_require_mapping(first_mf, "hyperparams")):
            raise MultiseedError(
                "implicit MF hyperparameters (except seed) must match across seeds"
            )
        if gnn_hp != _hyperparams_identity(_require_mapping(first_gnn, "hyperparams")):
            raise MultiseedError(
                "GraphSAGE hyperparameters (except seed) must match across seeds"
            )


def _model_aggregate(
    payloads: Sequence[Mapping[str, Any]],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    first = payloads[0]
    block: dict[str, Any] = {
        "n_epochs": first["hyperparams"].get("n_epochs"),
        "hyperparams": _hyperparams_identity(first["hyperparams"]),
        "recall_at_k": metric_moments(
            [payload["metrics"]["recall_at_k"] for payload in payloads]
        ),
        "ndcg_at_k": metric_moments(
            [payload["metrics"]["ndcg_at_k"] for payload in payloads]
        ),
    }
    if extra:
        block.update(dict(extra))
    if "optimizer" in first["hyperparams"]:
        block["optimizer"] = first["hyperparams"]["optimizer"]
    return block


def build_multiseed_comparison(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    seeds: Sequence[int] | None = None,
    chart_path: str | Path | None = None,
    table_path: str | Path | None = None,
    git_commit: str | None = None,
    generated_at: str | None = None,
    python_version: str | None = None,
    numpy_version: str | None = None,
    torch_version: str | None = None,
    torch_geometric_version: str | None = None,
    platform_info: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the machine-readable multi-seed leaderboard from stored results."""
    seed_list = normalize_seeds(
        seeds if seeds is not None else [pair[0]["seed"] for pair in pairs]
    )
    assert_shared_protocol_across_seeds(pairs, seeds=seed_list)
    mf_payloads = [pair[0] for pair in pairs]
    gnn_payloads = [pair[1] for pair in pairs]
    first_mf = mf_payloads[0]
    first_gnn = gnn_payloads[0]
    mf_epochs = first_mf["hyperparams"].get("n_epochs")
    gnn_epochs = first_gnn["hyperparams"].get("n_epochs")
    gnn_optimizer = first_gnn["hyperparams"].get("optimizer", "adam")
    per_seed = []
    for seed, mf, gnn in zip(seed_list, mf_payloads, gnn_payloads):
        per_seed.append(
            {
                "seed": seed,
                compare.MF_MODEL: {
                    "recall_at_k": mf["metrics"]["recall_at_k"],
                    "ndcg_at_k": mf["metrics"]["ndcg_at_k"],
                    "n_epochs": mf["hyperparams"].get("n_epochs"),
                },
                compare.GRAPHSAGE_MODEL: {
                    "recall_at_k": gnn["metrics"]["recall_at_k"],
                    "ndcg_at_k": gnn["metrics"]["ndcg_at_k"],
                    "n_epochs": gnn["hyperparams"].get("n_epochs"),
                    "sampler": gnn.get("sampler"),
                    "conv_stack": gnn.get("conv_stack"),
                },
            }
        )
    payload: dict[str, Any] = {
        "comparison": COMPARISON_ID,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_edition": first_mf["dataset_edition"],
        "split_policy_id": first_mf["split_policy_id"],
        "split_policy_version": first_mf["split_policy_version"],
        "min_interactions_for_eval": first_mf["min_interactions_for_eval"],
        "cold_start_policy": first_mf["cold_start_policy"],
        "split": first_mf["metrics"]["split"],
        "k": first_mf["metrics"]["k"],
        "seeds": list(seed_list),
        "n_seeds": len(seed_list),
        "continuity_seed": CONTINUITY_SEED,
        "n_evaluated_users": first_mf["metrics"]["n_evaluated_users"],
        "uncertainty": {
            "headline": "mean_pm_sample_std",
            "std_definition": STD_DEFINITION,
            "also_reports": ["median", "min", "max"],
        },
        "protocol_note": (
            "Same ADR-003 split, eligible users, candidate protocol, and "
            f"Recall@{first_mf['metrics']['k']} / NDCG@{first_mf['metrics']['k']} "
            f"across seeds {list(seed_list)}. Fairness is the shared eval "
            "protocol, not identical wall-clock or optimizer budget "
            f"(MF n_epochs={mf_epochs} SGD, GraphSAGE n_epochs={gnn_epochs} "
            f"{gnn_optimizer}). GraphSAGE uses PyG SAGEConv on native "
            "graph_sampler neighborhoods (ADR-006); not a NeighborLoader run."
        ),
        "historical_single_seed": dict(HISTORICAL_SINGLE_SEED),
        "models": {
            compare.MF_MODEL: _model_aggregate(
                mf_payloads, extra={"optimizer": "sgd"}
            ),
            compare.GRAPHSAGE_MODEL: _model_aggregate(
                gnn_payloads,
                extra={
                    "sampler": first_gnn.get("sampler"),
                    "conv_stack": first_gnn.get("conv_stack"),
                    "optimizer": gnn_optimizer,
                },
            ),
        },
        "per_seed": per_seed,
        "note": (
            "Values are copied from measured per-seed ranking runs in this "
            "environment. Historical Phase 5 files remain single-seed "
            "provenance and were not overwritten. This is not a SOTA claim."
        ),
    }
    provenance: dict[str, Any] = {}
    if git_commit is not None:
        provenance["git_commit"] = git_commit
    if generated_at is not None:
        provenance["generated_at"] = generated_at
    if python_version is not None:
        provenance["python"] = python_version
    if numpy_version is not None:
        provenance["numpy"] = numpy_version
    if torch_version is not None:
        provenance["torch"] = torch_version
    if torch_geometric_version is not None:
        provenance["torch_geometric"] = torch_geometric_version
    if platform_info is not None:
        provenance["platform"] = dict(platform_info)
    if provenance:
        payload["provenance"] = provenance
    if chart_path is not None:
        payload["chart_path"] = compare.stored_result_ref(chart_path)
    if table_path is not None:
        payload["table_path"] = compare.stored_result_ref(table_path)
    return payload


def _fmt_mean_std(moments: Mapping[str, Any]) -> str:
    return f"{moments['mean']:.6f} ± {moments['std']:.6f}"


def render_multiseed_markdown(comparison: Mapping[str, Any]) -> str:
    """Human-readable per-seed table plus mean±std aggregates."""
    mf = comparison["models"][compare.MF_MODEL]
    gnn = comparison["models"][compare.GRAPHSAGE_MODEL]
    k = comparison["k"]
    seed_rows = []
    for row in comparison["per_seed"]:
        seed_rows.append(
            f"| {row['seed']} | implicit MF | {row[compare.MF_MODEL]['n_epochs']} | "
            f"{row[compare.MF_MODEL]['recall_at_k']:.6f} | "
            f"{row[compare.MF_MODEL]['ndcg_at_k']:.6f} |"
        )
        seed_rows.append(
            f"| {row['seed']} | GraphSAGE | {row[compare.GRAPHSAGE_MODEL]['n_epochs']} | "
            f"{row[compare.GRAPHSAGE_MODEL]['recall_at_k']:.6f} | "
            f"{row[compare.GRAPHSAGE_MODEL]['ndcg_at_k']:.6f} |"
        )
    seeds = ", ".join(str(seed) for seed in comparison["seeds"])
    return (
        "# MovieLens 100K multi-seed GraphSAGE vs implicit MF\n"
        "\n"
        f"{comparison['protocol_note']}\n"
        "\n"
        f"Uncertainty is **mean ± {STD_DEFINITION}** over seeds {seeds} "
        f"(n={comparison['n_seeds']}). Median is also stored. Continuity seed "
        f"{comparison['continuity_seed']} is included so Phase 5 single-seed "
        "runs stay comparable as historical provenance, not mixed into these "
        "aggregates.\n"
        "\n"
        f"| Model | Epochs | Recall@{k} mean±std | Recall@{k} median | "
        f"NDCG@{k} mean±std | NDCG@{k} median | Evaluated users |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| implicit MF | {mf['n_epochs']} | "
        f"{_fmt_mean_std(mf['recall_at_k'])} | {mf['recall_at_k']['median']:.6f} | "
        f"{_fmt_mean_std(mf['ndcg_at_k'])} | {mf['ndcg_at_k']['median']:.6f} | "
        f"{comparison['n_evaluated_users']} |\n"
        f"| GraphSAGE | {gnn['n_epochs']} | "
        f"{_fmt_mean_std(gnn['recall_at_k'])} | {gnn['recall_at_k']['median']:.6f} | "
        f"{_fmt_mean_std(gnn['ndcg_at_k'])} | {gnn['ndcg_at_k']['median']:.6f} | "
        f"{comparison['n_evaluated_users']} |\n"
        "\n"
        "## Per-seed rows\n"
        "\n"
        f"| Seed | Model | Epochs | Recall@{k} | NDCG@{k} |\n"
        "| ---: | --- | ---: | ---: | ---: |\n"
        + "\n".join(seed_rows)
        + "\n\n"
        "GraphSAGE neighborhoods come from `graph_sampler` via "
        "`sagerec_minibatch.NativeMinibatchSampler` (ADR-005 uniform without "
        "replacement). Message passing is PyG `SAGEConv` (ADR-006). This "
        "comparison does not use a PyG NeighborLoader.\n"
        "\n"
        "Historical single-seed files "
        f"(`{HISTORICAL_SINGLE_SEED['implicit_mf']}`, "
        f"`{HISTORICAL_SINGLE_SEED['graphsage']}`, "
        f"`{HISTORICAL_SINGLE_SEED['comparison']}`) stay Phase 5 provenance "
        "and were not overwritten.\n"
        "\n"
        f"{comparison['note']}\n"
    )


def render_multiseed_svg(comparison: Mapping[str, Any]) -> str:
    """Grouped mean bars with ±1 sample-std error bars (no extra deps)."""
    mf = comparison["models"][compare.MF_MODEL]
    gnn = comparison["models"][compare.GRAPHSAGE_MODEL]
    groups = [
        (
            f"Recall@{comparison['k']}",
            float(mf["recall_at_k"]["mean"]),
            float(mf["recall_at_k"]["std"]),
            float(gnn["recall_at_k"]["mean"]),
            float(gnn["recall_at_k"]["std"]),
        ),
        (
            f"NDCG@{comparison['k']}",
            float(mf["ndcg_at_k"]["mean"]),
            float(mf["ndcg_at_k"]["std"]),
            float(gnn["ndcg_at_k"]["mean"]),
            float(gnn["ndcg_at_k"]["std"]),
        ),
    ]
    peaks = []
    for _label, mf_mean, mf_std, gnn_mean, gnn_std in groups:
        peaks.append(mf_mean + mf_std)
        peaks.append(gnn_mean + gnn_std)
    ymax = max(peaks + [0.01]) * 1.35
    width, height = 760, 460
    left, right, top, bottom = 70, 30, 72, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / len(groups)
    bar_w = group_w * 0.28
    mf_color = "#4C78A8"
    gnn_color = "#F58518"
    seeds = ", ".join(str(seed) for seed in comparison["seeds"])

    def x_for(group_index: int, bar_index: int) -> float:
        center = left + group_w * (group_index + 0.5)
        offset = -bar_w * 0.65 if bar_index == 0 else bar_w * 0.65
        return center + offset - bar_w / 2

    def y_for(value: float) -> float:
        clamped = max(0.0, value)
        bar_h = (clamped / ymax) * plot_h
        return top + plot_h - bar_h

    parts: list[str] = []
    for group_index, (label, mf_mean, mf_std, gnn_mean, gnn_std) in enumerate(groups):
        series = (
            (mf_mean, mf_std, mf_color),
            (gnn_mean, gnn_std, gnn_color),
        )
        for bar_index, (mean, std, color) in enumerate(series):
            x = x_for(group_index, bar_index)
            y = y_for(mean)
            bar_h = top + plot_h - y
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{bar_h:.1f}" fill="{color}"/>'
            )
            cx = x + bar_w / 2
            y_hi = y_for(mean + std)
            y_lo = y_for(max(0.0, mean - std))
            cap = 6.0
            parts.append(
                f'<line x1="{cx:.1f}" y1="{y_hi:.1f}" x2="{cx:.1f}" '
                f'y2="{y_lo:.1f}" stroke="#333" stroke-width="1.5"/>'
            )
            parts.append(
                f'<line x1="{cx - cap:.1f}" y1="{y_hi:.1f}" x2="{cx + cap:.1f}" '
                f'y2="{y_hi:.1f}" stroke="#333" stroke-width="1.5"/>'
            )
            parts.append(
                f'<line x1="{cx - cap:.1f}" y1="{y_lo:.1f}" x2="{cx + cap:.1f}" '
                f'y2="{y_lo:.1f}" stroke="#333" stroke-width="1.5"/>'
            )
            parts.append(
                f'<text x="{cx:.1f}" y="{y_hi - 8:.1f}" text-anchor="middle" '
                f'font-size="11" font-family="sans-serif">'
                f"{mean:.4f}±{std:.4f}</text>"
            )
        group_center = left + group_w * (group_index + 0.5)
        parts.append(
            f'<text x="{group_center:.1f}" y="{height - 36:.1f}" '
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
        f'<rect x="{width - 230}" y="14" width="14" height="14" fill="{mf_color}"/>'
        f'<text x="{width - 210}" y="26" font-size="13" font-family="sans-serif">'
        f"implicit MF</text>"
        f'<rect x="{width - 120}" y="14" width="14" height="14" fill="{gnn_color}"/>'
        f'<text x="{width - 100}" y="26" font-size="13" font-family="sans-serif">'
        f"GraphSAGE</text>"
    )
    title = (
        '<text x="380" y="28" text-anchor="middle" font-size="16" '
        'font-family="sans-serif">'
        f"MovieLens 100K {comparison['split']} ranking "
        f"(mean ± 1 sample std)</text>"
    )
    subtitle = (
        '<text x="380" y="50" text-anchor="middle" font-size="12" '
        'font-family="sans-serif">'
        f"seeds {seeds} · n={comparison['n_evaluated_users']} eligible users · "
        "not a SOTA chart</text>"
    )
    caption = (
        f'<text x="380" y="{height - 12:.1f}" text-anchor="middle" '
        'font-size="11" font-family="sans-serif">'
        "Error bars are ±1 sample standard deviation (n-1) across seeds"
        "</text>"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<rect width="100%" height="100%" fill="#ffffff"/>\n'
        f"{title}\n{subtitle}\n{axis}\n{''.join(parts)}\n{legend}\n{caption}\n"
        "</svg>\n"
    )


@dataclass(frozen=True)
class MultiseedArtifacts:
    """Paths written by ``write_multiseed_artifacts``."""

    payload: dict[str, Any]
    json_path: Path
    markdown_path: Path
    chart_path: Path


def write_multiseed_artifacts(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    chart_path: str | Path,
    seeds: Sequence[int] | None = None,
    git_commit: str | None = None,
    generated_at: str | None = None,
    python_version: str | None = None,
    numpy_version: str | None = None,
    torch_version: str | None = None,
    torch_geometric_version: str | None = None,
    platform_info: Mapping[str, str] | None = None,
) -> MultiseedArtifacts:
    """Write JSON, markdown, and SVG leaderboard artifacts from measured pairs."""
    json_out = Path(json_path)
    md_out = Path(markdown_path)
    chart_out = Path(chart_path)
    payload = build_multiseed_comparison(
        pairs,
        seeds=seeds,
        chart_path=chart_out,
        table_path=md_out,
        git_commit=git_commit,
        generated_at=generated_at,
        python_version=python_version,
        numpy_version=numpy_version,
        torch_version=torch_version,
        torch_geometric_version=torch_geometric_version,
        platform_info=platform_info,
    )
    markdown = render_multiseed_markdown(payload)
    svg = render_multiseed_svg(payload)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    chart_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(markdown, encoding="utf-8")
    chart_out.write_text(svg, encoding="utf-8")
    return MultiseedArtifacts(
        payload=payload,
        json_path=json_out,
        markdown_path=md_out,
        chart_path=chart_out,
    )


def load_seed_pair(
    mf_path: str | Path,
    gnn_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one measured MF/GraphSAGE pair and check the shared schema."""
    return (
        compare.load_ranking_result(mf_path),
        compare.load_ranking_result(gnn_path),
    )


def historical_single_seed_notes(repo_root: str | Path) -> dict[str, Any]:
    """Confirm Phase 5 files still exist and still label themselves single-seed."""
    root = Path(repo_root)
    notes: dict[str, Any] = {}
    for key, relative in (
        ("implicit_mf", HISTORICAL_SINGLE_SEED["implicit_mf"]),
        ("graphsage", HISTORICAL_SINGLE_SEED["graphsage"]),
    ):
        path = root / relative
        payload = compare.load_ranking_result(path)
        note = str(payload.get("note", "")).lower()
        if "single-seed" not in note:
            raise MultiseedError(f"{relative} is no longer labeled single-seed")
        if "multi-seed leaderboard" not in note:
            raise MultiseedError(
                f"{relative} no longer says it is not a multi-seed leaderboard"
            )
        if payload.get("seed") != CONTINUITY_SEED:
            raise MultiseedError(
                f"{relative} seed must remain {CONTINUITY_SEED}, "
                f"got {payload.get('seed')!r}"
            )
        notes[key] = {
            "path": relative,
            "seed": payload["seed"],
            "recall_at_k": payload["metrics"]["recall_at_k"],
            "ndcg_at_k": payload["metrics"]["ndcg_at_k"],
        }
    comparison_path = root / HISTORICAL_SINGLE_SEED["comparison"]
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    protocol_note = str(comparison.get("protocol_note", "")).lower()
    if "single-seed" not in protocol_note:
        raise MultiseedError(
            "results/gnn_vs_mf_movielens_100k.json is no longer labeled single-seed"
        )
    notes["comparison"] = {"path": HISTORICAL_SINGLE_SEED["comparison"]}
    return notes


def evaluate_mf_seed(
    split: Any,
    manifest: Mapping[str, Any],
    *,
    seed: int,
    k: int,
    target_split: str,
    provenance: Mapping[str, Any],
    config: Any | None = None,
) -> dict[str, Any]:
    """Train implicit MF and evaluate with the shared ranking protocol."""
    import sagerec_baseline as baseline
    import sagerec_metrics as metrics

    mf_config = (
        baseline.movielens_100k_config(seed=seed) if config is None else config
    )
    if mf_config.seed != seed:
        raise MultiseedError(
            f"MF config seed {mf_config.seed} does not match requested seed {seed}"
        )
    model = baseline.train_implicit_mf(split, mf_config)
    report = metrics.evaluate_ranking(
        model, split, target_split=target_split, k=k
    )
    return baseline.mf_result_payload(
        manifest=dict(manifest),
        config=mf_config,
        report=report,
        git_commit=provenance.get("git_commit"),
        generated_at=str(provenance["generated_at"]),
        python_version=str(provenance["python"]),
        numpy_version=str(provenance["numpy"]),
        platform_info=dict(provenance["platform"]),
    )


def evaluate_graphsage_seed(
    split: Any,
    manifest: Mapping[str, Any],
    *,
    seed: int,
    k: int,
    target_split: str,
    provenance: Mapping[str, Any],
    config: Any | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Train native-backed GraphSAGE and evaluate with the shared protocol."""
    import sagerec_graphsage as graphsage
    import sagerec_metrics as metrics
    import sagerec_minibatch as minibatch

    gs_config = (
        graphsage.movielens_100k_config(seed=seed) if config is None else config
    )
    if gs_config.seed != seed:
        raise MultiseedError(
            f"GraphSAGE config seed {gs_config.seed} does not match requested seed {seed}"
        )
    model = graphsage.train_graphsage(split, gs_config, progress=progress)
    if not isinstance(model.sampler, minibatch.NativeMinibatchSampler):
        raise TypeError(
            "GraphSAGE multi-seed run requires NativeMinibatchSampler; "
            "refusing a non-native sampler."
        )
    model.materialize_eval_embeddings()
    report = metrics.evaluate_ranking(
        model, split, target_split=target_split, k=k
    )
    return graphsage.graphsage_result_payload(
        manifest=manifest,
        config=gs_config,
        report=report,
        git_commit=provenance.get("git_commit"),
        generated_at=str(provenance["generated_at"]),
        python_version=str(provenance["python"]),
        numpy_version=str(provenance["numpy"]),
        torch_version=str(provenance["torch"]),
        platform_info=dict(provenance["platform"]),
    )


def seed_work_paths(work_dir: str | Path, seed: int) -> tuple[Path, Path]:
    """Gitignored per-seed JSON paths used to resume a long 100K run."""
    directory = Path(work_dir)
    return (
        directory / f"mf_seed_{seed}.json",
        directory / f"graphsage_seed_{seed}.json",
    )


def maybe_load_completed_pair(
    work_dir: str | Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return a previously measured pair when both resume files exist."""
    mf_path, gnn_path = seed_work_paths(work_dir, seed)
    if not mf_path.is_file() or not gnn_path.is_file():
        return None
    mf, gnn = load_seed_pair(mf_path, gnn_path)
    try:
        compare.assert_comparable(
            mf, gnn, mf_path=mf_path, gnn_path=gnn_path
        )
    except compare.ComparisonError as exc:
        raise MultiseedError(str(exc)) from exc
    if mf.get("seed") != seed or gnn.get("seed") != seed:
        raise MultiseedError(
            f"resume files for seed {seed} have MF seed {mf.get('seed')!r} "
            f"and GraphSAGE seed {gnn.get('seed')!r}"
        )
    return mf, gnn


def write_seed_pair(
    work_dir: str | Path,
    seed: int,
    mf: Mapping[str, Any],
    gnn: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Persist one measured pair under the resume directory."""
    mf_path, gnn_path = seed_work_paths(work_dir, seed)
    mf_path.parent.mkdir(parents=True, exist_ok=True)
    mf_path.write_text(json.dumps(dict(mf), indent=2) + "\n", encoding="utf-8")
    gnn_path.write_text(json.dumps(dict(gnn), indent=2) + "\n", encoding="utf-8")
    return mf_path, gnn_path
