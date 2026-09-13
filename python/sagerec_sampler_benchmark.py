"""Native vs Python reference neighbor-sampler benchmark (Phase 2 timing).

Times ``graph_sampler.BipartiteCSR.sample_neighbors`` against
``sagerec_reference_sampler.sample_neighbors`` on the same ADR-005
without-replacement workload. Parity is verified before any timed
repetition. Workload generation lives outside timed regions.

Default graphs are deterministic and synthetic so CI does not download
MovieLens. An optional train-only 100K path exists for live timing and
must not be the default.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import graph_sampler
import sagerec_reference_sampler as reference

PROTOCOL_VERSION = "sampler-timing-v1"
BENCHMARK_NAME = "native_vs_reference_sample_neighbors"
REPLACEMENT_POLICY = "uniform_without_replacement"
TIMING_STATISTIC = "median"
ADR_ID = "ADR-005"

REQUIRED_TIMING_KEYS = (
    "benchmark",
    "protocol_version",
    "parity_verified",
    "replacement_policy",
    "adr",
    "seed",
    "graph",
    "workload",
    "timing_statistic",
    "native",
    "reference",
    "speedup",
    "build_type",
    "compiler",
    "python",
    "cpu",
    "platform",
    "git_commit",
    "generated_at",
    "note",
)
REQUIRED_GRAPH_KEYS = (
    "kind",
    "num_users",
    "num_movies",
    "num_nodes",
    "num_undirected_edges",
    "num_directed_csr_entries",
)
REQUIRED_WORKLOAD_KEYS = (
    "k",
    "n_queries",
    "warmup_repetitions",
    "measured_repetitions",
    "replacement_policy",
)
REQUIRED_SIDE_KEYS = (
    "implementation",
    "seconds",
    "latency_ns_per_query",
    "throughput_queries_per_s",
)
REQUIRED_STAT_KEYS = ("median", "mean", "min", "max", "repetitions")

Query = tuple[int, int, int]
SampleFn = Callable[[int, int, int], Sequence[int]]
NativeSampleFn = Callable[[int, int, int], Sequence[int]]


class BenchmarkError(ValueError):
    """Invalid benchmark configuration, failed parity, or unusable timings."""


@dataclass(frozen=True)
class SamplerBenchmarkConfig:
    """Explicit sampler-timing configuration.

    ``source='synthetic'`` builds a deterministic regular-ish bipartite
    graph. ``source='movielens-100k'`` uses train-only pairs from an
    already-prepared ``data/processed/`` directory and never downloads.
    """

    source: str = "synthetic"
    num_users: int = 512
    num_movies: int = 1024
    degree: int = 32
    k: int = 10
    n_queries: int = 4000
    warmup_repetitions: int = 3
    measured_repetitions: int = 9
    seed: int = 7
    processed_dir: Path | None = None


@dataclass(frozen=True)
class TimingArtifacts:
    """Paths written by ``write_timing_artifacts``."""

    payload: dict[str, Any]
    json_path: Path
    markdown_path: Path
    chart_path: Path


def published_synthetic_config() -> SamplerBenchmarkConfig:
    """Workload used for the committed ``results/sampler_timing.*`` artifacts."""
    return SamplerBenchmarkConfig()


def ci_smoke_config() -> SamplerBenchmarkConfig:
    """Tiny deterministic graph for default CI. Does not download MovieLens."""
    return SamplerBenchmarkConfig(
        source="synthetic",
        num_users=8,
        num_movies=16,
        degree=4,
        k=2,
        n_queries=16,
        warmup_repetitions=1,
        measured_repetitions=2,
        seed=7,
    )


def synthetic_train_pairs(
    num_users: int, num_movies: int, degree: int
) -> list[tuple[int, int]]:
    """Deterministic bipartite edges: user ``u`` meets ``degree`` movies."""
    if num_users <= 0 or num_movies <= 0:
        raise BenchmarkError(
            f"synthetic graph needs positive num_users/num_movies, "
            f"got {num_users} / {num_movies}"
        )
    if degree <= 0:
        raise BenchmarkError(f"synthetic degree must be >= 1, got {degree}")
    if degree > num_movies:
        raise BenchmarkError(
            f"synthetic degree {degree} exceeds num_movies {num_movies}"
        )
    pairs: list[tuple[int, int]] = []
    for user_id in range(num_users):
        for offset in range(degree):
            movie_id = (user_id * degree + offset) % num_movies
            pairs.append((user_id, movie_id))
    return pairs


def build_queries(
    graph: object,
    *,
    k: int,
    n_queries: int,
    seed: int,
) -> list[Query]:
    """Build the node-query workload outside any timed region."""
    if k < 0:
        raise BenchmarkError(f"Invalid sample size k={k}; expected k >= 0")
    if n_queries <= 0:
        raise BenchmarkError(f"n_queries must be >= 1, got {n_queries}")
    if seed < 0:
        raise BenchmarkError(f"Invalid seed {seed}; expected a non-negative integer")
    num_nodes = int(graph.num_nodes)  # type: ignore[attr-defined]
    preferred = [
        node_id
        for node_id in range(num_nodes)
        if int(graph.degree(node_id)) > k  # type: ignore[attr-defined]
    ]
    fallback = [
        node_id
        for node_id in range(num_nodes)
        if int(graph.degree(node_id)) > 0  # type: ignore[attr-defined]
    ]
    candidates = preferred or fallback
    if not candidates:
        raise BenchmarkError("graph has no positive-degree nodes to query")
    queries: list[Query] = []
    for index in range(n_queries):
        node_id = candidates[index % len(candidates)]
        query_seed = seed + index
        queries.append((node_id, k, query_seed))
    return queries


def verify_parity(
    graph: object,
    queries: Sequence[Query],
    offsets: Sequence[int],
    neighbors: Sequence[int],
    *,
    native_sample: NativeSampleFn | None = None,
    reference_sample: Callable[
        [Sequence[int], Sequence[int], int, int, int], Sequence[int]
    ]
    | None = None,
) -> None:
    """Fail unless native and reference agree on every query (ADR-005)."""
    if native_sample is None:
        native_sample = graph.sample_neighbors  # type: ignore[attr-defined]
    if reference_sample is None:
        reference_sample = reference.sample_neighbors
    if not queries:
        raise BenchmarkError("parity requires a non-empty query workload")
    for node_id, k, seed in queries:
        native = list(native_sample(node_id, k, seed))
        ref = list(reference_sample(offsets, neighbors, node_id, k, seed))
        if native != ref:
            raise BenchmarkError(
                "native/reference parity failed before timing for "
                f"node_id={node_id} k={k} seed={seed}: "
                f"native={native!r} reference={ref!r}"
            )
        if len(native) != len(set(native)):
            raise BenchmarkError(
                f"sampled neighbors are not unique for node_id={node_id} "
                f"k={k} seed={seed}: {native!r}"
            )


def measure_seconds(
    sample: SampleFn,
    queries: Sequence[Query],
    *,
    warmup_repetitions: int,
    measured_repetitions: int,
    clock: Callable[[], float] = time.perf_counter,
) -> list[float]:
    """Warm up, then time full-workload repetitions. Returns seconds per rep."""
    if warmup_repetitions < 0:
        raise BenchmarkError(
            f"warmup_repetitions must be >= 0, got {warmup_repetitions}"
        )
    if measured_repetitions <= 0:
        raise BenchmarkError(
            f"measured_repetitions must be >= 1, got {measured_repetitions}"
        )
    for _ in range(warmup_repetitions):
        for node_id, k, seed in queries:
            sample(node_id, k, seed)
    times: list[float] = []
    for _ in range(measured_repetitions):
        started = clock()
        for node_id, k, seed in queries:
            sample(node_id, k, seed)
        times.append(clock() - started)
    if any(value <= 0.0 for value in times):
        raise BenchmarkError(
            f"measured repetition seconds must be positive, got {times!r}"
        )
    return times


def summarize_seconds(times: Sequence[float], n_queries: int) -> dict[str, Any]:
    """Latency/throughput summaries from measured repetition seconds."""
    if n_queries <= 0:
        raise BenchmarkError(f"n_queries must be >= 1, got {n_queries}")
    values = [float(item) for item in times]
    if not values:
        raise BenchmarkError("no measured repetitions to summarize")
    median_s = statistics.median(values)
    mean_s = statistics.mean(values)
    return {
        "seconds": {
            "median": median_s,
            "mean": mean_s,
            "min": min(values),
            "max": max(values),
            "repetitions": values,
        },
        "latency_ns_per_query": {
            "median": (median_s / n_queries) * 1e9,
            "mean": (mean_s / n_queries) * 1e9,
        },
        "throughput_queries_per_s": {
            "median": n_queries / median_s,
            "mean": n_queries / mean_s,
        },
    }


def compute_speedup(reference_seconds: float, native_seconds: float) -> float:
    """``reference / native`` from measured times only."""
    if isinstance(reference_seconds, bool) or isinstance(native_seconds, bool):
        raise BenchmarkError("speedup inputs must be real seconds, not bool")
    if native_seconds <= 0.0:
        raise BenchmarkError(
            f"native seconds must be positive to compute speedup, got {native_seconds}"
        )
    if reference_seconds <= 0.0:
        raise BenchmarkError(
            "reference seconds must be positive to compute speedup, "
            f"got {reference_seconds}"
        )
    return reference_seconds / native_seconds


def git_commit(repo_root: str | Path | None = None) -> str | None:
    cwd = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def parse_cmake_cache(build_dir: str | Path | None) -> dict[str, str]:
    """Read compiler / build-type fields from ``CMakeCache.txt`` when present."""
    if build_dir is None:
        return {}
    cache_path = Path(build_dir) / "CMakeCache.txt"
    if not cache_path.is_file():
        return {}
    wanted = {
        "CMAKE_BUILD_TYPE": "build_type",
        "CMAKE_CXX_COMPILER": "compiler_path",
        "CMAKE_CXX_COMPILER_ID": "compiler_id",
        "CMAKE_CXX_COMPILER_VERSION": "compiler_version",
    }
    found: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        key, sep, rest = stripped.partition(":")
        if not sep or key not in wanted:
            continue
        _type, _eq, value = rest.partition("=")
        if _eq:
            found[wanted[key]] = value.strip()
    return found


def compiler_version_string(compiler_path: str | None) -> str | None:
    if not compiler_path:
        return None
    try:
        output = subprocess.check_output(
            [compiler_path, "--version"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    first = output.strip().splitlines()
    return first[0] if first else None


def infer_compiler_id(compiler_path: str | None, version_line: str | None) -> str:
    name = Path(compiler_path).name.lower() if compiler_path else ""
    blob = f"{name} {version_line or ''}".lower()
    if "clang" in blob:
        return "Clang"
    if "g++" in blob or name.startswith("gcc") or "gcc" in blob:
        return "GNU"
    return "unknown"


def compiler_dumpversion(compiler_path: str | None) -> str | None:
    if not compiler_path:
        return None
    for flag in ("-dumpfullversion", "-dumpversion"):
        try:
            value = subprocess.check_output(
                [compiler_path, flag],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        if value:
            return value
    return None


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


def collect_environment(
    *,
    repo_root: str | Path | None = None,
    build_dir: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Provenance: commit, Python, compiler, CPU, build type."""
    cmake = parse_cmake_cache(build_dir)
    compiler_path = cmake.get("compiler_path")
    version_line = compiler_version_string(compiler_path)
    compiler_id = cmake.get("compiler_id") or infer_compiler_id(
        compiler_path, version_line
    )
    compiler_version = cmake.get("compiler_version") or compiler_dumpversion(
        compiler_path
    )
    return {
        "git_commit": git_commit(repo_root),
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "build_type": cmake.get("build_type") or "unknown",
        "compiler": {
            "id": compiler_id,
            "path": compiler_path or "unknown",
            "version": compiler_version or "unknown",
            "version_line": version_line,
        },
        "cpu": cpu_info(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cmake_cache": str(Path(build_dir) / "CMakeCache.txt")
        if build_dir is not None
        else None,
    }


def _graph_metadata(
    graph: object,
    *,
    kind: str,
    num_users: int,
    num_movies: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    neighbors = graph.neighbors()  # type: ignore[attr-defined]
    payload: dict[str, Any] = {
        "kind": kind,
        "num_users": num_users,
        "num_movies": num_movies,
        "num_nodes": int(graph.num_nodes),  # type: ignore[attr-defined]
        "num_undirected_edges": int(len(neighbors) // 2),
        "num_directed_csr_entries": int(len(neighbors)),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def load_benchmark_graph(
    config: SamplerBenchmarkConfig,
) -> tuple[object, dict[str, Any]]:
    """Construct the CSR used for parity and timing."""
    if config.source == "synthetic":
        pairs = synthetic_train_pairs(
            config.num_users, config.num_movies, config.degree
        )
        graph = graph_sampler.BipartiteCSR(
            config.num_users, config.num_movies, pairs
        )
        metadata = _graph_metadata(
            graph,
            kind="synthetic_bipartite",
            num_users=config.num_users,
            num_movies=config.num_movies,
            extra={
                "requested_degree": config.degree,
                "construction": (
                    "deterministic regular-ish bipartite: user u connects to "
                    "movies (u * degree + offset) % num_movies"
                ),
            },
        )
        return graph, metadata
    if config.source == "movielens-100k":
        if config.processed_dir is None:
            raise BenchmarkError(
                "source='movielens-100k' requires processed_dir pointing at "
                "existing ADR-003 artifacts. Default CI must use synthetic graphs."
            )
        import sagerec_dataset as dataset

        split = dataset.load_split_from_processed(config.processed_dir)
        pairs = split.train_positive_pairs()
        held_out = split.positive_pair_set("validation") | split.positive_pair_set(
            "test"
        )
        overlap = set(pairs) & held_out
        if overlap:
            user_id, movie_id = next(iter(overlap))
            raise BenchmarkError(
                "held-out positive leaked into timing graph: "
                f"(user_id={user_id}, movie_id={movie_id})"
            )
        graph = graph_sampler.BipartiteCSR(split.num_users, split.num_movies, pairs)
        metadata = _graph_metadata(
            graph,
            kind="movielens_100k_train_only",
            num_users=split.num_users,
            num_movies=split.num_movies,
            extra={
                "processed_dir": str(Path(config.processed_dir)),
                "construction": (
                    "ADR-003 train-only positives; held-out edges excluded"
                ),
            },
        )
        return graph, metadata
    raise BenchmarkError(
        f"unknown benchmark source {config.source!r}; "
        "expected 'synthetic' or 'movielens-100k'"
    )


def build_timing_payload(
    *,
    graph_meta: Mapping[str, Any],
    workload: Mapping[str, Any],
    native_times: Sequence[float],
    reference_times: Sequence[float],
    environment: Mapping[str, Any],
    seed: int,
    note: str | None = None,
    chart_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble the machine-readable timing record from measured seconds."""
    n_queries = int(workload["n_queries"])
    native = summarize_seconds(native_times, n_queries)
    ref = summarize_seconds(reference_times, n_queries)
    native_median = float(native["seconds"]["median"])
    ref_median = float(ref["seconds"]["median"])
    speedup = compute_speedup(ref_median, native_median)
    payload: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "parity_verified": True,
        "replacement_policy": REPLACEMENT_POLICY,
        "adr": ADR_ID,
        "seed": seed,
        "graph": dict(graph_meta),
        "workload": dict(workload),
        "timing_statistic": {
            "name": TIMING_STATISTIC,
            "note": (
                "Headline latency, throughput, and speedup use the median of "
                "measured repetitions. Mean/min/max are also stored. "
                "Workload generation and parity checks are outside timed regions."
            ),
        },
        "native": {
            "implementation": "graph_sampler.BipartiteCSR.sample_neighbors",
            **native,
        },
        "reference": {
            "implementation": "sagerec_reference_sampler.sample_neighbors",
            **ref,
        },
        "speedup": {
            "statistic": TIMING_STATISTIC,
            "native_over_reference": speedup,
            "formula": "reference_median_seconds / native_median_seconds",
        },
        "build_type": environment.get("build_type", "unknown"),
        "compiler": dict(environment.get("compiler") or {}),
        "python": environment.get("python"),
        "python_implementation": environment.get("python_implementation"),
        "cpu": dict(environment.get("cpu") or {}),
        "platform": dict(environment.get("platform") or {}),
        "git_commit": environment.get("git_commit"),
        "generated_at": environment.get("generated_at"),
        "note": note
        or (
            "One-machine neighbor-sampler timing with ADR-005 without-replacement "
            "semantics. Not a production latency claim and not a quality metric."
        ),
    }
    if chart_path is not None:
        payload["chart_path"] = _stored_ref(chart_path, repo_root)
    if markdown_path is not None:
        payload["markdown_path"] = _stored_ref(markdown_path, repo_root)
    missing = [key for key in REQUIRED_TIMING_KEYS if key not in payload]
    if missing:
        raise BenchmarkError(f"timing payload missing required keys: {missing}")
    return payload


def validate_timing_payload(payload: Mapping[str, Any]) -> None:
    """Assert the stored/generated JSON has the required benchmark schema."""
    missing = [key for key in REQUIRED_TIMING_KEYS if key not in payload]
    if missing:
        raise BenchmarkError(f"timing payload missing required keys: {missing}")
    if payload.get("benchmark") != BENCHMARK_NAME:
        raise BenchmarkError(
            f"benchmark must be {BENCHMARK_NAME!r}, got {payload.get('benchmark')!r}"
        )
    if payload.get("replacement_policy") != REPLACEMENT_POLICY:
        raise BenchmarkError(
            "replacement_policy must be "
            f"{REPLACEMENT_POLICY!r}, got {payload.get('replacement_policy')!r}"
        )
    if payload.get("parity_verified") is not True:
        raise BenchmarkError("parity_verified must be true before trusting timings")
    graph = payload.get("graph")
    if not isinstance(graph, Mapping):
        raise BenchmarkError("graph must be an object")
    missing_graph = [key for key in REQUIRED_GRAPH_KEYS if key not in graph]
    if missing_graph:
        raise BenchmarkError(f"graph is missing required keys: {missing_graph}")
    workload = payload.get("workload")
    if not isinstance(workload, Mapping):
        raise BenchmarkError("workload must be an object")
    missing_work = [key for key in REQUIRED_WORKLOAD_KEYS if key not in workload]
    if missing_work:
        raise BenchmarkError(f"workload is missing required keys: {missing_work}")
    for side in ("native", "reference"):
        block = payload.get(side)
        if not isinstance(block, Mapping):
            raise BenchmarkError(f"{side} must be an object")
        missing_side = [key for key in REQUIRED_SIDE_KEYS if key not in block]
        if missing_side:
            raise BenchmarkError(f"{side} is missing required keys: {missing_side}")
        seconds = block.get("seconds")
        if not isinstance(seconds, Mapping):
            raise BenchmarkError(f"{side}.seconds must be an object")
        missing_stat = [key for key in REQUIRED_STAT_KEYS if key not in seconds]
        if missing_stat:
            raise BenchmarkError(
                f"{side}.seconds is missing required keys: {missing_stat}"
            )
    speedup = payload.get("speedup")
    if not isinstance(speedup, Mapping) or "native_over_reference" not in speedup:
        raise BenchmarkError("speedup.native_over_reference is required")
    value = speedup["native_over_reference"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise BenchmarkError(
            f"speedup.native_over_reference must be a positive number, got {value!r}"
        )
    statistic = payload.get("timing_statistic")
    if not isinstance(statistic, Mapping) or statistic.get("name") != TIMING_STATISTIC:
        raise BenchmarkError(
            f"timing_statistic.name must be {TIMING_STATISTIC!r}"
        )


def _stored_ref(path: str | Path, repo_root: str | Path | None = None) -> str:
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


def render_timing_markdown(payload: Mapping[str, Any]) -> str:
    """Short human-readable summary of measured native vs reference timings."""
    native = payload["native"]
    ref = payload["reference"]
    graph = payload["graph"]
    workload = payload["workload"]
    speedup = payload["speedup"]["native_over_reference"]
    return (
        "# Native vs Python reference neighbor-sampler timing\n"
        "\n"
        f"{payload['note']}\n"
        "\n"
        f"- Protocol: `{payload['protocol_version']}` ({payload['adr']}, "
        f"{payload['replacement_policy']})\n"
        f"- Graph: `{graph['kind']}` with {graph['num_users']} users, "
        f"{graph['num_movies']} movies, {graph['num_nodes']} nodes, "
        f"{graph['num_undirected_edges']} undirected edges "
        f"({graph['num_directed_csr_entries']} directed CSR entries)\n"
        f"- Workload: {workload['n_queries']} queries, k={workload['k']}, "
        f"{workload['warmup_repetitions']} warm-up + "
        f"{workload['measured_repetitions']} measured repetitions, "
        f"seed {payload['seed']}\n"
        f"- Timing statistic: **{payload['timing_statistic']['name']}** of "
        "measured repetitions\n"
        f"- Build: `{payload['build_type']}`; compiler "
        f"`{payload['compiler'].get('id', 'unknown')} "
        f"{payload['compiler'].get('version', '')}`"
        + (
            f" ({payload['compiler']['version_line']})"
            if payload["compiler"].get("version_line")
            else ""
        )
        + "\n"
        f"- Python: {payload['python']}; CPU: {payload['cpu'].get('model')}\n"
        "\n"
        "| Sampler | Median seconds | Mean seconds | Median ns/query | "
        "Median queries/s |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        f"| native `graph_sampler` | {native['seconds']['median']:.6f} | "
        f"{native['seconds']['mean']:.6f} | "
        f"{native['latency_ns_per_query']['median']:.1f} | "
        f"{native['throughput_queries_per_s']['median']:.1f} |\n"
        f"| Python reference | {ref['seconds']['median']:.6f} | "
        f"{ref['seconds']['mean']:.6f} | "
        f"{ref['latency_ns_per_query']['median']:.1f} | "
        f"{ref['throughput_queries_per_s']['median']:.1f} |\n"
        "\n"
        f"Median speedup (reference / native): **{speedup:.3f}×**\n"
        "\n"
        "Parity was verified on the timed query set before measurement. "
        "These numbers are one-machine evidence, not a SOTA or production "
        "latency claim.\n"
    )


def render_timing_svg(payload: Mapping[str, Any]) -> str:
    """Bar chart of median throughput for native vs reference (no extra deps)."""
    native_qps = float(payload["native"]["throughput_queries_per_s"]["median"])
    ref_qps = float(payload["reference"]["throughput_queries_per_s"]["median"])
    speedup = float(payload["speedup"]["native_over_reference"])
    values = [native_qps, ref_qps]
    ymax = max(values + [1.0]) * 1.35
    width, height = 720, 420
    left, right, top, bottom = 80, 30, 56, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    labels = ("native graph_sampler", "Python reference")
    colors = ("#4C78A8", "#F58518")
    bar_w = plot_w * 0.22

    def x_for(index: int) -> float:
        center = left + plot_w * ((index + 0.5) / len(values))
        return center - bar_w / 2

    def format_qps(value: float) -> str:
        if value >= 100:
            return f"{value:,.0f}"
        return f"{value:.1f}"

    bars: list[str] = []
    for index, (value, label, color) in enumerate(zip(values, labels, colors)):
        bar_h = (value / ymax) * plot_h
        x = x_for(index)
        y = top + plot_h - bar_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{bar_h:.1f}" fill="{color}"/>'
        )
        bars.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" '
            f'text-anchor="middle" font-size="12" font-family="sans-serif">'
            f"{format_qps(value)} q/s</text>"
        )
        bars.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 28:.1f}" '
            f'text-anchor="middle" font-size="14" font-family="sans-serif">'
            f"{label}</text>"
        )
    axis = (
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="#333" stroke-width="1.5"/>'
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" '
        f'y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>'
    )
    title = (
        '<text x="360" y="28" text-anchor="middle" font-size="16" '
        'font-family="sans-serif">'
        "Neighbor-sampler median throughput "
        f"(seed {payload['seed']}, k={payload['workload']['k']})</text>"
    )
    subtitle = (
        '<text x="360" y="48" text-anchor="middle" font-size="12" '
        'font-family="sans-serif">'
        f"median speedup {speedup:.2f}× · {payload['graph']['kind']} · "
        f"{payload['workload']['n_queries']} queries</text>"
    )
    ylabel = (
        f'<text x="18" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
        f'font-size="12" font-family="sans-serif" '
        f'transform="rotate(-90 18 {top + plot_h / 2:.1f})">'
        "median queries / s</text>"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<rect width="100%" height="100%" fill="#ffffff"/>\n'
        f"{title}\n{subtitle}\n{axis}\n{ylabel}\n{''.join(bars)}\n"
        "</svg>\n"
    )


def write_timing_artifacts(
    payload: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    chart_path: str | Path,
    repo_root: str | Path | None = None,
) -> TimingArtifacts:
    """Write JSON, markdown, and SVG. Does not invent timings."""
    validate_timing_payload(payload)
    json_out = Path(json_path)
    md_out = Path(markdown_path)
    chart_out = Path(chart_path)
    recorded = dict(payload)
    recorded["chart_path"] = _stored_ref(chart_out, repo_root)
    recorded["markdown_path"] = _stored_ref(md_out, repo_root)
    markdown = render_timing_markdown(recorded)
    svg = render_timing_svg(recorded)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    chart_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(markdown, encoding="utf-8")
    chart_out.write_text(svg, encoding="utf-8")
    return TimingArtifacts(
        payload=recorded,
        json_path=json_out,
        markdown_path=md_out,
        chart_path=chart_out,
    )


def run_benchmark(
    config: SamplerBenchmarkConfig,
    *,
    repo_root: str | Path | None = None,
    build_dir: str | Path | None = None,
    native_sample: NativeSampleFn | None = None,
    reference_sample: Callable[
        [Sequence[int], Sequence[int], int, int, int], Sequence[int]
    ]
    | None = None,
    measure: Callable[..., list[float]] = measure_seconds,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Verify parity, then time native vs reference on one workload."""
    if config.warmup_repetitions < 0 or config.measured_repetitions <= 0:
        raise BenchmarkError(
            "need warmup_repetitions >= 0 and measured_repetitions >= 1"
        )
    graph, graph_meta = load_benchmark_graph(config)
    offsets = list(graph.offsets())  # type: ignore[attr-defined]
    neighbors = list(graph.neighbors())  # type: ignore[attr-defined]
    queries = build_queries(
        graph, k=config.k, n_queries=config.n_queries, seed=config.seed
    )
    native_fn = native_sample or graph.sample_neighbors  # type: ignore[attr-defined]
    ref_fn = reference_sample or reference.sample_neighbors
    verify_parity(
        graph,
        queries,
        offsets,
        neighbors,
        native_sample=native_fn,
        reference_sample=ref_fn,
    )

    def _reference_query(node_id: int, k: int, seed: int) -> Sequence[int]:
        return ref_fn(offsets, neighbors, node_id, k, seed)

    native_times = measure(
        native_fn,
        queries,
        warmup_repetitions=config.warmup_repetitions,
        measured_repetitions=config.measured_repetitions,
    )
    reference_times = measure(
        _reference_query,
        queries,
        warmup_repetitions=config.warmup_repetitions,
        measured_repetitions=config.measured_repetitions,
    )
    environment = collect_environment(
        repo_root=repo_root,
        build_dir=build_dir,
        generated_at=generated_at,
    )
    workload = {
        "k": config.k,
        "n_queries": config.n_queries,
        "warmup_repetitions": config.warmup_repetitions,
        "measured_repetitions": config.measured_repetitions,
        "replacement_policy": REPLACEMENT_POLICY,
        "query_selection": (
            "cycle nodes with degree > k when available, else any "
            "positive-degree node; per-query seed is config.seed + index"
        ),
        "csr_views": (
            "reference reads offsets/neighbors captured once before timing; "
            "native uses graph_sampler.BipartiteCSR.sample_neighbors"
        ),
    }
    return build_timing_payload(
        graph_meta=graph_meta,
        workload=workload,
        native_times=native_times,
        reference_times=reference_times,
        environment=environment,
        seed=config.seed,
        repo_root=repo_root,
    )
