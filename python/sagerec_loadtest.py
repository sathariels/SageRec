"""Local HTTP load test for ADR-009 retrieval and ADR-010 re-ranking.

Starts a uvicorn subprocess on a synthetic catalog (or a caller-supplied
artifact), issues concurrent ``GET /recommend`` requests with httpx, and
writes p50/p95/p99 latency plus throughput for retrieval-only and for
retrieval plus re-rank. Default user and item counts match the public
MovieLens 100K cardinality. Embeddings and the re-rank MLP are seeded
noise, not a trained model.

This module does not download MovieLens and does not import ``graph_sampler``.
httpx and uvicorn are imported only when a load test runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sagerec_rerank as rerank
import sagerec_serving as serving

PROTOCOL_VERSION = "serving-latency-v2"
ADR_ID = "ADR-010"
RETRIEVAL_ADR = "ADR-009"
DEFAULT_NUM_USERS = 943
DEFAULT_NUM_ITEMS = 1682
DEFAULT_EMBEDDING_DIM = 16
DEFAULT_SEEN_PER_USER = 20
DEFAULT_SEED = 7
DEFAULT_REQUESTS = 1000
DEFAULT_WARMUP = 100
DEFAULT_CONCURRENCY = 16
DEFAULT_K = serving.DEFAULT_K
DEFAULT_STAGE1_N = rerank.DEFAULT_STAGE1_N
PERCENTILE_METHOD = "linear"
LATENCY_MODES = ("retrieval_only", "retrieval_plus_rerank")

REQUIRED_LATENCY_KEYS = (
    "benchmark",
    "protocol_version",
    "adr",
    "endpoint",
    "catalog",
    "workload",
    "latency_ms",
    "throughput_requests_per_s",
    "wall_seconds",
    "n_requests",
    "n_errors",
    "modes",
    "cpu",
    "platform",
    "python",
    "dependencies",
    "git_commit",
    "git_dirty",
    "generated_at",
    "note",
)


class LoadTestError(RuntimeError):
    """The load test could not produce a usable measurement."""


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


def latency_percentiles_ms(seconds: Sequence[float]) -> dict[str, float | str]:
    """p50/p95/p99 via ``numpy.percentile`` (linear), plus mean/min/max."""
    if not seconds:
        raise LoadTestError("latency sample is empty")
    arr = np.asarray(list(seconds), dtype=np.float64) * 1000.0
    if not np.isfinite(arr).all():
        raise LoadTestError("latency sample contains non-finite values")
    return {
        "p50": float(np.percentile(arr, 50, method=PERCENTILE_METHOD)),
        "p95": float(np.percentile(arr, 95, method=PERCENTILE_METHOD)),
        "p99": float(np.percentile(arr, 99, method=PERCENTILE_METHOD)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "percentile_method": PERCENTILE_METHOD,
    }


def _dependency_versions() -> dict[str, str]:
    import fastapi
    import httpx
    import uvicorn

    return {
        "numpy": np.__version__,
        "fastapi": fastapi.__version__,
        "uvicorn": uvicorn.__version__,
        "httpx": httpx.__version__,
    }


def build_latency_payload(
    *,
    catalog: Mapping[str, Any],
    workload: Mapping[str, Any],
    latency_ms: Mapping[str, Any],
    throughput_requests_per_s: float,
    wall_seconds: float,
    n_requests: int,
    n_errors: int,
    modes: Mapping[str, Any],
    root: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Machine-readable serving-latency record. Does not invent timings.

    Top-level latency fields duplicate ``modes['retrieval_only']``. The
    re-rank mode lives under ``modes['retrieval_plus_rerank']``.
    """
    root = repo_root() if root is None else root
    missing_modes = [name for name in LATENCY_MODES if name not in modes]
    if missing_modes:
        raise LoadTestError(f"latency modes missing {missing_modes}")
    dirty = git_dirty(root)
    note = (
        "One-machine HTTP latency for GET /recommend on a local uvicorn "
        "process, measured twice: retrieval-only (ADR-009 NumPy top-K) and "
        "retrieval plus re-rank (ADR-010 MLP over the stage-1 pool). "
        "Top-level latency_ms duplicates retrieval_only. Not a "
        "ranking-quality result, not sampler timing, and not a production SLO."
    )
    if catalog.get("kind") == "synthetic":
        note += (
            " The catalog is seeded synthetic noise with MovieLens 100K "
            "user/item counts; no ratings were read."
        )
    if catalog.get("rerank_kind") == "synthetic":
        note += (
            " The re-ranker is a seeded synthetic MLP over synthetic "
            "popularity, genre, and recency tables, not the MovieLens-trained "
            "model."
        )
    if dirty:
        note += (
            " git_dirty was true at generation, so git_commit is the HEAD "
            "the process saw and may not contain this results file."
        )
    return {
        "benchmark": "sagerec_http_recommend",
        "protocol_version": PROTOCOL_VERSION,
        "adr": ADR_ID,
        "retrieval_adr": RETRIEVAL_ADR,
        "endpoint": "GET /recommend",
        "catalog": dict(catalog),
        "workload": dict(workload),
        "latency_ms": dict(latency_ms),
        "throughput_requests_per_s": float(throughput_requests_per_s),
        "wall_seconds": float(wall_seconds),
        "n_requests": int(n_requests),
        "n_errors": int(n_errors),
        "modes": {name: dict(modes[name]) for name in LATENCY_MODES},
        "cpu": cpu_info(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": sys.version.split()[0],
        "dependencies": _dependency_versions(),
        "git_commit": git_commit(root),
        "git_dirty": dirty,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "note": note,
    }


def render_serving_latency_markdown(payload: Mapping[str, Any]) -> str:
    """Markdown table rendered from a latency payload (no extra rounding)."""
    missing = [key for key in REQUIRED_LATENCY_KEYS if key not in payload]
    if missing:
        raise LoadTestError(f"latency payload is missing keys {missing}")
    latency = payload["latency_ms"]
    catalog = payload["catalog"]
    workload = payload["workload"]
    cpu = payload["cpu"]

    def fmt(value: float) -> str:
        return f"{float(value):.6f}"

    modes = payload["modes"]
    for mode_name in LATENCY_MODES:
        if mode_name not in modes:
            raise LoadTestError(f"latency payload is missing modes.{mode_name}")
        mode_latency = modes[mode_name]["latency_ms"]
        for key in ("p50", "p95", "p99", "mean"):
            if key not in mode_latency:
                raise LoadTestError(f"modes.{mode_name}.latency_ms is missing {key}")

    lines = [
        "# SageRec HTTP recommend latency",
        "",
        "One-machine measurement of `GET /recommend` against a local uvicorn "
        "process. Retrieval-only is ADR-009 NumPy top-K. Retrieval plus "
        "re-rank is the ADR-010 MLP over the stage-1 pool. Not a "
        "ranking-quality result and not a production SLO. Top-level numbers "
        "in the JSON duplicate retrieval-only.",
        "",
        f"- Protocol: `{payload['protocol_version']}` ({payload['adr']})",
        (
            f"- Catalog: `{catalog['kind']}` with {catalog['num_users']} users, "
            f"{catalog['num_items']} items, dim {catalog['embedding_dim']}, "
            f"seed {catalog.get('seed')}"
        ),
        (
            f"- Workload: {workload['n_requests']} measured requests per mode, "
            f"{workload['warmup_requests']} warm-up, concurrency "
            f"{workload['concurrency']}, k={workload['k']}, "
            f"stage1_n={workload.get('stage1_n')}"
        ),
        f"- Percentile method: `{latency['percentile_method']}`",
        (
            f"- Python: {payload['python']}; CPU: {cpu['model']} "
            f"({cpu['logical_cpus']} logical); {payload['platform']['system']} "
            f"{payload['platform']['release']}"
        ),
        f"- Commit: `{payload['git_commit']}` (dirty={payload['git_dirty']})",
        "",
        "| Mode | p50 ms | p95 ms | p99 ms | requests/s |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode_name in LATENCY_MODES:
        mode = modes[mode_name]
        mode_latency = mode["latency_ms"]
        lines.append(
            f"| {mode_name} | {fmt(mode_latency['p50'])} | "
            f"{fmt(mode_latency['p95'])} | {fmt(mode_latency['p99'])} | "
            f"{fmt(mode['throughput_requests_per_s'])} |"
        )
    lines.extend(
        [
            "",
            payload["note"],
            "",
        ]
    )
    return "\n".join(lines)


def write_latency_results(
    payload: Mapping[str, Any],
    json_path: str | Path,
    md_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and the markdown rendered from that same payload."""
    missing = [key for key in REQUIRED_LATENCY_KEYS if key not in payload]
    if missing:
        raise LoadTestError(f"refusing to write latency results missing {missing}")
    json_dest = Path(json_path)
    md_dest = Path(md_path)
    json_dest.parent.mkdir(parents=True, exist_ok=True)
    md_dest.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, indent=2) + "\n"
    md_text = render_serving_latency_markdown(payload)
    json_dest.write_text(json_text, encoding="utf-8")
    md_dest.write_text(md_text, encoding="utf-8")
    return json_dest, md_dest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_healthy(base_url: str, proc: subprocess.Popen[str], timeout_s: float) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_s
    last_error = "no response"
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise LoadTestError(
                f"uvicorn exited early with status {proc.returncode} "
                f"before /healthz succeeded ({last_error})"
            )
        try:
            with urllib.request.urlopen(base_url + "/healthz", timeout=1) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = str(exc)
            time.sleep(0.05)
            continue
        if body.get("status") != "ok":
            raise LoadTestError(f"/healthz returned unexpected body {body}")
        return body
    raise LoadTestError(f"server did not become healthy within {timeout_s}s ({last_error})")


async def _collect_latencies(
    base_url: str,
    user_ids: Sequence[int],
    *,
    k: int,
    concurrency: int,
    warmup: int,
    extra_params: Mapping[str, Any] | None = None,
) -> tuple[list[float], int, float]:
    import httpx

    timeout = httpx.Timeout(30.0)
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    semaphore = asyncio.Semaphore(concurrency)
    errors = 0

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, limits=limits) as client:

        async def one(user_id: int) -> tuple[int, float]:
            async with semaphore:
                started = time.perf_counter()
                params: dict[str, Any] = {"user_id": int(user_id), "k": int(k)}
                if extra_params:
                    params.update(dict(extra_params))
                response = await client.get("/recommend", params=params)
                elapsed = time.perf_counter() - started
                return response.status_code, elapsed

        warmup_results = await asyncio.gather(
            *(one(int(user_id)) for user_id in user_ids[:warmup])
        )
        for status, _elapsed in warmup_results:
            if status != 200:
                raise LoadTestError(
                    f"warmup recommend request returned HTTP {status}"
                )
        wall_started = time.perf_counter()
        measured = await asyncio.gather(
            *(one(int(user_id)) for user_id in user_ids[warmup:])
        )
        wall_seconds = time.perf_counter() - wall_started

    latencies: list[float] = []
    for status, elapsed in measured:
        if status != 200:
            errors += 1
            continue
        latencies.append(elapsed)
    return latencies, errors, wall_seconds


def _mode_record(
    *,
    rerank_enabled: bool,
    stage1_n: int | None,
    latencies: Sequence[float],
    n_errors: int,
    wall_seconds: float,
    n_requests: int,
) -> dict[str, Any]:
    if n_errors:
        raise LoadTestError(
            f"{n_errors} recommend requests failed; refusing to write latency results"
        )
    if len(latencies) != n_requests:
        raise LoadTestError(
            f"expected {n_requests} latency samples, got {len(latencies)}"
        )
    throughput = float(n_requests) / float(wall_seconds) if wall_seconds > 0 else 0.0
    record: dict[str, Any] = {
        "rerank": bool(rerank_enabled),
        "latency_ms": latency_percentiles_ms(latencies),
        "throughput_requests_per_s": throughput,
        "wall_seconds": float(wall_seconds),
        "n_requests": int(n_requests),
        "n_errors": int(n_errors),
    }
    if rerank_enabled:
        record["stage1_n"] = int(stage1_n if stage1_n is not None else DEFAULT_STAGE1_N)
    return record


def run_http_load_test(
    *,
    artifact_path: str | Path | None = None,
    rerank_path: str | Path | None = None,
    num_users: int = DEFAULT_NUM_USERS,
    num_items: int = DEFAULT_NUM_ITEMS,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    seen_per_user: int = DEFAULT_SEEN_PER_USER,
    seed: int = DEFAULT_SEED,
    n_requests: int = DEFAULT_REQUESTS,
    warmup_requests: int = DEFAULT_WARMUP,
    concurrency: int = DEFAULT_CONCURRENCY,
    k: int = DEFAULT_K,
    stage1_n: int = DEFAULT_STAGE1_N,
    json_out: str | Path | None = None,
    md_out: str | Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Measure recommend latency with and without re-ranking.

    When ``artifact_path`` is omitted, a synthetic catalog is written to a
    temporary file and deleted afterward. The committed default uses that
    synthetic catalog plus a synthetic re-rank MLP. ``stage1_n`` is the
    pool width for the re-rank requests.
    """
    root = repo_root() if root is None else Path(root)
    if (
        n_requests < 1
        or warmup_requests < 0
        or concurrency < 1
        or k < 1
        or stage1_n < k
    ):
        raise LoadTestError(
            "n_requests, concurrency, and k must be >= 1, warmup must be >= 0, "
            "and stage1_n must be >= k, "
            f"got n_requests={n_requests}, warmup={warmup_requests}, "
            f"concurrency={concurrency}, k={k}, stage1_n={stage1_n}"
        )
    python_dir = root / "python"
    if json_out is None:
        json_out = root / "results" / "serving_latency.json"
    if md_out is None:
        md_out = root / "results" / "serving_latency.md"

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if artifact_path is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="sagerec-serve-")
        artifact = Path(temp_dir.name) / "embeddings.npz"
        index = serving.build_synthetic_serving_index(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            seen_per_user=seen_per_user,
            seed=seed,
        )
        serving.save_serving_artifact(artifact, index)
        catalog = {
            "kind": "synthetic",
            "num_users": index.num_users,
            "num_items": index.num_items,
            "embedding_dim": index.embedding_dim,
            "seen_per_user": seen_per_user,
            "seed": seed,
            "model": index.metadata["model"],
            "rerank_kind": "synthetic" if rerank_path is None else "provided",
        }
    else:
        artifact = Path(artifact_path)
        index = serving.load_serving_artifact(artifact)
        catalog = {
            "kind": str(index.metadata.get("catalog_kind") or "provided"),
            "num_users": index.num_users,
            "num_items": index.num_items,
            "embedding_dim": index.embedding_dim,
            "seen_per_user": index.metadata.get("seen_per_user"),
            "seed": index.metadata.get("seed"),
            "model": index.metadata["model"],
            "artifact_file": artifact.name,
            "rerank_kind": "synthetic" if rerank_path is None else "provided",
        }

    rerank_temp: tempfile.TemporaryDirectory[str] | None = None
    if rerank_path is None:
        rerank_temp = tempfile.TemporaryDirectory(prefix="sagerec-rerank-")
        rerank_file = Path(rerank_temp.name) / "rerank.npz"
        bundle = rerank.build_synthetic_rerank_bundle(
            num_users=index.num_users,
            num_items=index.num_items,
            seed=seed,
            stage1_model=str(index.metadata["model"]),
            stage1_n=stage1_n,
        )
        rerank.save_rerank_artifact(rerank_file, bundle)
    else:
        rerank_file = Path(rerank_path)
        bundle = rerank.load_rerank_artifact(rerank_file)
        rerank.assert_rerank_compatible(
            bundle,
            num_users=index.num_users,
            num_items=index.num_items,
            stage1_model=str(index.metadata["model"]),
        )

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    stderr_fd, stderr_name = tempfile.mkstemp(prefix="sagerec-uvicorn-", suffix=".log")
    os.close(stderr_fd)
    stderr_path = Path(stderr_name)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(python_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["SAGEREC_SERVING_ARTIFACT"] = str(artifact.resolve())
    env["SAGEREC_RERANK_ARTIFACT"] = str(rerank_file.resolve())
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "sagerec_api:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=str(root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=stderr_handle,
        text=True,
    )
    try:
        health = _wait_healthy(base_url, proc, timeout_s=20.0)
        if int(health["num_users"]) != index.num_users:
            raise LoadTestError(
                f"/healthz num_users {health['num_users']} does not match "
                f"the artifact ({index.num_users})"
            )
        if health.get("rerank_loaded") is not True:
            raise LoadTestError(
                "/healthz did not report rerank_loaded=true. The load test "
                "measures both modes and needs SAGEREC_RERANK_ARTIFACT."
            )
        rng = np.random.default_rng(seed)
        user_ids = [int(user_id) for user_id in rng.integers(
            0, index.num_users, size=warmup_requests + n_requests
        )]
        mode_params = {
            "retrieval_only": {"rerank": "false"},
            "retrieval_plus_rerank": {"rerank": "true", "n": str(stage1_n)},
        }
        modes: dict[str, Any] = {}
        for mode_name in LATENCY_MODES:
            latencies, n_errors, wall_seconds = asyncio.run(
                _collect_latencies(
                    base_url,
                    user_ids,
                    k=k,
                    concurrency=concurrency,
                    warmup=warmup_requests,
                    extra_params=mode_params[mode_name],
                )
            )
            modes[mode_name] = _mode_record(
                rerank_enabled=mode_name == "retrieval_plus_rerank",
                stage1_n=stage1_n,
                latencies=latencies,
                n_errors=n_errors,
                wall_seconds=wall_seconds,
                n_requests=n_requests,
            )
        retrieval = modes["retrieval_only"]
        workload = {
            "k": int(k),
            "stage1_n": int(stage1_n),
            "n_requests": int(n_requests),
            "warmup_requests": int(warmup_requests),
            "concurrency": int(concurrency),
            "request_seed": int(seed),
            "percentile_method": PERCENTILE_METHOD,
            "server": "uvicorn",
            "client": "httpx.AsyncClient",
            "modes": list(LATENCY_MODES),
        }
        payload = build_latency_payload(
            catalog=catalog,
            workload=workload,
            latency_ms=retrieval["latency_ms"],
            throughput_requests_per_s=retrieval["throughput_requests_per_s"],
            wall_seconds=retrieval["wall_seconds"],
            n_requests=n_requests,
            n_errors=retrieval["n_errors"],
            modes=modes,
            root=root,
        )
        write_latency_results(payload, json_out, md_out)
        return payload
    finally:
        stderr_handle.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if temp_dir is not None:
            temp_dir.cleanup()
        if rerank_temp is not None:
            rerank_temp.cleanup()
        stderr_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load-test GET /recommend with and without re-ranking and write "
            "results/serving_latency.json and .md. Default catalog and "
            "re-ranker are synthetic. Does not download MovieLens."
        )
    )
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument(
        "--rerank",
        type=Path,
        default=None,
        help="Optional re-rank artifact. Default is a synthetic MLP matched to the catalog.",
    )
    parser.add_argument("--num-users", type=int, default=DEFAULT_NUM_USERS)
    parser.add_argument("--num-items", type=int, default=DEFAULT_NUM_ITEMS)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--seen-per-user", type=int, default=DEFAULT_SEEN_PER_USER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--stage1-n", type=int, default=DEFAULT_STAGE1_N)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Defaults to results/serving_latency.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Defaults to results/serving_latency.md",
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
        payload = run_http_load_test(
            artifact_path=args.artifact,
            rerank_path=args.rerank,
            num_users=args.num_users,
            num_items=args.num_items,
            embedding_dim=args.embedding_dim,
            seen_per_user=args.seen_per_user,
            seed=args.seed,
            n_requests=args.requests,
            warmup_requests=args.warmup,
            concurrency=args.concurrency,
            k=args.k,
            stage1_n=args.stage1_n,
            json_out=args.json_out,
            md_out=args.md_out,
        )
    except (LoadTestError, serving.ServingError, rerank.RerankError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    retrieval = payload["modes"]["retrieval_only"]["latency_ms"]
    reranked = payload["modes"]["retrieval_plus_rerank"]["latency_ms"]
    print(
        "retrieval p50_ms={p50:.6f} p95_ms={p95:.6f} p99_ms={p99:.6f} "
        "rerank p50_ms={r50:.6f} p95_ms={r95:.6f} p99_ms={r99:.6f} cpu={cpu}".format(
            p50=retrieval["p50"],
            p95=retrieval["p95"],
            p99=retrieval["p99"],
            r50=reranked["p50"],
            r95=reranked["p95"],
            r99=reranked["p99"],
            cpu=payload["cpu"]["model"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
