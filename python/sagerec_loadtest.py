"""Local HTTP load test for the ADR-009 serving API.

Starts a uvicorn subprocess on a synthetic catalog (or a caller-supplied
artifact), issues concurrent ``GET /recommend`` requests with httpx, and
writes p50/p95/p99 latency plus throughput. Default user and item counts
match the public MovieLens 100K cardinality so the NumPy dot product is a
similar size. Embeddings are seeded noise, not a trained model.

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

import sagerec_serving as serving

PROTOCOL_VERSION = "serving-latency-v1"
ADR_ID = "ADR-009"
DEFAULT_NUM_USERS = 943
DEFAULT_NUM_ITEMS = 1682
DEFAULT_EMBEDDING_DIM = 16
DEFAULT_SEEN_PER_USER = 20
DEFAULT_SEED = 7
DEFAULT_REQUESTS = 1000
DEFAULT_WARMUP = 100
DEFAULT_CONCURRENCY = 16
DEFAULT_K = serving.DEFAULT_K
PERCENTILE_METHOD = "linear"

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
    root: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Machine-readable serving-latency record. Does not invent timings."""
    root = repo_root() if root is None else root
    dirty = git_dirty(root)
    note = (
        "One-machine ADR-009 HTTP latency for GET /recommend on a local "
        "uvicorn process. Top-K is exact NumPy over precomputed embeddings. "
        "Not a ranking-quality result, not sampler timing, and not a "
        "production SLO."
    )
    if catalog.get("kind") == "synthetic":
        note += (
            " The catalog is seeded synthetic noise with MovieLens 100K "
            "user/item counts; no ratings were read."
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
        "endpoint": "GET /recommend",
        "catalog": dict(catalog),
        "workload": dict(workload),
        "latency_ms": dict(latency_ms),
        "throughput_requests_per_s": float(throughput_requests_per_s),
        "wall_seconds": float(wall_seconds),
        "n_requests": int(n_requests),
        "n_errors": int(n_errors),
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

    lines = [
        "# SageRec HTTP recommend latency",
        "",
        "One-machine ADR-009 measurement of `GET /recommend` against a local "
        "uvicorn process. Top-K is exact NumPy. Not a ranking-quality result "
        "and not a production SLO.",
        "",
        f"- Protocol: `{payload['protocol_version']}` ({payload['adr']})",
        (
            f"- Catalog: `{catalog['kind']}` with {catalog['num_users']} users, "
            f"{catalog['num_items']} items, dim {catalog['embedding_dim']}, "
            f"seed {catalog.get('seed')}"
        ),
        (
            f"- Workload: {workload['n_requests']} measured requests, "
            f"{workload['warmup_requests']} warm-up, concurrency "
            f"{workload['concurrency']}, k={workload['k']}"
        ),
        f"- Percentile method: `{latency['percentile_method']}`",
        (
            f"- Python: {payload['python']}; CPU: {cpu['model']} "
            f"({cpu['logical_cpus']} logical); {payload['platform']['system']} "
            f"{payload['platform']['release']}"
        ),
        f"- Commit: `{payload['git_commit']}` (dirty={payload['git_dirty']})",
        "",
        "| Statistic | Value |",
        "| --- | ---: |",
        f"| p50 ms | {fmt(latency['p50'])} |",
        f"| p95 ms | {fmt(latency['p95'])} |",
        f"| p99 ms | {fmt(latency['p99'])} |",
        f"| mean ms | {fmt(latency['mean'])} |",
        f"| throughput requests/s | {fmt(payload['throughput_requests_per_s'])} |",
        f"| wall seconds | {fmt(payload['wall_seconds'])} |",
        f"| errors | {payload['n_errors']} |",
        "",
        payload["note"],
        "",
    ]
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
                response = await client.get(
                    "/recommend",
                    params={"user_id": int(user_id), "k": int(k)},
                )
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


def run_http_load_test(
    *,
    artifact_path: str | Path | None = None,
    num_users: int = DEFAULT_NUM_USERS,
    num_items: int = DEFAULT_NUM_ITEMS,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    seen_per_user: int = DEFAULT_SEEN_PER_USER,
    seed: int = DEFAULT_SEED,
    n_requests: int = DEFAULT_REQUESTS,
    warmup_requests: int = DEFAULT_WARMUP,
    concurrency: int = DEFAULT_CONCURRENCY,
    k: int = DEFAULT_K,
    json_out: str | Path | None = None,
    md_out: str | Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Measure recommend latency and write JSON plus markdown.

    When ``artifact_path`` is omitted, a synthetic catalog is written to a
    temporary file and deleted afterward. The committed default uses that
    synthetic catalog.
    """
    root = repo_root() if root is None else Path(root)
    if n_requests < 1 or warmup_requests < 0 or concurrency < 1 or k < 1:
        raise LoadTestError(
            "n_requests and concurrency and k must be >= 1, and warmup must be >= 0, "
            f"got n_requests={n_requests}, warmup={warmup_requests}, "
            f"concurrency={concurrency}, k={k}"
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
        }

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    stderr_fd, stderr_name = tempfile.mkstemp(prefix="sagerec-uvicorn-", suffix=".log")
    os.close(stderr_fd)
    stderr_path = Path(stderr_name)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(python_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["SAGEREC_SERVING_ARTIFACT"] = str(artifact.resolve())
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
        rng = np.random.default_rng(seed)
        user_ids = rng.integers(0, index.num_users, size=warmup_requests + n_requests)
        latencies, n_errors, wall_seconds = asyncio.run(
            _collect_latencies(
                base_url,
                [int(user_id) for user_id in user_ids],
                k=k,
                concurrency=concurrency,
                warmup=warmup_requests,
            )
        )
        if n_errors:
            raise LoadTestError(
                f"{n_errors} recommend requests failed; refusing to write latency results"
            )
        if len(latencies) != n_requests:
            raise LoadTestError(
                f"expected {n_requests} latency samples, got {len(latencies)}"
            )
        latency_ms = latency_percentiles_ms(latencies)
        throughput = float(n_requests) / float(wall_seconds) if wall_seconds > 0 else 0.0
        workload = {
            "k": int(k),
            "n_requests": int(n_requests),
            "warmup_requests": int(warmup_requests),
            "concurrency": int(concurrency),
            "request_seed": int(seed),
            "percentile_method": PERCENTILE_METHOD,
            "server": "uvicorn",
            "client": "httpx.AsyncClient",
        }
        payload = build_latency_payload(
            catalog=catalog,
            workload=workload,
            latency_ms=latency_ms,
            throughput_requests_per_s=throughput,
            wall_seconds=wall_seconds,
            n_requests=n_requests,
            n_errors=n_errors,
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
        stderr_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load-test GET /recommend and write results/serving_latency.json "
            "and .md. Default catalog is synthetic (MovieLens 100K cardinality, "
            "seeded noise). Does not download MovieLens."
        )
    )
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--num-users", type=int, default=DEFAULT_NUM_USERS)
    parser.add_argument("--num-items", type=int, default=DEFAULT_NUM_ITEMS)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--seen-per-user", type=int, default=DEFAULT_SEEN_PER_USER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
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
            num_users=args.num_users,
            num_items=args.num_items,
            embedding_dim=args.embedding_dim,
            seen_per_user=args.seen_per_user,
            seed=args.seed,
            n_requests=args.requests,
            warmup_requests=args.warmup,
            concurrency=args.concurrency,
            k=args.k,
            json_out=args.json_out,
            md_out=args.md_out,
        )
    except (LoadTestError, serving.ServingError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    latency = payload["latency_ms"]
    print(
        "p50_ms={p50:.6f} p95_ms={p95:.6f} p99_ms={p99:.6f} "
        "throughput_rps={rps:.6f} cpu={cpu}".format(
            p50=latency["p50"],
            p95=latency["p95"],
            p99=latency["p99"],
            rps=payload["throughput_requests_per_s"],
            cpu=payload["cpu"]["model"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
