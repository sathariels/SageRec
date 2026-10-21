# SageRec HTTP recommend latency

One-machine ADR-009 measurement of `GET /recommend` against a local uvicorn process. Top-K is exact NumPy. Not a ranking-quality result and not a production SLO.

- Protocol: `serving-latency-v1` (ADR-009)
- Catalog: `synthetic` with 943 users, 1682 items, dim 16, seed 7
- Workload: 1000 measured requests, 100 warm-up, concurrency 16, k=10
- Percentile method: `linear`
- Python: 3.12.3; CPU: Intel(R) Xeon(R) Processor (4 logical); Linux 6.12.94+
- Commit: `70a565104f0aa00cfe85923487d3a5813f13fefb` (dirty=True)

| Statistic | Value |
| --- | ---: |
| p50 ms | 19.909990 |
| p95 ms | 59.100509 |
| p99 ms | 85.913701 |
| mean ms | 24.707682 |
| throughput requests/s | 617.951794 |
| wall seconds | 1.618249 |
| errors | 0 |

One-machine ADR-009 HTTP latency for GET /recommend on a local uvicorn process. Top-K is exact NumPy over precomputed embeddings. Not a ranking-quality result, not sampler timing, and not a production SLO. The catalog is seeded synthetic noise with MovieLens 100K user/item counts; no ratings were read. git_dirty was true at generation, so git_commit is the HEAD the process saw and may not contain this results file.
