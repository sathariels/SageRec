# SageRec HTTP recommend latency

One-machine measurement of `GET /recommend` against a local uvicorn process. Retrieval-only is ADR-009 NumPy top-K. Retrieval plus re-rank is the ADR-010 MLP over the stage-1 pool. Not a ranking-quality result and not a production SLO. Top-level numbers in the JSON duplicate retrieval-only.

- Protocol: `serving-latency-v2` (ADR-010)
- Catalog: `synthetic` with 943 users, 1682 items, dim 16, seed 7
- Workload: 1000 measured requests per mode, 100 warm-up, concurrency 16, k=10, stage1_n=200
- Percentile method: `linear`
- Python: 3.12.3; CPU: Intel(R) Xeon(R) Processor (4 logical); Linux 6.12.94+
- Commit: `03c7891040c9972abb6621845f9803544642ef5a` (dirty=True)

| Mode | p50 ms | p95 ms | p99 ms | requests/s |
| --- | ---: | ---: | ---: | ---: |
| retrieval_only | 23.443567 | 66.870426 | 93.590268 | 534.012774 |
| retrieval_plus_rerank | 21.012950 | 66.638233 | 94.781428 | 568.753940 |

One-machine HTTP latency for GET /recommend on a local uvicorn process, measured twice: retrieval-only (ADR-009 NumPy top-K) and retrieval plus re-rank (ADR-010 MLP over the stage-1 pool). Top-level latency_ms duplicates retrieval_only. Not a ranking-quality result, not sampler timing, and not a production SLO. The catalog is seeded synthetic noise with MovieLens 100K user/item counts; no ratings were read. The re-ranker is a seeded synthetic MLP over synthetic popularity, genre, and recency tables, not the MovieLens-trained model. git_dirty was true at generation, so git_commit is the HEAD the process saw and may not contain this results file.
