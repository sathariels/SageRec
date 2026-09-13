# Native vs Python reference neighbor-sampler timing

One-machine neighbor-sampler timing with ADR-005 without-replacement semantics. Not a production latency claim and not a quality metric.

- Protocol: `sampler-timing-v1` (ADR-005, uniform_without_replacement)
- Graph: `synthetic_bipartite` with 512 users, 1024 movies, 1536 nodes, 16384 undirected edges (32768 directed CSR entries)
- Workload: 4000 queries, k=10, 3 warm-up + 9 measured repetitions, seed 7
- Timing statistic: **median** of measured repetitions
- Build: `Release`; compiler `GNU 13.3.0` (g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0)
- Python: 3.12.3; CPU: Intel(R) Xeon(R) Processor

| Sampler | Median seconds | Mean seconds | Median ns/query | Median queries/s |
| --- | ---: | ---: | ---: | ---: |
| native `graph_sampler` | 0.003911 | 0.003933 | 977.9 | 1022637.6 |
| Python reference | 0.434863 | 0.435244 | 108715.8 | 9198.3 |

Median speedup (reference / native): **111.177×**

Parity was verified on the timed query set before measurement. These numbers are one-machine evidence, not a SOTA or production latency claim.
