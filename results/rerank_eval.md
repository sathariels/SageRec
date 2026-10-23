# SageRec two-stage re-rank evaluation

MovieLens 100K, ADR-003 leave-one-out, test split. Stage 1 is implicit MF retrieval. Stage 2 re-ranks that top-N pool. Not a GraphSAGE result and not the Phase 5 or Phase 7 leaderboard.

- Protocol: `rerank-eval-v1` (ADR-010)
- Seed: 7; stage-1 N: 200; users: 943; stage-1 hit rate @N: 0.357370
- Features: retrieval_score, item_popularity, genre_affinity, item_recency; genres_available=True
- Python: 3.12.3; NumPy: 1.26.4; CPU: Intel(R) Xeon(R) Processor (4 logical)
- Commit: `03c7891040c9972abb6621845f9803544642ef5a` (dirty=True)

| System | Recall@10 | NDCG@10 | MAP@10 |
| --- | ---: | ---: | ---: |
| retrieval only | 0.038176 | 0.020303 | 0.014950 |
| retrieval + re-rank | 0.053022 | 0.026510 | 0.018500 |
| delta (re-rank − retrieval) | 0.014846 | 0.006207 | 0.003550 |

Single-seed MovieLens 100K two-stage comparison (ADR-010). Stage 1 is implicit MF (ADR-004) dot-product retrieval, the same score ADR-009 serves, trained on ADR-003 train positives only. Stage 2 is a NumPy MLP with a linear skip, fit listwise on validation items that stage 1 retrieved, and scored on the test split. Both systems rank inside that stage-1 top-N pool. MAP is MAP@10. Features use train interactions only. Genres come from u.item. This is not a GraphSAGE result, not the Phase 5 or Phase 7 leaderboard, and not a SOTA claim. Re-ranking cannot recover a test item that stage 1 did not retrieve. git_dirty was true at generation, so git_commit is the HEAD the process saw and may not contain this results file.
