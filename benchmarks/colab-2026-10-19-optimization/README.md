# GPU optimization experiment: short prefixes

Original archive preserved as `original.zip`. NVIDIA T4, SmolLM2-135M float32, hot seed 99, 32 MiB, variable prefix lengths 32–256, concurrency 1. Three repeats of seven configurations (1,260 request executions), plus a 60-request full-logits oracle. All candidate output IDs matched the oracle. Independent checks verified hashes, order, budgets, p95 calculations and aggregate comparisons.

Stage profiling disabled; final-token-only logits used equally in timed runs. Boundary and first-token synchronization remain. All 21 measured runs passed. This is one synthetic trace and three timing repeats, not independent workload replications.

No reliable end-to-end cache speedup was established. Learned shared-prefix p95 was lower than learned original-copy p95 in two of three repeats, with improved throughput in all three, but remained slower than no-cache in two of three repeats. LRU shared-prefix paired p95 reduction versus no-cache had a median of only 0.188%, with large variability. Learned hit rate was 28/60 (46.7%), below LRU 29/60 (48.3%). The previous 150-request offline workload's 55.3%/50.0% result is separate and unchanged.

A longer-prefix workload is a separate experiment, not a replacement for these negative results. Do not claim a general inference speedup from this run.

Archive SHA256: 7407d890759561f5aa7b8fc1bc7c40ede7e8b95fd9fc350c5a83ef92238aa856
