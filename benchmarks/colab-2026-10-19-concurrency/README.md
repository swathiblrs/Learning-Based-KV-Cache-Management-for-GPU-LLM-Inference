# GPU HTTP concurrency-4 smoke check

Preserved original archive: `kv-serving-20261019-152151-942435.zip`.

All 150 request executions matched no-cache output IDs. CUDA float32 SmolLM2-135M, hot seed 99, 32 MiB, one repeat, four generated tokens. Independent checks verified output equality, per-request sequence coverage, memory budget, trace/predictor hashes and p95 recalculation. Prometheus reports 30 successes per configuration. One configuration reordered the first two requests; this is not an identical-order policy comparison.

HTTP p95 ranged from 548.9 ms (no cache) to 634.9 ms (learned original copy). Queue p95 ranged from 410.5 to 485.9 ms. This verifies serialized-service queueing and correctness, not improved scalability. The archive records CUDA but not the hardware model. No end-to-end cache speedup was established.
