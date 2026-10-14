# Colab results — October 13, 2026

These are actual user-downloaded Colab artifacts, preserved without changing their
contents. Each run contains the original ZIP and an extracted `reports/` directory.
`archive-index.json` records original filenames, sizes, and SHA-256 checksums.
The original files remain in the user's Downloads folder.

| Directory | Scope | Correctness status |
|---|---|---|
| `initial-fp16` | T4, FP16, 100 requests, 5 strategies, 3 repeats | Failed: cached and no-cache output hashes differ |
| `short-fp32` | T4, float32, 30 requests, 5 strategies, 1 repeat | New `gpu-fp32-check.json` passed; ZIP also contains the old failed `gpu.json` |
| `extended-fp32` | T4, float32, 150 requests, 5 strategies, 3 repeats, 3 workloads, 2 budgets | All 90 runs completed; all six group correctness checks passed |

## Extended experiment

The suite covers hot-prefix, popularity-shift, and sequential-scan workloads at
8 and 32 MiB retained-KV budgets, with four generated tokens per request. It ran
13,500 request executions with SmolLM2-135M. Training seed is 42; the held-out
workload seed is 99. Three repeats measure timing variability, not independent
workload samples. Inputs are synthetic token streams, not production prompts.

`extended-summary.csv` reports all 30 workload/budget/strategy combinations. Its
latency column is the **median of three per-run p95 values**, not a pooled p95.
The JSON reports preserve individual runs, token outputs and diagnostics.

Selected 32 MiB results:

| Workload | LRU hit rate | Learned hit rate | LRU median run-p95 TTFT | Learned median run-p95 TTFT |
|---|---:|---:|---:|---:|
| Hot prefixes | 50.0% | 55.3% | 63.25 ms | 65.19 ms |
| Popularity shift | 49.3% | 50.0% | 60.62 ms | 66.14 ms |
| Scan | 0% | 0% | 61.29 ms | 71.16 ms |

The learned policy sometimes improved reuse over LRU but did **not** demonstrate
a latency or throughput advantage in these measurements. It tied LFU's 83/150
hits on hot-prefix traffic at 32 MiB. Cache-copying and decision overhead are
hypotheses to investigate through profiling, not established causal conclusions.
Passing float32 checks does not establish FP16 correctness. Do not use the failed
FP16 timing results to claim a correct implementation's speedup.

## Reproduction and provenance limits

Use the repository's suite or single-group notebooks. Exact package versions,
hardware, dtype, predictor hash, trace hash and measurements are in each report.
These original artifacts do not record a Git revision; no exact code commit has
been retroactively assigned. Setup/dependency warnings and full notebook logs
were not included in the downloaded ZIPs. No downloaded model weights are stored
here. The archive copies intentionally preserve duplicated earlier results.
