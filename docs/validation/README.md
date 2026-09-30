# Initial local validation

Run on September 30, 2026. These are local development results, **not NVIDIA GPU
results**. The checked-in artifacts are an example snapshot, not a trained model
recommended for deployment.

- Python 3.12.14, NumPy 2.5.3, PyTorch 2.14.0, Transformers 4.57.6.
- 14 unit/inference tests passed, including actual tiny-Llama KV reuse.
- Default synthetic experiment completed all 90 simulator runs.
- CPU tiny-model CLI benchmark completed 30 requests per policy, with identical
  greedy outputs across no-cache, FIFO, LRU, LFU, and learned policies.
- Neither CUDA nor MPS was available to this execution environment.

Mean simulated request hit rates at the 1 MiB budget, three held-out seeds:

| Workload | FIFO | LRU | LFU | Learned |
|---|---:|---:|---:|---:|
| Stable hot prefixes | 0.658 | 0.727 | 0.790 | 0.808 |
| Popularity shift | 0.632 | 0.693 | 0.552 | 0.704 |
| Sequential scan | 0.000 | 0.000 | 0.000 | 0.000 |

The synthetic workload favors reusable hot prefixes. These numbers cannot be
used to claim real-workload generalization or GPU speedup. In the tiny CPU run,
no-cache had a lower p95 TTFT than every caching policy: cloning and bookkeeping
can cost more than the avoided computation. CPU timing is noisy and this single
short run is a correctness smoke test, not a statistical performance conclusion.

Reproduce with the README commands. Run IDs/seeds, budgets, per-run measurements,
and predictor parameters are in the JSON files next to this document.
