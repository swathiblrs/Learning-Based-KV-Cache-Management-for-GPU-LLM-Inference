# Local inference serving and GPU validation

This extension turns the offline cache experiment into a persistent, single-process HTTP service. It is a research prototype, not evidence of production deployment. Existing benchmark results remain unchanged. Serving timings include stage synchronization and cannot be compared directly with the original uninstrumented benchmarks.

## Run locally

```sh
pip install -e '.[serving]'
kv-cache experiment --output results/experiment
KV_TINY=1 KV_DEVICE=cpu KV_POLICY=lru KV_BUDGET_BYTES=4096 uvicorn kv_cache.serving:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

The tiny model has random weights: use it only for correctness. For the real model set `KV_TINY=0`, `KV_DEVICE=cuda`, `KV_POLICY=learned`, `KV_PREDICTOR=results/experiment/predictor.json`, and `KV_BUDGET_BYTES=33554432`. Precision is float32; FP16 remains unvalidated.

```sh
curl http://127.0.0.1:8000/readyz
curl http://127.0.0.1:8000/info
curl -X POST http://127.0.0.1:8000/generate -H 'Content-Type: application/json' -d '{"prefix_ids":[3,4,5],"suffix_ids":[6,7],"new_tokens":4}'
curl http://127.0.0.1:8000/metrics
```

The API accepts pre-tokenized prefix/suffix IDs, not text. Tokenize both consistently for the configured model. Prefix identity is derived from the exact prefix IDs; clients cannot alias different prefixes under one name. Output is fixed-length greedy token IDs without early EOS stopping, matching the experimental setup. The API is nonstreaming; `compute_ttft_ms` measures computed first-token time, not time to a client-visible streamed token. HTTP latency includes serialization/network/queueing and is measured by the load runner.

One model and one cache live for the process lifetime. A lock serializes inference. Concurrent HTTP requests queue (default at most 8 admitted requests including the active one); excess requests receive 503. This is not dynamic batching or concurrent GPU inference. Use exactly one Uvicorn worker. Retained KV bytes exclude active decoding tensors, transient copies, model weights, allocator reservations, and CPU metadata. Prefix history is capped at 10,000 identities; new identities then receive 503 until restart. Inference failures mark readiness unhealthy and require restart. State is intentionally not persisted across restarts.

This unauthenticated service is for trusted local experiments only: bind to loopback and do not publish a tunnel or share it between tenants. Compose ports are loopback-only. There is no public reset/admin endpoint.

## Monitoring stack

```sh
docker compose -f deploy/compose.yaml up --build
```

The default is a CPU tiny-model demo. On a Linux NVIDIA host with Docker GPU support and a generated `results/experiment/predictor.json`:

```sh
docker compose -f deploy/compose.yaml -f deploy/compose.gpu.yaml up --build
```

Open Grafana at localhost:3000, Prometheus at localhost:9090, and the API at localhost:8000. The provisioned dashboard shows request throughput/outcomes, cache-hit ratio, p95 handler/queue/compute-first-token latency, KV budget/usage, CUDA allocations, eviction rate, and stage means. GPU allocations are PyTorch tensor allocations, not total device utilization. Prediction time is nested inside eviction time; do not add both. Metrics use bounded labels and histograms, never raw prompts or prefix hashes. Failed requests and queue rejection are exported; FastAPI schema-validation 422s occur before the generation handler and are not counted in its outcome counter.

The dashboard measures service behavior, not predictor accuracy or detected distribution drift. The offline `model-evaluation.json` supplies precision, recall, positive rates and Brier scores for held-out hot/shift/scan traces. Predictor JSON includes training provenance; `/info` exposes predictor SHA256 and the resolved model revision. Serving run bundles preserve predictor bytes, inputs, package versions, source commit/dirty status, output IDs and raw Prometheus metrics. A repeat on one seed measures timing variation, not independent workload evidence.

## HTTP evaluation and copy experiment

```sh
python -m kv_cache.serve_eval --device cpu --tiny --predictor results/experiment/predictor.json --requests 8 --budget-mib .001 --concurrency 1 2 --output results/serving-cpu
```

For a GPU, start with the provided `notebooks/gpu_serving.ipynb`. It runs a short group and downloads the checkpoint ZIP immediately. It downloads the project from GitHub, uses an isolated virtualenv with Colab's CUDA torch, and never needs Drive storage. A runtime reset can still lose files not yet downloaded. The last download cell locates the ZIP even after an ordinary execution error; it does not rerun the benchmark.

The runner starts a fresh local server for each policy/copy-mode/concurrency/repeat, warms the model before the measured trace, and checks every request against no-cache output IDs. It saves after every configuration and stops on mismatches. HTTP failures abort the group and preserve logs. `none` is always included. Default quick policies are none/LRU/learned; all five can be selected with `--policies none fifo lru lfu learned`. Broaden to multiple independent held-out seeds, workloads, budgets and repeats only after the smoke run passes.

Use concurrency=1 for controlled eviction-policy comparisons: the same request order is guaranteed. At higher concurrency, arrival order may differ across processes; reports record service sequence numbers. Those runs evaluate queueing/correctness and must not be presented as identical-order cache-policy comparisons.

`KV_COPY_MODE=baseline` preserves deepcopy+crop+clone retention. The experimental `sliced` mode builds a fresh full-attention DynamicCache and clones only prefix tensor slices. It avoids copying suffix tokens and a second prefix clone on retention; hit-path copying is unchanged. CPU tests verify independent storage and greedy output equivalence. It is not a zero-copy cache and no GPU speedup is claimed. Compare retention-stage timings, end-to-end HTTP latency and CUDA peaks under the same configuration before deciding whether to adopt it. The service defaults to baseline.

## Validation status

Automated tests cover all policies and both copy modes on a tiny CPU model, immutable prefix storage, eviction budgets, invalid IDs/context, metrics, queue overload and unhealthy readiness after failure. Real HTTP smoke evaluation and GPU results must be distinguished from unit tests. Docker/Grafana deployment and real NVIDIA model validation require their corresponding runtime; do not describe their configuration files as an already-running production system.
