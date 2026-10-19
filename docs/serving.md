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


## Optimization experiment (GPU result pending)

Open `notebooks/gpu_optimization.ipynb`. It uses concurrency 1, 60 requests per trace and three repeats. The seven timed configurations are no-cache plus LRU/learned with baseline, sliced and shared copy paths. One independent full-logits no-cache run supplies the strict output oracle. This totals 22 runs / 1,320 request executions. All configurations include a two-pass warmup at the largest input length before timing. No-cache is included in the shuffled configuration order on each repeat.

- `--no-profile-stages` / `KV_PROFILE_STAGES=0` disables synchronized stage instrumentation. Request-boundary/first-token synchronization and lightweight Prometheus collection remain. Missing stage histograms mean disabled, not zero work.
- `--logits-mode last` / `KV_LOGITS_MODE=last` computes only the final position's logits, equally for cached and no-cache policies. Strict generated-token checks against the full-logits reference remain mandatory because changes in matrix shape can change numerical results.
- `--copy-modes shared` / `KV_COPY_MODE=shared` clones retained prefix slices once on a miss. On a hit, it creates separate cache/layer wrappers referring to the immutable resident tensors. Transformers DynamicLayer.update concatenates into fresh tensors and rebinds only the request-local wrapper, avoiding a preliminary deep copy. There is still concatenation and active-KV copying: this is not zero-copy/paged attention. Version and exact cache-layer type guards restrict this path to full-attention Llama / Transformers 4.57.x. The serving extra is pinned to that minor version. CPU tests verify unchanged resident tensor values and lengths across varied suffixes and decode lengths.

`python -m kv_cache.serve_summary RESULTS_DIRECTORY` reports the median of per-run p95s and paired per-repeat reductions relative to no-cache. Positive means faster in that comparison. Negative means slower; do not select only positive repeats. The full-logits oracle is excluded from timing comparisons. No speedup is established until new GPU results pass correctness and repeated timing evaluation.

The default retains the existing short-prefix workload. A separate `--prefix-tokens 1024 --budget-mib 256` experiment can locate a longer-prefix operating regime. SmolLM2-135M float32 uses 46,080 KV bytes per prefix token (45 MiB at 1,024 tokens), so the old 32 MiB budget would admit no such prefix. Increasing length/budget changes the workload and must be reported as such. Length 1,024 is outside the reuse predictor's original 32–256 training range. Do not use that result as a replacement for the original workload or tune on reported held-out traces.

The prior concurrency-1 and concurrency-4 GPU smoke results are preserved separately in `benchmarks/`. New timing mode, logit scope and warmup changes mean the new results must be compared to their own matched no-cache baseline, not directly to historical timings.

## Deployment boundary and longer-prefix validation

`KV_REQUIRE_AUTH=1` requires an API key (environment or `KV_API_KEY_FILE`); all non-health routes are protected. A 64 KiB request-body cap is now enforced before JSON parsing, even for chunked requests. Local experiment defaults remain unauthenticated/loopback. See [deployment runbook](deployment.md) for host prerequisites, verification and limitations.

`notebooks/gpu_long_prefix.ipynb` predefines 1,024-token prefixes, 256 MiB, hot workloads at seeds 101/102/103, three repeats per seed, concurrency 1, and matched no-cache/LRU-shared/learned-shared comparisons. Run each seed separately and retain all results. The primary measures are paired mean HTTP latency and throughput, with p95 also reported. A descriptive all-repeats 5% improvement threshold is declared before these runs; it is not a statistical significance test. A longer-prefix speedup, if observed, applies to that workload, not the prior short-prefix tests or arbitrary production traffic. Frozen predictor training was on shorter prefixes.
