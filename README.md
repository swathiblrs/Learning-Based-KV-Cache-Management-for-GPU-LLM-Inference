# Learning-Based KV-Cache Management for GPU LLM Inference

Can a small reuse predictor make better prefix-cache retention decisions than
traditional eviction rules? This repository provides a runnable research
prototype to investigate that question. It does **not** claim a speedup in advance.

## What is implemented

- Reproducible synthetic hot-prefix, popularity-shift, and scan workloads.
- Byte-budgeted whole-prefix caching: no-cache, FIFO, LRU, LFU, and learned eviction.
- A NumPy logistic-regression reuse predictor with inspectable JSON artifacts.
- Independent train/validation/test traces, complete-horizon labels, and ML metrics.
- Real Hugging Face Transformers KV reuse, fixed-length greedy decoding, and
  synchronized CPU/MPS/CUDA benchmarks; copied KV protects immutable cached prefixes.
- Unit and tiny-model correctness tests without downloading pretrained weights.

This is a sequential research harness, not a vLLM replacement. It does not
implement continuous batching, paged attention, concurrent serving, multi-GPU
execution, custom CUDA kernels, or partial-prefix matching.

## Start locally (no GPU required)

Python 3.10+ is required. On macOS, the system `python3` may be too old.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
kv-cache experiment --output results/experiment
```

The experiment trains on seed 42, validates on seed 43, and tests seeds 44–46
across three workloads and two budgets. It writes:

- `predictor.json`: weights, training-only normalization, and provenance.
- `model-evaluation.json`: precision/recall at 0.5, prevalence, and Brier score.
- `simulation.json`: reuse, retained bytes, evictions, and host policy timing.
- JSONL traces for replay.

Simulation has **no GPU TTFT or inference-throughput result**. Reused token count
is a proxy for prefill work avoided, not a measured FLOP count.

## Validate real KV reuse on CPU

```bash
python -m pip install -e '.[inference]'
python -m unittest discover -s tests -v
kv-cache benchmark --predictor results/experiment/predictor.json \
  --tiny --device cpu --requests 30 --new-tokens 3 --repeats 1 \
  --budget-mib 0.05 --output results/cpu-correctness.json
```

`--tiny` uses a randomly initialized two-layer Llama. It checks execution and
cache semantics; its timings are **not pretrained-LLM performance evidence**.

## Run a real NVIDIA GPU experiment

On an available free GPU notebook, install this repository and select a GPU
runtime. GPU availability and service quotas are not guaranteed. Run benchmarks
inside the notebook; this project requires no paid API or hosted endpoint.

```bash
python -m pip install -e '.[inference]'
kv-cache experiment --output results/experiment
kv-cache benchmark --predictor results/experiment/predictor.json \
  --device cuda --model HuggingFaceTB/SmolLM2-135M \
  --requests 100 --budget-mib 4 --new-tokens 8 --repeats 3 \
  --output results/gpu.json
```

The first pretrained run downloads model weights. An unavailable CUDA device is
an error, not a silent CPU fallback. See [the notebook](notebooks/gpu_experiment.ipynb)
and [experiment protocol](docs/EXPERIMENTS.md).

The harness replays identical inputs for all policies, shuffles policy order,
warms up before every run, synchronizes the accelerator, and saves hardware,
versions, dtype, model identity, input/model-artifact hashes, memory, and latency.
It compares generated-token hashes against the no-cache path. A mismatch exits
with an error and saves diagnostics; numerical differences must be investigated.

## How the learned policy works

At request t, four features describe each eviction candidate: recency, cumulative
frequency, prefix length, and mean historical interarrival gap. Log transforms
and normalization feed a logistic model predicting reuse within the next 32
requests. Labels may look ahead; features never do. The final 32 training steps
are excluded because their labels are incomplete.

On a miss, all policies admit a fitting prefix and evict enough retained entries.
The learned policy evicts the lowest estimated `P(reuse) × prefix_tokens / bytes`
first. This is an approximate retention score, not optimal cache control. For a
fixed full-attention model, bytes scale linearly with tokens, so that ratio
largely cancels; length can still affect the learned probability.

The cache holds independent, exact full prefixes. Active generation uses separate
KV state and finishes before the next request. **Active KV is never evicted.**
The byte budget covers only retained reusable KV tensors; model weights, active
KV, cloning, inputs, and allocator overhead require additional device memory.

## Scope and evidence

- The first Colab T4 run completed but failed strict output equality; see the investigation below.
- Synthetic traces/token IDs do not establish production workload or answer quality.
- Independent test seeds prevent sequence reuse, but distributions remain synthetic.
- Metadata grows with unique prefixes; production deployment needs bounded metadata.
- Learned decisions scan all residents; policy cost can exceed reuse savings.
- No speedup or generalization claim should be made until measured on relevant data.

For a resume, report only measured outcomes, identify the hardware/workloads,
and distinguish simulator results from real GPU experiments.

## References

- [Hugging Face cache documentation](https://huggingface.co/docs/transformers/en/kv_cache)
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- [SGLang / RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)

These systems motivate the experiment; this repository is an independent,
simplified implementation and makes no claim to invent learned caching.

## Initial validation

See [checked-in validation artifacts](docs/validation/README.md) for the first
90 simulation runs and a real-KV CPU correctness smoke test. All 14 tests passed;
These initial results predate the Colab investigation below.


## First T4 run and corrective changes

The user-provided Colab run completed 15 measurements (100 requests × five
strategies × three repeats) on a Tesla T4 using FP16 SmolLM2-135M. All four caching
strategies produced the same hash in every repeat; the no-cache path consistently
produced a different hash. Raw results are preserved in
[the failed-run artifact](docs/validation/t4-first-run.json). They are **not a
passing correctness result** and should not support resume speedup claims.

The old miss path computed prefix and suffix in separate model forwards, unlike
the no-cache path. Misses now use one full-prompt forward, then clone/crop the
retained prefix. This removes redundant forward overhead and aligns miss execution
shapes. Hits still use prefix reuse, so GPU numerical discrepancies remain possible.
FP16 rounding is a hypothesis, not a confirmed root cause from output hashes alone.

The updated notebook starts with 30 requests, one repeat, and `--dtype float32`.
All strategies use that same precision. It isolates its dependencies from Colab's
Gradio/Diffusers environment, stops on test failures, and archives results even if
the benchmark fails. Reload the notebook from GitHub to get these cells.

New reports preserve generated token IDs, hit flags, first divergent token per
request, and completion status. Reports checkpoint atomically after each completed
policy. A runtime terminated mid-policy still loses that policy's unfinished work.
Strict output checks remain enabled. A float32 pass does not prove FP16 correctness;
repeat the same workload in FP16 separately before publishing FP16 measurements.

Local regression tests and tiny-model CPU checks pass. Confirmation on the T4 is
still required; this Mac environment cannot validate the failing CUDA path.

## Extended GPU evaluation notebook

After the short float32 check passes, use
[`gpu_benchmark_suite.ipynb`](notebooks/gpu_benchmark_suite.ipynb).
It runs 150 requests × five strategies × three repeats × three workloads × two
retained-KV budgets (8/32 MiB): 90 runs / 13,500 request executions, four generated
tokens each. This takes substantially longer than the short check. Workload seed
99 is held out from training; repeats measure timing variation, not new samples.
The suite stops on failed correctness and archives completed/partial reports.
Abrupt session termination can still lose un-downloaded files.

```bash
python -m kv_cache.suite --output results/gpu-suite --device cuda
```

Use a new output directory for each suite. Archives are refreshed after every
workload/budget group and in the failure handler. Download the ZIP before a free
runtime expires. No passing GPU result for this extended suite is claimed yet.

### Limited time or no Drive storage

Use [`gpu_benchmark_groups.ipynb`](notebooks/gpu_benchmark_groups.ipynb) to run
**one selected group per session**, with live progress and a ZIP download at the
end. Select groups 1–6 in turn; Run all executes only the selected group. Keep
completed ZIPs on your laptop and select the next unfinished group later.
No Drive storage is used. An unfinished group can still be lost on runtime
termination, so confirm each download before closing your laptop. This uses the
same 150-request / 3-repeat / float32 protocol as the full suite.

## Saved Colab GPU results

[All downloaded October 13 Colab results](benchmarks/colab-2026-10-13/README.md)
are now preserved, including original ZIPs, extracted reports, checksums, and a
CSV summary. The extended T4 float32 suite completed all 90 runs with matching
outputs. Learned eviction sometimes improved reuse but did not demonstrate a
performance advantage. The earlier FP16 failure remains preserved separately.
