# Experiment protocol

## Before benchmarking

1. Train once and freeze the JSON predictor. Do not tune against held-out results.
2. Keep a held-out trace and record its origin. Default traces are synthetic;
   JSONL input uses `{"prefix": "stable-id", "tokens": 128}` per request.
   A prefix ID identifies a whole identical token sequence, not semantic similarity.
3. For imported real traces, document authorization and remove sensitive content.
   The current inference harness maps IDs to deterministic synthetic token IDs;
   even an imported access trace does not replay its original text.
4. Choose a cache budget smaller than the working set, but leave device headroom
   for weights, active requests, cloned KV, and temporary tensors.

## Metrics and timing boundaries

- Cache hit rate: requests reusing a whole retained prefix / total requests.
- Token reuse rate: prefix tokens reused / total requested prefix tokens.
- TTFT: lookup, eviction, cloning, prefill, suffix processing and first greedy
  token selection. Input preparation and model loading are outside the boundary.
- Request latency: the same start through fixed-length decode, synchronized.
- Measured throughput: requests/output tokens divided by summed request times.
  This excludes inter-request harness/reporting work and is **not** concurrent
  server capacity or wall-clock end-to-end application throughput.
- Resident peak bytes: exact retained KV tensor bytes only.
- CUDA peak allocated bytes: includes weights, active KV, resident KV, and
  temporaries tracked by PyTorch; not allocator-reserved or total GPU memory.
- Decision p95: reserve/eviction host time on admissible misses, including cheap
  non-evicting reservations; it does not include cloning or all lookup overhead.

All policies use the same weights, dtype, input trace, budget, and decode length.
No-cache prefill uses a single full prompt. Cache misses process prefix and suffix
separately to obtain reusable state, so cache policies pay that extra overhead.
Fixed-length decoding deliberately ignores EOS to keep work comparable; it is
not representative of real output-length distributions.

## Repeat and interpret

Run multiple seeds and budgets. The CLI repeats real-model runs three times and
randomizes policy order. Report distributions across repeats, not the best run.
Do not average p95 values and call that a pooled p95. Keep per-run values visible.

Test stable popularity, popularity shifts, and scans. Include negative results.
A model trained on one synthetic generator may learn that generator rather than
real behavior. Future work should add real anonymized traces, gradient-boosted
baselines, feature ablations, calibration analysis, and bounded metadata.

If creating chronological splits in one trace, purge each training/validation
boundary by at least the label horizon so labels cannot see the next split.
The default experiment uses independent traces instead.

## Correctness and safety of reuse

Whole prefixes are separately stored, and their bytes are measured from tensors.
A deep copy is made before a retained cache is extended. This costs time and
memory, both of which matter. There is no shared paged-memory graph here.
Requests are sequential; a retained entry is never an active request's sole KV.

CPU tiny-model tests compare full-prefill and reused-prefix logits, then compare
all policies' generated outputs under eviction. GPU runs independently compare
output hashes. Exact token equality can fail near argmax ties due to floating
point differences; investigate instead of hiding the mismatch.

## Zero-cost workflow

Develop/train on the laptop. Use `--device mps` optionally for Apple GPU runs;
label those as Apple GPU results, never CUDA results. Run NVIDIA measurements
only when a free authorized GPU session is available. Save artifacts before the
session ends. No cloud account provisioning, paid service, or API key is required.
