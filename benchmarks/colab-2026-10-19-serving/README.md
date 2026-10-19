# GPU HTTP serving smoke check

Source: `kv-serving-20261019-003422-632047.zip`; preserved unmodified as `original.zip`.

All 150 request executions across five configurations matched no-cache output IDs. SmolLM2-135M ran in float32 on CUDA, with concurrency 1, one repeat, hot workload seed 99, 32 MiB retained KV budget, and four output tokens. Hardware model was not recorded in the service report; CUDA is verified, an NVIDIA T4 designation relies on the user's selected runtime.

Learned caching hit 14/30 requests (46.7%) versus LRU 12/30 (40.0%). These are separate from the earlier 90-run offline results. Learned retention copying averaged 5.764 ms with the original copy path and 2.873 ms with sliced copying. End-to-end p95 remained above the no-cache baseline. This is one short instrumented smoke check, not evidence of general speedup.

Concurrency-4 GPU validation remains pending. Docker/Grafana runtime validation remains pending. Prometheus output was captured from the live inference service.
