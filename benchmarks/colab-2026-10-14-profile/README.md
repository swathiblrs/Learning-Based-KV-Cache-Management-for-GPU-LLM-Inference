# NVIDIA T4 stage profiling

Original downloaded archive from October 14, 2026 (America/New_York). Its filename
uses UTC, October 15. `original.zip` is unchanged; `archive-index.json` records its
checksum. `code-commit.txt`, `profile.json`, and `predictor.json` are exact extracted
artifacts. Hardware strings inside raw reports are intentionally unchanged.

All five strategies completed 60 requests each in float32 with matching outputs.
The workload was hot-prefix traffic, seed 99, 32 MiB retained cache, four generated
tokens, one repeat. The frozen predictor hash differs from the Colab-trained
extended suite's artifact; do not treat the two runs as identical model artifacts.

Learned-policy measurements: features/prediction ran 13 times (2.30 ms total),
hit copying ran 28 times (144.09 ms total), and retaining prefixes ran 32 times
(193.98 ms total). Learned had 28/60 hits; LRU had 29/60.

These are synchronized host wall-clock diagnostics, including framework overhead.
Prediction is nested inside eviction time. Synchronization perturbs execution;
do not use these timings to claim uninstrumented speedups. Copying is a worthwhile
optimization target, not a proven explanation of every performance difference.
