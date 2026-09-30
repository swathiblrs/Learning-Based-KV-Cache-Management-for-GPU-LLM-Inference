from time import perf_counter
import numpy as np
from .cache import PrefixCache


def simulate(trace, budget, policy, predictor=None, bytes_per_token=1024):
    if bytes_per_token < 1:
        raise ValueError("bytes_per_token must be positive")
    cache = PrefixCache(budget, policy, predictor)
    hits = saved = total = 0
    start = perf_counter()
    for t, r in enumerate(trace):
        total += r.tokens
        if cache.lookup(r, t) is not None:
            hits += 1
            saved += r.tokens
        else:
            size = r.tokens * bytes_per_token
            if cache.reserve(size, t):
                cache.insert(r.prefix, size, None, t)
    return dict(kind="simulation", policy=policy, requests=len(trace), hits=hits,
                hit_rate=hits / len(trace), reused_prefix_tokens=saved,
                prefix_token_reuse_rate=saved / total, retained_peak_bytes=cache.peak,
                cache_budget_bytes=budget, bytes_per_token=bytes_per_token,
                evictions=cache.evictions,
                policy_decision_p95_ms=float(np.percentile(cache.decision_ms, 95)) if cache.decision_ms else 0,
                simulator_wall_seconds=perf_counter() - start)
