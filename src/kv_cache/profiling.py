"""Synchronized wall-clock stage diagnostics, separate from performance claims."""
from contextlib import contextmanager
from time import perf_counter
import numpy as np


class StageProfiler:
    def __init__(self, synchronize):
        self.synchronize = synchronize
        self.requests = []
        self.current = None

    def begin(self, index):
        self.current = dict(request_index=index, cache_hit=False, stages_ms={})
        self.requests.append(self.current)

    @contextmanager
    def stage(self, name):
        self.synchronize()
        start = perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            elapsed = (perf_counter() - start) * 1000
            stages = self.current["stages_ms"]
            stages[name] = stages.get(name, 0.) + elapsed

    def report(self):
        summary = {}
        names = sorted({k for r in self.requests for k in r["stages_ms"]})
        for name in names:
            values = [r["stages_ms"][name] for r in self.requests if name in r["stages_ms"]]
            summary[name] = dict(calls=len(values), total_ms=sum(values),
                                 mean_ms=float(np.mean(values)), p95_ms=float(np.percentile(values, 95)))
        return dict(method="synchronized host wall time", timing_is_instrumented=True,
                    note="Stage synchronization changes runtime. prediction is nested inside eviction_decision; do not sum both. Absent stages are not called, not zero-cost observations.",
                    summary=summary, requests=self.requests)
