"""Whole-prefix cache. Resident bytes exclude model weights and active state."""
from dataclasses import dataclass
from time import perf_counter
from .model import features, observe


@dataclass
class Entry:
    size: int
    payload: object
    inserted: int


class PrefixCache:
    def __init__(self, budget, policy="lru", predictor=None):
        if budget < 0 or policy not in ("none", "fifo", "lru", "lfu", "learned"):
            raise ValueError("Invalid budget or policy")
        if policy == "learned" and predictor is None:
            raise ValueError("Learned policy requires a predictor")
        self.budget, self.policy, self.predictor = budget, policy, predictor
        self.entries, self.history = {}, {}
        self.used = self.peak = self.evictions = 0
        self.decision_ms = []

    def lookup(self, request, step):
        observe(self.history, request, step)
        return self.entries.get(request.prefix)

    def reserve(self, size, step):
        """Evict before allocating the new retained payload. Always-admit policy."""
        if size <= 0:
            raise ValueError("Payload size must be positive")
        if self.policy == "none" or size > self.budget:
            return False
        start = perf_counter()
        if self.used + size > self.budget:
            keys = list(self.entries)
            if self.policy == "learned":
                probabilities = self.predictor.predict([features(self.history[k], step) for k in keys])
                # Approximate prefill work saved per resident byte, not measured FLOPs.
                scores = {k: float(p) * self.history[k].tokens / self.entries[k].size
                          for k, p in zip(keys, probabilities)}
                order = sorted(keys, key=lambda k: (scores[k], self.history[k].last, k))
            elif self.policy == "lfu":
                order = sorted(keys, key=lambda k: (self.history[k].count, self.history[k].last, k))
            elif self.policy == "fifo":
                order = sorted(keys, key=lambda k: (self.entries[k].inserted, k))
            else:
                order = sorted(keys, key=lambda k: (self.history[k].last, k))
            for k in order:
                self.used -= self.entries.pop(k).size
                self.evictions += 1
                if self.used + size <= self.budget:
                    break
        self.decision_ms.append((perf_counter() - start) * 1000)
        return True

    def insert(self, key, size, payload, step):
        if key in self.entries or size <= 0 or self.used + size > self.budget:
            raise ValueError("Duplicate, invalid, or unreserved payload")
        self.entries[key] = Entry(size, payload, step)
        self.used += size
        self.peak = max(self.peak, self.used)
