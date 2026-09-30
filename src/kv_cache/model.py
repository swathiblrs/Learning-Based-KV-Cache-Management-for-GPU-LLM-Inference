"""A small, inspectable logistic reuse predictor; JSON artifacts, no pickle."""
from bisect import bisect_right
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

FEATURES = ["log_recency", "log_frequency", "log_prefix_tokens", "log_mean_gap"]


@dataclass
class History:
    first: int
    last: int
    count: int
    tokens: int


def observe(history, request, step):
    old = history.get(request.prefix)
    history[request.prefix] = History(old.first if old else step, step,
                                     old.count + 1 if old else 1, request.tokens)


def features(h, step):
    gap = (h.last - h.first) / (h.count - 1) if h.count > 1 else step + 1
    return np.log1p([step - h.last, h.count, h.tokens, gap])


def examples(trace, horizon=32, candidates=8, seed=0):
    """Features use history through t; labels inspect (t, t+horizon].

    The incomplete tail is omitted, not mislabeled as no-reuse. Candidate
    sampling is independent of future labels. Full-prefix metadata survives
    eviction in both training and serving; it is CPU metadata, not KV storage.
    """
    if horizon < 1 or len(trace) <= horizon or candidates < 1:
        raise ValueError("Need a positive horizon/candidate count and a trace longer than the horizon")
    positions = {}
    for t, r in enumerate(trace):
        positions.setdefault(r.prefix, []).append(t)
    history, x, y = {}, [], []
    rng = np.random.default_rng(seed)
    for t, r in enumerate(trace):
        observe(history, r, t)
        if t >= len(trace) - horizon:
            break
        keys = list(history)
        for k in rng.choice(keys, min(candidates, len(keys)), replace=False):
            x.append(features(history[k], t))
            future = positions[k]
            j = bisect_right(future, t)
            y.append(int(j < len(future) and future[j] <= t + horizon))
    return np.asarray(x), np.asarray(y)


class Predictor:
    def __init__(self, mean, scale, weights, bias, horizon=32):
        self.mean, self.scale, self.weights = (np.asarray(v, dtype=float) for v in (mean, scale, weights))
        self.bias, self.horizon = float(bias), int(horizon)
        if any(v.shape != (len(FEATURES),) for v in (self.mean, self.scale, self.weights)):
            raise ValueError("Invalid model dimensions")
        if not all(np.isfinite(v).all() for v in (self.mean, self.scale, self.weights)) or not np.isfinite(self.bias) or (self.scale <= 0).any():
            raise ValueError("Invalid model parameters")

    @classmethod
    def fit(cls, x, y, horizon=32, epochs=400, lr=.1, l2=.001):
        if len(np.unique(y)) != 2:
            raise ValueError("Training requires positive and negative examples")
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-8] = 1
        z = (x - mean) / scale
        weights, bias = np.zeros(x.shape[1]), 0.
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-np.clip(z @ weights + bias, -30, 30)))
            error = p - y
            weights -= lr * (z.T @ error / len(y) + l2 * weights)
            bias -= lr * error.mean()
        return cls(mean, scale, weights, bias, horizon)

    def predict(self, x):
        z = (np.asarray(x) - self.mean) / self.scale
        return 1 / (1 + np.exp(-np.clip(z @ self.weights + self.bias, -30, 30)))

    def save(self, path, provenance=None):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(schema=1, features=FEATURES, mean=self.mean.tolist(),
            scale=self.scale.tolist(), weights=self.weights.tolist(), bias=self.bias,
            horizon=self.horizon, provenance=provenance or {}), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        if d.get("schema") != 1 or d.get("features") != FEATURES:
            raise ValueError("Unsupported model schema")
        return cls(d["mean"], d["scale"], d["weights"], d["bias"], d["horizon"])


def evaluate(model, x, y):
    p = model.predict(x)
    pred = p >= .5
    tp = int(((y == 1) & pred).sum())
    precision = tp / max(1, int(pred.sum()))
    recall = tp / max(1, int(y.sum()))
    return dict(examples=len(y), positives=int(y.sum()), positive_rate=float(y.mean()),
                threshold=.5, precision=precision, recall=recall,
                brier_score=float(np.mean((p - y) ** 2)))
