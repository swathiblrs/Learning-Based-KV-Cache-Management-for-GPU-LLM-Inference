import tempfile
import unittest
from pathlib import Path
import numpy as np
from kv_cache.cache import PrefixCache
from kv_cache.model import History, Predictor, examples, features
from kv_cache.simulate import simulate
from kv_cache.traces import Request, generate, read_trace, write_trace


class CacheTests(unittest.TestCase):
    def replay(self, policy, keys):
        c = PrefixCache(2, policy)
        for t, k in enumerate(keys):
            if c.lookup(Request(k, 1), t) is None and c.reserve(1, t):
                c.insert(k, 1, None, t)
            self.assertLessEqual(c.used, c.budget)
        return c

    def test_lru(self):
        self.assertEqual(set(self.replay("lru", "abac").entries), {"a", "c"})

    def test_fifo(self):
        self.assertEqual(set(self.replay("fifo", "abac").entries), {"b", "c"})

    def test_lfu(self):
        self.assertEqual(set(self.replay("lfu", "aabbac").entries), {"a", "c"})

    def test_oversized_does_not_evict(self):
        c = self.replay("lru", "ab")
        self.assertFalse(c.reserve(3, 2))
        self.assertEqual(set(c.entries), {"a", "b"})

    def test_multiple_evictions_and_zero_budget(self):
        c = self.replay("lru", "ab")
        self.assertTrue(c.reserve(2, 2))
        self.assertEqual(c.used, 0)
        self.assertEqual(c.evictions, 2)
        self.assertFalse(PrefixCache(0).reserve(1, 0))

    def test_learned_evicts_low_value(self):
        class Stub:
            def predict(self, x):
                return np.array([.1, .9])
        c = PrefixCache(2, "learned", Stub())
        for t, key in enumerate("ab"):
            c.lookup(Request(key, 1), t)
            c.insert(key, 1, None, t)
        c.reserve(1, 2)
        self.assertEqual(set(c.entries), {"b"})

    def test_known_simulation(self):
        trace = [Request(k, 1) for k in "abac"]
        row = simulate(trace, 2, "lru", bytes_per_token=1)
        self.assertEqual(row["hits"], 1)
        self.assertEqual(row["reused_prefix_tokens"], 1)
        self.assertEqual(row["retained_peak_bytes"], 2)


class ModelTests(unittest.TestCase):
    def test_future_labels_and_censored_tail(self):
        x, y = examples([Request(k, 1) for k in "aba"], horizon=2)
        self.assertEqual(len(x), 1)
        self.assertEqual(y.tolist(), [1])
        _, y = examples([Request(k, 1) for k in "abc"], horizon=2)
        self.assertEqual(y.tolist(), [0])

    def test_future_does_not_change_features(self):
        a, _ = examples([Request(k, 1) for k in "abac"], horizon=2)
        b, _ = examples([Request(k, 1) for k in "abzz"], horizon=2)
        np.testing.assert_array_equal(a, b)

    def test_feature_values(self):
        np.testing.assert_allclose(features(History(0, 2, 2, 8), 4), np.log1p([2, 2, 8, 2]))

    def test_fit_and_json_roundtrip(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(200, 4))
        y = (x[:, 0] > 0).astype(int)
        m = Predictor.fit(x, y)
        self.assertGreater(np.mean((m.predict(x) >= .5) == y), .9)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            m.save(p)
            np.testing.assert_allclose(m.predict(x), Predictor.load(p).predict(x))

    def test_trace_roundtrip_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "trace.jsonl"
            trace = generate(10)
            write_trace(p, trace)
            self.assertEqual(read_trace(p), trace)
            with self.assertRaises(ValueError):
                write_trace(p, [Request("a", 1), Request("a", 2)])


if __name__ == "__main__":
    unittest.main()
