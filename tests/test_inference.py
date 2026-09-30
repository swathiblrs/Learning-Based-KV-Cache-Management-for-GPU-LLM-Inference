import copy
import importlib.util
import unittest
import numpy as np
from kv_cache.model import Predictor
from kv_cache.traces import Request

AVAILABLE = all(importlib.util.find_spec(m) for m in ("torch", "transformers"))


@unittest.skipUnless(AVAILABLE, "Install .[inference] for actual KV correctness tests")
class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from kv_cache.inference import load_model
        torch.set_num_threads(1)
        cls.model = load_model("", "cpu", tiny=True)

    def test_reuse_matches_full_logits_and_does_not_mutate_prefix(self):
        import torch
        from kv_cache.inference import cache_bytes
        ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
        with torch.inference_mode():
            full = self.model(ids).logits[:, -1]
            pref = self.model(ids[:, :4], use_cache=True).past_key_values
            before = cache_bytes(pref)
            reused = self.model(ids[:, 4:], past_key_values=copy.deepcopy(pref)).logits[:, -1]
        torch.testing.assert_close(full, reused, atol=1e-5, rtol=1e-4)
        self.assertEqual(pref.get_seq_length(), 4)
        self.assertEqual(cache_bytes(pref), before)

    def test_policies_preserve_outputs_and_budget(self):
        from kv_cache.inference import benchmark
        trace = [Request(k, 8) for k in "abacabad"]
        predictor = Predictor(np.zeros(4), np.ones(4), np.zeros(4), 0)
        rows = [benchmark(self.model, trace, p, 4096, predictor, "cpu", 3)
                for p in ("none", "lru", "lfu", "fifo", "learned")]
        self.assertEqual(len({r["output_sha256"] for r in rows}), 1)
        self.assertGreater(rows[1]["hits"], 0)
        self.assertGreater(rows[1]["evictions"], 0)
        for r in rows:
            self.assertLessEqual(r["retained_peak_bytes"], 4096)


if __name__ == "__main__":
    unittest.main()
