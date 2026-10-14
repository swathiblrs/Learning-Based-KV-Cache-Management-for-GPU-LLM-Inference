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

    def test_miss_uses_one_prefill_and_hit_uses_only_suffix(self):
        from kv_cache.inference import benchmark
        lengths = []
        hook = self.model.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
        try:
            row = benchmark(self.model, [Request("a", 8), Request("a", 8)], "lru", 4096,
                            device="cpu", new_tokens=1)
        finally:
            hook.remove()
        self.assertEqual(lengths, [16, 16, 16, 8])  # Two warmups, one miss, one hit.
        self.assertEqual(row["cache_hit_by_request"], [False, True])

    def test_stage_profile_preserves_outputs_and_covers_eviction_and_hits(self):
        from kv_cache.inference import benchmark
        trace = [Request(k, 8) for k in "abacabad"]
        predictor = Predictor(np.zeros(4), np.ones(4), np.zeros(4), 0)
        plain = benchmark(self.model, trace, "learned", 4096, predictor, "cpu", 3)
        profiled = benchmark(self.model, trace, "learned", 4096, predictor, "cpu", 3,
                             profile_stages=True)
        self.assertEqual(plain["output_sha256"], profiled["output_sha256"])
        self.assertIsNone(plain["profiling"])
        self.assertEqual(profiled["kind"], "real_model_profile")
        summary = profiled["profiling"]["summary"]
        self.assertEqual(summary["cache_lookup"]["calls"], len(trace))
        self.assertEqual(summary["hit_cache_copy"]["calls"], profiled["hits"])
        self.assertEqual(summary["full_prefill"]["calls"], len(trace) - profiled["hits"])
        self.assertGreater(summary["prediction"]["calls"], 0)
        self.assertEqual(summary["decode"]["calls"], len(trace))
        self.assertEqual(len(profiled["profiling"]["requests"]), len(trace))
        for item in summary.values():
            self.assertGreaterEqual(item["total_ms"], 0)

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


class OutputDiagnosticsTests(unittest.TestCase):
    def test_reports_first_divergence_without_accepting_it(self):
        from kv_cache.inference import compare_outputs
        rows = [dict(policy="none", repeat=0, generated_token_ids=[[1, 2], [3, 4]]),
                dict(policy="lru", repeat=0, generated_token_ids=[[1, 2], [3, 5]])]
        result = compare_outputs(rows)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["mismatches"][0]["request_index"], 1)
        self.assertEqual(result["mismatches"][0]["first_different_token"], 1)

    def test_pending_until_reference_and_repeat_mismatch(self):
        from kv_cache.inference import compare_outputs
        self.assertEqual(compare_outputs([])["status"], "pending")
        rows = [dict(policy="none", repeat=0, generated_token_ids=[[1]]),
                dict(policy="none", repeat=1, generated_token_ids=[[2]])]
        self.assertEqual(compare_outputs(rows)["status"], "failed")

    def test_request_count_mismatch_rejected(self):
        from kv_cache.inference import compare_outputs
        with self.assertRaises(ValueError):
            compare_outputs([dict(policy="none", repeat=0, generated_token_ids=[[1]]),
                             dict(policy="lru", repeat=0, generated_token_ids=[])])

if __name__ == "__main__":
    unittest.main()
