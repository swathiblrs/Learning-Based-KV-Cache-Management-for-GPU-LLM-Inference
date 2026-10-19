import importlib.util
import tempfile
import threading
import unittest
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(m) for m in ("torch","transformers","fastapi","httpx","prometheus_client"))


@unittest.skipUnless(AVAILABLE, "Install .[serving]")
class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_all_policies_and_copy_modes_preserve_outputs_under_eviction(self):
        import numpy as np
        from kv_cache.model import Predictor
        from kv_cache.serving import Engine, Metrics, Settings
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"predictor.json")
            Predictor(np.zeros(4),np.ones(4),np.zeros(4),0).save(path)
            reference = None
            for policy in ("none","fifo","lru","lfu","learned"):
                for mode in ("baseline","sliced"):
                    engine = Engine(Settings(tiny=True,policy=policy,copy_mode=mode,budget_bytes=2048,predictor=path),Metrics())
                    rows = [engine.generate([k]*8,[7,8],3) for k in (3,3,4,4,5,3)]
                    outputs = [r["token_ids"] for r in rows]
                    if reference is None:
                        reference = outputs
                    self.assertEqual(outputs,reference)
                    self.assertLessEqual(engine.cache.used,2048)
                    if policy!="none":
                        self.assertTrue(rows[1]["cache_hit"])
                        self.assertGreater(engine.cache.evictions,0)
                        self.assertEqual(engine.cache.entries[next(iter(engine.cache.entries))].payload.get_seq_length(),8)

    def test_sliced_copy_has_independent_storage(self):
        import torch
        from kv_cache.inference import load_model,cache_bytes
        from kv_cache.serving import clone_prefix
        model = load_model("","cpu",tiny=True)
        with torch.inference_mode():
            original = model(torch.tensor([[3]*16]),use_cache=True).past_key_values
            result = clone_prefix(original,8,"sliced")
            self.assertEqual(cache_bytes(result)*2,cache_bytes(original))
            for a,b in zip(original.layers,result.layers):
                torch.testing.assert_close(a.keys[...,:8,:],b.keys)
                self.assertNotEqual(a.keys.data_ptr(),b.keys.data_ptr())
                b.keys.zero_()
            self.assertGreater(original.layers[0].keys.abs().sum().item(),0)

    def test_http_validation_and_metrics(self):
        from fastapi.testclient import TestClient
        from kv_cache.serving import create_app, Settings
        with TestClient(create_app(Settings(tiny=True,budget_bytes=4096))) as client:
            self.assertEqual(client.get('/readyz').status_code,200)
            self.assertEqual(client.get('/info').json()['dtype'],'float32')
            body=dict(prefix_ids=[3,4,5],suffix_ids=[6,7],new_tokens=2)
            a=client.post('/generate',json=body)
            b=client.post('/generate',json=body)
            self.assertEqual(a.status_code,200,a.text)
            self.assertEqual(a.json()['token_ids'],b.json()['token_ids'])
            self.assertTrue(b.json()['cache_hit'])
            for field,value in [('prefix_ids',[-1]),('prefix_ids',[128]),('prefix_ids',[]),('prefix_ids',[True]),('new_tokens',65)]:
                self.assertEqual(client.post('/generate',json={**body,field:value}).status_code,422)
            metrics=client.get('/metrics').text
            self.assertIn('kv_requests_total{outcome="success"} 2.0',metrics)
            self.assertIn('kv_cache_lookups_total{result="hit"} 1.0',metrics)
            self.assertIn('kv_stage_seconds_count{stage="retain_copy"}',metrics)

    def test_metadata_limit_and_context_limit(self):
        from kv_cache.serving import Engine,Metrics,Settings
        engine=Engine(Settings(tiny=True,max_prefixes=1),Metrics())
        engine.generate([3],[4],1)
        with self.assertRaises(OverflowError):
            engine.generate([5],[4],1)
        with self.assertRaises(ValueError):
            engine.generate([3]*2048,[4],1)

    def test_queue_rejection_and_failure_readiness(self):
        from concurrent.futures import ThreadPoolExecutor
        from fastapi.testclient import TestClient
        from kv_cache.serving import create_app,Settings
        entered,release=threading.Event(),threading.Event()
        class Blocking:
            version={}
            def __init__(self,*args): pass
            def generate(self,*args):
                entered.set()
                release.wait(10)
                raise RuntimeError('simulated device failure')
        with TestClient(create_app(Settings(max_pending=1),Blocking)) as client:
            with ThreadPoolExecutor() as pool:
                body=dict(prefix_ids=[3],suffix_ids=[4])
                pending=pool.submit(client.post,'/generate',json=body)
                self.assertTrue(entered.wait(5))
                self.assertEqual(client.post('/generate',json=body).status_code,503)
                release.set()
                self.assertEqual(pending.result().status_code,500)
                self.assertEqual(client.get('/readyz').status_code,503)
                self.assertEqual(client.get('/healthz').status_code,200)
