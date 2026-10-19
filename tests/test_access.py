import importlib.util
import unittest

AVAILABLE = all(importlib.util.find_spec(m) for m in ('fastapi','httpx','prometheus_client'))


@unittest.skipUnless(AVAILABLE,'Install serving dependencies')
class AccessTests(unittest.TestCase):
    def test_authentication_and_body_limit(self):
        from fastapi.testclient import TestClient
        from kv_cache.serving import Settings,create_app
        class FakeEngine:
            version={'model':'test'}
            def __init__(self,*args): pass
            def generate(self,*args): return {'token_ids':[1]}
        settings=Settings(api_key='x'*32,require_auth=True,max_body_bytes=128)
        self.assertNotIn('x'*32,repr(settings))
        with TestClient(create_app(settings,FakeEngine)) as client:
            self.assertEqual(client.get('/healthz').status_code,200)
            self.assertEqual(client.get('/readyz').status_code,200)
            for route in ('/info','/metrics','/docs'):
                self.assertEqual(client.get(route).status_code,401)
                self.assertEqual(client.get(route,headers={'Authorization':'Bearer wrong'}).status_code,401)
                self.assertEqual(client.get(route,headers={'Authorization':'Bearer '+'x'*32}).status_code,200)
            headers={'Authorization':'Bearer '+'x'*32,'Content-Type':'application/json'}
            self.assertEqual(client.post('/generate',headers=headers,json={'prefix_ids':[3],'suffix_ids':[4]}).status_code,200)
            self.assertEqual(client.post('/generate',headers=headers,content=b'x'*129).status_code,413)
            self.assertEqual(client.post('/generate',headers=headers,content=iter([b'x'*64,b'x'*65])).status_code,413)

    def test_deployment_fails_closed_without_secret(self):
        from kv_cache.serving import Settings
        with self.assertRaises(ValueError): Settings(require_auth=True)
        with self.assertRaises(ValueError): Settings(require_auth=True,api_key='short')
