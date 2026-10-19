"""Check a user's deployment; never print the API key. Token comes from env."""
import argparse
import json
import os
from pathlib import Path
import urllib.request
import urllib.error


def check(url, token, allow_http=False):
    from urllib.parse import urlparse
    parsed= urlparse(url)
    if parsed.scheme != 'https' and not (parsed.scheme=='http' and (parsed.hostname in ('127.0.0.1','localhost') or allow_http)):
        raise ValueError('Use HTTPS for remote deployment; HTTP loopback is supported')
    if not token:
        raise ValueError('Set KV_API_KEY; do not pass secrets on the command line')
    url=url.rstrip('/')
    def request(path, body=None, auth=True):
        headers={'Authorization':'Bearer '+token} if auth else {}
        if body is not None: headers['Content-Type']='application/json'
        req=urllib.request.Request(url+path,data=json.dumps(body).encode() if body else None,headers=headers)
        with urllib.request.urlopen(req,timeout=120) as response:
            return response.read()
    request('/readyz',auth=False)
    try:
        request('/info',auth=False)
    except urllib.error.HTTPError as error:
        if error.code != 401: raise
    else:
        raise RuntimeError('Deployment did not reject unauthenticated configuration access')
    info=json.loads(request('/info'))
    body={'prefix_ids':[3,4,5,6], 'suffix_ids':[7,8], 'new_tokens':4}
    first=json.loads(request('/generate',body))
    second=json.loads(request('/generate',body))
    if first['token_ids']!=second['token_ids']:
        raise RuntimeError('Repeated cached request changed output')
    if info['policy']!='none' and info['budget_bytes']>=len(body['prefix_ids'])*46080 and not second['cache_hit']:
        raise RuntimeError('Expected reuse was not observed')
    metrics=request('/metrics').decode()
    if 'kv_requests_total' not in metrics: raise RuntimeError('Missing request metrics')
    return {'status':'passed','url':url,'model_info':info,'repeated_outputs_match':True,
            'second_request_cache_hit':second['cache_hit'],
            'note':'Deployment smoke check only; not sustained load, availability or performance evidence'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('url')
    p.add_argument('--output',default='results/deployment-check.json')
    args=p.parse_args()
    result=check(args.url,os.getenv('KV_API_KEY',''))
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print('Deployment smoke check passed; report:',target)
