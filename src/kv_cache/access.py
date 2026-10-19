"""Small ASGI access boundary for a single-owner deployment."""
import hmac


class AccessBoundary:
    def __init__(self, app, api_key="", max_body_bytes=65536):
        self.app, self.api_key, self.max_body_bytes = app, api_key, max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        async def reject(status, message):
            body = ('{"detail":"'+message+'"}').encode()
            await send({'type':'http.response.start','status':status,
                        'headers':[(b'content-type',b'application/json'),(b'content-length',str(len(body)).encode())]})
            await send({'type':'http.response.body','body':body})
        # Health endpoints reveal no configuration and must work for platform probes.
        public = scope['path'] in ('/healthz','/readyz') and scope['method'] in ('GET','HEAD')
        if self.api_key and not public:
            headers = dict(scope.get('headers',[]))
            expected = ('Bearer '+self.api_key).encode()
            if not hmac.compare_digest(headers.get(b'authorization',b''),expected):
                return await reject(401,'Valid bearer token required')
        # Read with a strict bound even when Content-Length is absent/chunked.
        body = bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            chunk = message.get('body',b'')
            if len(body)+len(chunk)>self.max_body_bytes:
                return await reject(413,'Request body too large')
            body.extend(chunk)
            if not message.get('more_body',False):
                break
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type':'http.request','body':bytes(body),'more_body':False}
            return await receive()
        await self.app(scope,replay,send)
