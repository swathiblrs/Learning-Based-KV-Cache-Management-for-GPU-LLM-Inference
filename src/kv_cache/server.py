"""Entrypoint compatible with hosts that supply PORT; always one model worker."""
import os
import uvicorn


def main():
    uvicorn.run('kv_cache.serving:create_app',factory=True,
                host=os.getenv('HOST','0.0.0.0'),port=int(os.getenv('PORT','8000')),
                workers=1,log_level='info',proxy_headers=False)


if __name__=='__main__':
    main()
