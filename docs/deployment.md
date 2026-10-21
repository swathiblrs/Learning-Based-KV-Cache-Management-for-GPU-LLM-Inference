# Deployment runbook and evidence requirements

Status: the CPU container build and authenticated tiny-model smoke test passed GitHub Actions on source commit `7d0eed3`; [verified CI evidence](validation/container-ci.json). No external host has been selected or deployed. The long-prefix GPU experiment is also pending. Neither prepared files nor a temporary Colab run establish production deployment.

## Existing Linux host

The host needs Docker/Compose, sufficient CPU/RAM/disk, and network access to download model weights on first startup. The CPU configuration runs real SmolLM2-135M, not the tiny random model. GPU performance claims cannot be inferred from CPU deployment.

Create a local API secret without printing it:

```sh
mkdir -p deploy/secrets
python3 -c 'import secrets; from pathlib import Path; p=Path("deploy/secrets/api_key"); p.write_text(secrets.token_urlsafe(32)); p.chmod(0o644)'
chmod 700 deploy/secrets
docker compose -f deploy/compose.server.yaml up --build -d
```

The parent directory is private; the file is readable inside the non-root container through a Compose secret mount. Never commit it. Local Compose secrets are files, not an encrypted secret store. A managed host should inject `KV_API_KEY` through its own secret manager instead. The container refuses to start without an adequate API key when `KV_REQUIRE_AUTH=1`.

All API routes, metrics and docs require `Authorization: Bearer ...` except health/readiness. Requests are capped at 64 KiB including chunked bodies. One worker serializes model access, with a bounded request queue and a bounded count of unique prefix metadata records. The API remains pre-tokenized, fixed-length greedy generation. No multi-tenant cache isolation is provided; deploy only as a single-owner service.

The default server Compose binds only to `127.0.0.1:8000`. Access it through an SSH tunnel from your own machine or a host-provided HTTPS reverse proxy. Do not publish plain HTTP with a bearer key. The deployment platform must supply TLS if remote public access is needed.

## Verify on the selected host

```sh
export KV_API_KEY="$(cat deploy/secrets/api_key)"
python3 scripts/check_deployment.py http://127.0.0.1:8000
unset KV_API_KEY
```

For managed HTTPS hosting, use its URL and set the same secret through your shell's secure environment tooling. Do not paste keys into chat. The checker verifies unauthenticated access is denied, readiness succeeds, repeated generation is stable, prefix reuse is observed, and metrics exist. It writes a report without the secret. This does not replace a load test or a service availability observation period.

Persist the deployment report alongside: image digest, source commit, model revision, predictor SHA256, runtime package list, hosting target, and timestamp. `/info` exposes model/predictor provenance; `/app/runtime-packages.txt` records installed container packages. The image pins direct runtime versions and the model commit. Base image/transitive dependencies are not a complete immutable lock, so retain the built image digest rather than assuming a later rebuild is identical.

## Monitoring and operation

For a protected service, a Prometheus scraper must send the same bearer key (e.g. `authorization.credentials_file` mounted from the secret), and use a trusted network. The existing unauthenticated localhost monitoring demo is separate; it must not be mistaken for production monitoring deployment.

Readiness is false after an inference failure. Docker restart policy handles process exits, not merely an unhealthy check; investigate logs and restart the service after such failures. The prefix-history cap returns 503 for new identities once full; restart starts a fresh session. A long-lived unbounded production workload needs a metadata-aging strategy and further evaluation, not a claim that this research cache is already general-purpose serving infrastructure.

A host-provided persistent volume caches downloaded weights. The inference cache is intentionally lost at restart. Retained KV budget excludes active tensors, model weights and allocator memory. Back up configuration/provenance, not GPU KV tensors.

To stop: `docker compose -f deploy/compose.server.yaml down`. To roll back: select the previous tested source/image digest and restart, then rerun the deployment checker. No billing account, paid instance or public endpoint is created by these files.

## Automated container check

The `container-smoke` GitHub Actions job builds `Dockerfile.cpu`, starts an authenticated tiny-model container and runs the deployment checker. It saves logs and runtime evidence. A passing job verifies the container packaging and API boundary only; the tiny-model CI job does not validate real-model hosting or NVIDIA performance. Job execution/results must be checked after push.

Verified container run: [GitHub Actions 39283544724](https://github.com/swathiblrs/Learning-Based-KV-Cache-Management-for-GPU-LLM-Inference/actions/runs/39283544724). All three jobs passed. This closes container-runtime validation; it does not establish a persistent deployed service.
