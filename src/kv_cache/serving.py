"""Single-process, single-model local inference service. Not a multi-tenant server."""
import copy
import hashlib
import json
import os
import threading
from importlib.metadata import version
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .cache import PrefixCache
from .inference import cache_bytes, load_model, sync
from .model import Predictor
from .traces import Request


@dataclass(frozen=True)
class Settings:
    model: str = "HuggingFaceTB/SmolLM2-135M"
    device: str = "cpu"
    policy: str = "lru"
    budget_bytes: int = 32 * 1024**2
    predictor: str = ""
    tiny: bool = False
    copy_mode: str = "baseline"
    profile_stages: bool = True
    logits_mode: str = "all"
    warmup_tokens: int = 3
    max_pending: int = 8
    max_prefixes: int = 10000

    def __post_init__(self):
        if self.device not in ("cpu", "cuda") or self.policy not in ("none", "fifo", "lru", "lfu", "learned"):
            raise ValueError("Invalid device/policy")
        if self.copy_mode not in ("baseline", "sliced", "shared") or self.budget_bytes < 0:
            raise ValueError("Invalid copy mode/budget")
        if self.logits_mode not in ("all", "last") or not 1 <= self.warmup_tokens <= 2048:
            raise ValueError("Invalid logits mode or warmup length")
        if self.max_pending < 1 or self.max_prefixes < 1:
            raise ValueError("Limits must be positive")
        if self.policy == "learned" and not self.predictor:
            raise ValueError("Learned serving requires KV_PREDICTOR")

    @classmethod
    def from_env(cls):
        return cls(model=os.getenv("KV_MODEL", cls.model), device=os.getenv("KV_DEVICE", "cpu"),
                   policy=os.getenv("KV_POLICY", "lru"), budget_bytes=int(os.getenv("KV_BUDGET_BYTES", str(32*1024**2))),
                   predictor=os.getenv("KV_PREDICTOR", ""), tiny=os.getenv("KV_TINY", "0") == "1",
                   copy_mode=os.getenv("KV_COPY_MODE", "baseline"),
                   profile_stages=os.getenv("KV_PROFILE_STAGES", "1") == "1",
                   logits_mode=os.getenv("KV_LOGITS_MODE", "all"),
                   warmup_tokens=int(os.getenv("KV_WARMUP_TOKENS", "3")),
                   max_pending=int(os.getenv("KV_MAX_PENDING", "8")),
                   max_prefixes=int(os.getenv("KV_MAX_PREFIXES", "10000")))


def clone_prefix(past, tokens, mode):
    """Sliced candidate copies only retained tokens; baseline preserves old behavior.

    Restricted to full-attention DynamicCache, as enforced at engine startup.
    Both paths own their tensors; decode must never mutate resident tensors.
    """
    if mode == "baseline":
        result = copy.deepcopy(past)
        result.crop(tokens)
        for layer in result.layers:
            layer.keys = layer.keys.clone()
            layer.values = layer.values.clone()
        return result
    if mode == "shared":
        # Reuse only metadata here. Resident prefix tensors still own storage.
        result = fork_prefix(past)
        for layer in result.layers:
            layer.keys = layer.keys[..., :tokens, :].clone()
            layer.values = layer.values[..., :tokens, :].clone()
        return result
    from transformers import DynamicCache
    result = DynamicCache()
    for i, layer in enumerate(past.layers):
        result.update(layer.keys[..., :tokens, :].clone(), layer.values[..., :tokens, :].clone(), i)
    return result


def fork_prefix(past):
    """Request-local wrapper sharing immutable tensors, restricted to DynamicLayer.

    Transformers 4.57 DynamicLayer.update replaces tensors with torch.cat; it
    does not write into the previous tensor storage. Only full-attention Llama
    is supported for this opt-in path. No crop/reset/in-place writes may run on
    these wrappers. Retention still owns cloned tensors; this is not paged KV.
    """
    from transformers import DynamicCache
    from transformers.cache_utils import DynamicLayer
    if type(past) is not DynamicCache or any(type(layer) is not DynamicLayer for layer in past.layers):
        raise RuntimeError("Shared prefix requires full-attention DynamicCache/DynamicLayer")
    result = copy.copy(past)
    result.layers = [copy.copy(layer) for layer in past.layers]
    return result


class Metrics:
    def __init__(self):
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
        self.registry = CollectorRegistry()
        def counter(name, help, labels=()):
            return Counter("kv_"+name, help, labels, registry=self.registry)
        def gauge(name, help):
            return Gauge("kv_"+name, help, registry=self.registry)
        def hist(name, help, labels=()):
            return Histogram("kv_"+name, help, labels, registry=self.registry,
                             buckets=(.0001,.001,.005,.01,.025,.05,.1,.25,.5,1,2,5,10,30,60))
        self.requests = counter("requests", "Generation requests by outcome", ("outcome",))
        self.hits = counter("cache_lookups", "Cache lookup outcomes", ("result",))
        self.evictions = counter("evictions", "Evicted prefixes")
        self.tokens = counter("generated_tokens", "Generated output tokens")
        self.stage = hist("stage_seconds", "Synchronized diagnostic stage time; prediction nested in eviction", ("stage",))
        self.queue = hist("queue_seconds", "Time waiting for model lock")
        self.latency = hist("request_seconds", "Generation handler time including queue; excludes network")
        self.ttft = hist("compute_ttft_seconds", "Time to computed first token; API is nonstreaming")
        self.used = gauge("retained_bytes", "Resident KV tensors only")
        self.budget = gauge("budget_bytes", "Resident KV budget; excludes model and active tensors")
        self.gpu = gauge("gpu_allocated_bytes", "PyTorch current CUDA allocated bytes")
        self.gpu_peak = gauge("gpu_peak_allocated_bytes", "PyTorch lifetime CUDA peak allocated bytes")
        self.pending = gauge("pending_requests", "Admitted queued and executing requests")
        self.history = gauge("history_prefixes", "CPU prefix metadata records")


class Timer:
    def __init__(self, device, metrics, enabled=True):
        self.device, self.metrics, self.enabled = device, metrics, enabled

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        sync(self.device)
        start = perf_counter()
        try:
            yield
        finally:
            sync(self.device)
            self.metrics.stage.labels(name).observe(perf_counter()-start)


class Engine:
    def __init__(self, settings, metrics):
        import torch
        self.settings, self.metrics = settings, metrics
        self.model = load_model(settings.model, settings.device, settings.tiny, "float32")
        if getattr(self.model.config, "sliding_window", None):
            raise ValueError("Only full-attention models supported")
        self.predictor = Predictor.load(settings.predictor) if settings.predictor else None
        if settings.copy_mode == "shared":
            if self.model.config.model_type != "llama" or not version("transformers").startswith("4.57."):
                raise ValueError("Shared prefix path is validated only for Llama with Transformers 4.57.x")
        if settings.logits_mode == "last":
            import inspect
            if "logits_to_keep" not in inspect.signature(self.model.forward).parameters:
                raise ValueError("Model does not support logits_to_keep")
        self.forward_options = {"logits_to_keep": 1} if settings.logits_mode == "last" else {}
        self.timer = Timer(settings.device, metrics, settings.profile_stages)
        self.cache = PrefixCache(settings.budget_bytes, settings.policy, self.predictor, self.timer)
        self.lock, self.step = threading.Lock(), 0
        self.metrics.budget.set(settings.budget_bytes)
        self.version = dict(model="random-tiny-llama" if settings.tiny else settings.model,
            model_revision=getattr(self.model.config, "_commit_hash", None),
            predictor_sha256=hashlib.sha256(Path(settings.predictor).read_bytes()).hexdigest() if settings.predictor else None,
            policy=settings.policy, budget_bytes=settings.budget_bytes, device=settings.device,
            dtype="float32", copy_mode=settings.copy_mode, torch_version=torch.__version__,
            packages={name: version(name) for name in ("transformers", "numpy", "fastapi", "uvicorn", "prometheus-client")},
            timing_is_instrumented=settings.profile_stages, logits_mode=settings.logits_mode,
            warmup_tokens=settings.warmup_tokens,
            hardware=torch.cuda.get_device_name() if settings.device == "cuda" else "CPU",
            timing_note="Request-boundary and first-token synchronization retained; stage timers optional",
            batching=False, model_concurrency=1)
        # Warm model without modifying cache history.
        with torch.inference_mode():
            if settings.warmup_tokens >= self.model.config.max_position_embeddings:
                raise ValueError("Warmup exceeds model context")
            for _ in range(2):
                warm = self.model(torch.full((1,settings.warmup_tokens), 3, device=settings.device),
                                  use_cache=True, **self.forward_options)
                del warm
        sync(settings.device)

    def generate(self, prefix_ids, suffix_ids, new_tokens):
        import torch
        start = perf_counter()
        with self.lock:
            queue = perf_counter()-start
            self.metrics.queue.observe(queue)
            c = self.model.config
            if len(prefix_ids)+len(suffix_ids)+new_tokens > c.max_position_embeddings:
                raise ValueError("Input plus output exceeds context limit")
            if max(prefix_ids+suffix_ids) >= c.vocab_size:
                raise ValueError("Token ID outside model vocabulary")
            key = hashlib.sha256(json.dumps(prefix_ids, separators=(",", ":")).encode()).hexdigest()
            if key not in self.cache.history and len(self.cache.history) >= self.settings.max_prefixes:
                raise OverflowError("Prefix metadata limit reached; restart service for a new session")
            prefix = torch.tensor([prefix_ids], device=self.settings.device)
            suffix = torch.tensor([suffix_ids], device=self.settings.device)
            r = Request(key, len(prefix_ids))
            t = self.step
            self.step += 1
            before = self.cache.evictions
            sync(self.settings.device)
            compute_start = perf_counter()
            with torch.inference_mode():
                with self.timer.stage("lookup"):
                    entry = self.cache.lookup(r, t)
                hit = entry is not None
                self.metrics.hits.labels("hit" if hit else "miss").inc()
                if hit:
                    with self.timer.stage("hit_copy"):
                        past = fork_prefix(entry.payload) if self.settings.copy_mode == "shared" else copy.deepcopy(entry.payload)
                    with self.timer.stage("prefill"):
                        out = self.model(suffix, past_key_values=past, use_cache=True, **self.forward_options)
                else:
                    heads = getattr(c, "num_key_value_heads", c.num_attention_heads)
                    dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
                    size = len(prefix_ids)*2*c.num_hidden_layers*heads*dim*next(self.model.parameters()).element_size()
                    with self.timer.stage("eviction"):
                        retain = self.cache.reserve(size, t)
                    # Cache decision samples are exported to histograms, not kept forever.
                    self.cache.decision_ms.clear()
                    with self.timer.stage("prefill"):
                        out = self.model(torch.cat((prefix, suffix), dim=1), use_cache=True, **self.forward_options)
                    if retain:
                        with self.timer.stage("retain_copy"):
                            retained = clone_prefix(out.past_key_values, len(prefix_ids), self.settings.copy_mode)
                            if cache_bytes(retained) != size:
                                raise RuntimeError("Unsupported KV layout")
                            self.cache.insert(key, size, retained, t)
                token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                generated = [token]
                sync(self.settings.device)
                ttft = perf_counter()-compute_start
                with self.timer.stage("decode"):
                    for _ in range(new_tokens-1):
                        out = self.model(token, past_key_values=out.past_key_values, use_cache=True, **self.forward_options)
                        token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                        generated.append(token)
                ids = torch.cat(generated, dim=1).cpu().tolist()[0]
            self.metrics.evictions.inc(self.cache.evictions-before)
            self.metrics.used.set(self.cache.used)
            self.metrics.history.set(len(self.cache.history))
            self.metrics.tokens.inc(len(ids))
            self.metrics.ttft.observe(ttft)
            if self.settings.device == "cuda":
                self.metrics.gpu.set(torch.cuda.memory_allocated())
                self.metrics.gpu_peak.set(torch.cuda.max_memory_allocated())
            return dict(token_ids=ids, cache_hit=hit, queue_ms=queue*1000,
                        compute_ttft_ms=ttft*1000, retained_bytes=self.cache.used, sequence=t)


def create_app(settings=None, engine_factory=Engine):
    from fastapi import FastAPI, HTTPException, Response
    from pydantic import BaseModel, Field, StrictInt
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    from starlette.concurrency import run_in_threadpool
    settings = settings or Settings.from_env()
    metrics = Metrics()

    class Generation(BaseModel):
        prefix_ids: list[StrictInt] = Field(min_length=1, max_length=2048)
        suffix_ids: list[StrictInt] = Field(min_length=1, max_length=2048)
        new_tokens: int = Field(default=4, ge=1, le=64, strict=True)

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = await run_in_threadpool(engine_factory, settings, metrics)
        app.state.pending = 0
        app.state.failed = False
        yield
        app.state.engine = None

    app = FastAPI(title="KV-cache inference research service", lifespan=lifespan)

    @app.get("/healthz")
    def health():
        return {"status": "alive"}

    @app.get("/readyz")
    def ready():
        if not getattr(app.state, "engine", None) or app.state.failed:
            raise HTTPException(503, "Not ready; restart after inference failure")
        return {"status": "ready"}

    @app.get("/info")
    def info():
        return app.state.engine.version

    @app.get("/metrics")
    def export():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/generate")
    async def generate(body: Generation):
        if min(body.prefix_ids+body.suffix_ids) < 0:
            metrics.requests.labels("invalid").inc()
            raise HTTPException(422, "Token IDs must be nonnegative")
        if app.state.failed or app.state.pending >= settings.max_pending:
            metrics.requests.labels("rejected").inc()
            raise HTTPException(503, "Service unavailable or queue full")
        app.state.pending += 1
        metrics.pending.set(app.state.pending)
        start = perf_counter()
        try:
            result = await run_in_threadpool(app.state.engine.generate, body.prefix_ids, body.suffix_ids, body.new_tokens)
            metrics.requests.labels("success").inc()
            return result
        except ValueError as error:
            metrics.requests.labels("invalid").inc()
            raise HTTPException(422, str(error)) from error
        except OverflowError as error:
            metrics.requests.labels("rejected").inc()
            raise HTTPException(503, str(error)) from error
        except Exception:
            app.state.failed = True
            metrics.requests.labels("error").inc()
            raise HTTPException(500, "Inference failed; restart service")
        finally:
            app.state.pending -= 1
            metrics.pending.set(app.state.pending)
            metrics.latency.observe(perf_counter()-start)
    return app
