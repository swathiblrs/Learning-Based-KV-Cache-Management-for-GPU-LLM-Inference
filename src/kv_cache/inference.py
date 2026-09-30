"""Sequential Hugging Face inference benchmark with real, immutable prefix KV.

Synthetic token IDs isolate cache behavior; this is not an answer-quality test.
Whole prefixes are independent objects, so no shared-page dependency is evicted.
"""
import copy
import hashlib
import platform
from time import perf_counter
import numpy as np
from .cache import PrefixCache


def sync(device):
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def token_ids(request, vocab_size):
    import torch
    seed = int.from_bytes(hashlib.sha256(request.prefix.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.integers(3, vocab_size, size=(1, request.tokens)), dtype=torch.long)


def cache_bytes(past):
    return sum(t.numel() * t.element_size() for layer in past.layers for t in (layer.keys, layer.values))


def load_model(name, device, tiny=False):
    import torch
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use a free NVIDIA GPU notebook, or --device cpu --tiny for correctness only.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    torch.manual_seed(0)
    if tiny:
        model = LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=32,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=2048))
    else:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                                                   attn_implementation="eager")
    return model.to(device).eval()


def benchmark(model, trace, policy, budget, predictor=None, device="cuda", new_tokens=8):
    import torch
    import transformers
    if new_tokens < 1:
        raise ValueError("new_tokens must be positive")
    config = model.config
    if getattr(config, "sliding_window", None):
        raise ValueError("This prototype supports full-attention models only")
    if max(r.tokens for r in trace) + 8 + new_tokens > config.max_position_embeddings:
        raise ValueError("Request exceeds the model context limit")
    heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    bpt = 2 * config.num_hidden_layers * heads * dim * next(model.parameters()).element_size()
    cache = PrefixCache(budget, policy, predictor)
    # Pre-tokenized synthetic requests, same suffix and fixed decode length for all policies.
    inputs = [token_ids(r, config.vocab_size).to(device) for r in trace]
    suffix = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]], device=device)
    ttft, latencies, outputs = [], [], []
    hits = reused = 0
    with torch.inference_mode():
        for _ in range(2):
            warm = model(torch.cat((inputs[0], suffix), dim=1), use_cache=True)
            del warm
        sync(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for t, (r, prefix) in enumerate(zip(trace, inputs)):
            sync(device)
            start = perf_counter()
            entry = cache.lookup(r, t)
            if entry is not None:
                hits += 1
                reused += r.tokens
                past = copy.deepcopy(entry.payload)  # Never mutate resident KV during decoding.
                out = model(suffix, past_key_values=past, use_cache=True)
            elif policy == "none":
                out = model(torch.cat((prefix, suffix), dim=1), use_cache=True)
            else:
                retain = cache.reserve(r.tokens * bpt, t)
                pref = model(prefix, use_cache=True)
                past = pref.past_key_values
                if retain:
                    size = cache_bytes(past)
                    if size != r.tokens * bpt:
                        raise ValueError("Unsupported KV layout: actual bytes differ from reservation")
                    cache.insert(r.prefix, size, copy.deepcopy(past), t)
                del pref
                out = model(suffix, past_key_values=past, use_cache=True)
            token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = [token]
            sync(device)
            ttft.append((perf_counter() - start) * 1000)
            for _ in range(new_tokens - 1):
                out = model(token, past_key_values=out.past_key_values, use_cache=True)
                token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                generated.append(token)
            sync(device)
            latencies.append((perf_counter() - start) * 1000)
            outputs.append(torch.cat(generated, dim=1).cpu().tolist()[0])
            del out, token, generated, entry
            if "past" in locals():
                del past
    total_seconds = sum(latencies) / 1000
    return dict(kind="real_model", device=device, policy=policy, requests=len(trace),
        model_type=config.model_type, random_weights=bool(getattr(model, "_benchmark_tiny", False)),
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        hardware=torch.cuda.get_device_name() if device == "cuda" else platform.machine(),
        dtype=str(next(model.parameters()).dtype), batch_size=1, concurrency=1,
        fixed_output_tokens=new_tokens, hits=hits, hit_rate=hits / len(trace),
        reused_prefix_tokens=reused, prefix_token_reuse_rate=reused / sum(r.tokens for r in trace),
        ttft_p50_ms=float(np.percentile(ttft, 50)), ttft_p95_ms=float(np.percentile(ttft, 95)),
        request_p95_ms=float(np.percentile(latencies, 95)),
        measured_requests_per_second=len(trace) / total_seconds,
        measured_output_tokens_per_second=len(trace) * new_tokens / total_seconds,
        retained_peak_bytes=cache.peak, cache_budget_bytes=budget,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
        eviction_decision_p95_ms=float(np.percentile(cache.decision_ms, 95)) if cache.decision_ms else 0,
        output_sha256=hashlib.sha256(str(outputs).encode()).hexdigest(), evictions=cache.evictions)
