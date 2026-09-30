"""Deterministic synthetic workloads; no claim of production representativeness."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random


@dataclass(frozen=True)
class Request:
    prefix: str
    tokens: int


def validate(trace):
    if not trace:
        raise ValueError("Trace must contain requests")
    lengths = {}
    for r in trace:
        if not isinstance(r.prefix, str) or not r.prefix or type(r.tokens) is not int or r.tokens < 1:
            raise ValueError("Each request needs a nonempty prefix and positive integer tokens")
        if r.prefix in lengths and lengths[r.prefix] != r.tokens:
            raise ValueError("A prefix identity must always have the same token length")
        lengths[r.prefix] = r.tokens
    return trace


def generate(n=2000, seed=42, workload="hot", catalog=64):
    if n < 1 or catalog < 2 or workload not in ("hot", "scan", "shift"):
        raise ValueError("Invalid workload, request count, or catalog size")
    rng = random.Random(seed)
    lengths = [rng.choice((32, 64, 128, 256)) for _ in range(catalog)]
    rows = []
    for i in range(n):
        if workload == "scan":
            key = i % catalog
        else:
            offset = catalog // 2 if workload == "shift" and i >= n // 2 else 0
            key = ((rng.randrange(max(2, catalog // 8)) if rng.random() < .85
                    else rng.randrange(catalog)) + offset) % catalog
        rows.append(Request(f"p{key}", lengths[key]))
    return rows


def write_trace(path, trace):
    validate(trace)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(asdict(r)) + "\n" for r in trace))


def read_trace(path):
    return validate([Request(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()])
