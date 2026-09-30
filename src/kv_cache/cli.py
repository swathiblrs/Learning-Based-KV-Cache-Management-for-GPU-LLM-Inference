import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .model import Predictor, evaluate, examples
from .simulate import simulate
from .traces import generate, read_trace, write_trace


def dump(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + "\n")


def experiment(output, requests=2000, seed=42):
    """Independent train/validation/test traces; frozen hyperparameters, no test tuning."""
    out = Path(output)
    train = generate(requests, seed, "hot")
    validation = generate(requests, seed + 1, "hot")
    x, y = examples(train)
    model = Predictor.fit(x, y)
    model.save(out / "predictor.json", dict(train_seed=seed, requests=requests,
               training_workload="hot", candidates=8, epochs=400, l2=.001))
    write_trace(out / "train.jsonl", train)
    write_trace(out / "validation.jsonl", validation)
    evaluations = {"train": evaluate(model, x, y),
                   "validation": evaluate(model, *examples(validation))}
    results = []
    for workload in ("hot", "shift", "scan"):
        for test_seed in (seed + 2, seed + 3, seed + 4):
            trace = generate(requests, test_seed, workload)
            write_trace(out / f"test-{workload}-{test_seed}.jsonl", trace)
            evaluations[f"test-{workload}-{test_seed}"] = evaluate(model, *examples(trace))
            for budget in (256 * 1024, 1024 * 1024):
                for policy in ("none", "fifo", "lru", "lfu", "learned"):
                    results.append(dict(workload=workload, seed=test_seed,
                                        **simulate(trace, budget, policy, model)))
    dump(out / "model-evaluation.json", evaluations)
    dump(out / "simulation.json", results)
    print(f"Saved model, traces, ML metrics and {len(results)} simulation runs to {out}")
    print("Simulation outputs are NOT GPU latency or throughput measurements.")


def main():
    parser = argparse.ArgumentParser(description="Learned whole-prefix KV-cache experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("experiment", help="Train and evaluate a reproducible synthetic experiment")
    exp.add_argument("--output", default="results/experiment")
    exp.add_argument("--requests", type=int, default=2000)
    exp.add_argument("--seed", type=int, default=42)
    gpu = sub.add_parser("benchmark", help="Run real Transformers inference; CUDA by default")
    gpu.add_argument("--predictor", required=True)
    gpu.add_argument("--trace", help="Optional JSONL trace; must be held out from training")
    gpu.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    gpu.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    gpu.add_argument("--tiny", action="store_true", help="Random tiny Llama for correctness; no download")
    gpu.add_argument("--requests", type=int, default=100)
    gpu.add_argument("--seed", type=int, default=99)
    gpu.add_argument("--budget-mib", type=float, default=4)
    gpu.add_argument("--new-tokens", type=int, default=8)
    gpu.add_argument("--repeats", type=int, default=3)
    gpu.add_argument("--output", default="results/gpu.json")
    args = parser.parse_args()
    try:
        if args.command == "experiment":
            experiment(args.output, args.requests, args.seed)
        else:
            import gc
            from .inference import benchmark, load_model
            if args.repeats < 1 or args.budget_mib < 0:
                raise ValueError("Repeats must be positive; budget must be nonnegative")
            predictor = Predictor.load(args.predictor)
            trace = read_trace(args.trace) if args.trace else generate(args.requests, args.seed, "shift")
            model = load_model(args.model, args.device, args.tiny)
            model._benchmark_tiny = args.tiny
            rows = []
            for repeat in range(args.repeats):
                policies = ["none", "fifo", "lru", "lfu", "learned"]
                np.random.default_rng(args.seed + repeat).shuffle(policies)
                for policy in policies:
                    gc.collect()
                    row = benchmark(model, trace, policy, int(args.budget_mib * 1024 ** 2),
                                    predictor, args.device, args.new_tokens)
                    row.update(repeat=repeat, model="random-tiny-llama" if args.tiny else args.model)
                    rows.append(row)
                    print(f"{policy}: p95 TTFT={row['ttft_p95_ms']:.2f} ms; hits={row['hits']}/{len(trace)}")
            digest = hashlib.sha256("\n".join(f"{r.prefix}:{r.tokens}" for r in trace).encode()).hexdigest()
            dump(args.output, dict(trace_sha256=digest, predictor_sha256=hashlib.sha256(Path(args.predictor).read_bytes()).hexdigest(),
                                  greedy_outputs_match=len({r["output_sha256"] for r in rows}) == 1, runs=rows))
            if len({r["output_sha256"] for r in rows}) != 1:
                raise ValueError("Greedy outputs differ across policies; results saved for investigation, do not claim correctness")
            print(f"Saved {args.output}; identical greedy outputs across all policies.")
    except (ValueError, ImportError) as e:
        parser.exit(2, f"Error: {e}\n")


if __name__ == "__main__":
    main()
