"""Run a fixed benchmark matrix; archive partial results even after failure."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from .cli import dump
from .traces import generate, write_trace


def run_suite(output, device="cuda", tiny=False, requests=150, repeats=3,
              budgets=(8., 32.), workloads=("hot", "shift", "scan"), seed=99):
    if requests < 1 or repeats < 1 or not budgets or any(b <= 0 for b in budgets):
        raise ValueError("Positive requests, repeats and budgets are required")
    root = Path(output).resolve()
    # Never mix reports with older configurations or overwrite previous evidence.
    root.mkdir(parents=True, exist_ok=False)
    archive = str(root) + ".zip"
    manifest = dict(status="running", device=device, tiny=tiny, dtype="float32",
                    requests=requests, repeats=repeats, budgets_mib=list(budgets),
                    workloads=list(workloads), test_seed=seed, train_seed=42,
                    output_tokens=4, completed=[])
    try:
        dump(root / "manifest.json", manifest)
        subprocess.run([sys.executable, "-m", "kv_cache.cli", "experiment",
                        "--output", str(root / "experiment")], check=True)
        for workload in workloads:
            trace = root / f"trace-{workload}.jsonl"
            write_trace(trace, generate(requests, seed, workload))
            for budget in budgets:
                name = f"{workload}-{budget:g}MiB"
                print(f"\nStarting {name}: {requests} requests × 5 strategies × {repeats} repeats", flush=True)
                command = [sys.executable, "-u", "-m", "kv_cache.cli", "benchmark",
                           "--predictor", str(root / "experiment/predictor.json"),
                           "--trace", str(trace), "--seed", str(seed),
                           "--device", device, "--dtype", "float32",
                           "--budget-mib", str(budget), "--new-tokens", "4",
                           "--repeats", str(repeats), "--output", str(root / f"{name}.json")]
                if tiny:
                    command.append("--tiny")
                subprocess.run(command, check=True)
                report = json.loads((root / f"{name}.json").read_text())
                if not report.get("complete") or report.get("greedy_outputs_match") is not True:
                    raise ValueError(f"Incomplete or failed correctness check: {name}")
                manifest["completed"].append(name)
                dump(root / "manifest.json", manifest)
                shutil.make_archive(str(root), "zip", root)
                print(f"Checkpoint archive ready: {archive}", flush=True)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        dump(root / "manifest.json", manifest)
        shutil.make_archive(str(root), "zip", root)
        print(f"Saved {archive}; status={manifest['status']}", flush=True)
    return archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="New output directory; cannot already exist")
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--requests", type=int, default=150)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--budgets", nargs="+", type=float, default=[8, 32])
    p.add_argument("--workloads", nargs="+", choices=("hot", "shift", "scan"), default=["hot", "shift", "scan"])
    p.add_argument("--seed", type=int, default=99)
    a = p.parse_args()
    run_suite(a.output, a.device, a.tiny, a.requests, a.repeats, a.budgets, a.workloads, a.seed)


if __name__ == "__main__":
    main()
