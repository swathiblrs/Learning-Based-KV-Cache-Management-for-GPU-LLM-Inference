"""Restart-isolated HTTP load evaluation; checkpoint each completed configuration."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import httpx
import numpy as np
from .cli import dump
from .inference import token_ids
from .traces import generate


async def load(url, payloads, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        async def one(index, payload):
            async with semaphore:
                start = time.perf_counter()
                response = await client.post(url+"/generate", json=payload)
                elapsed = (time.perf_counter()-start)*1000
                response.raise_for_status()
                return dict(index=index, http_ms=elapsed, **response.json())
        start = time.perf_counter()
        rows = await asyncio.gather(*(one(i,p) for i,p in enumerate(payloads)))
        seconds = time.perf_counter()-start
        metrics = (await client.get(url+"/metrics")).text
    return dict(requests=rows, elapsed_seconds=seconds,
                requests_per_second=len(rows)/seconds,
                output_tokens_per_second=sum(len(r["token_ids"]) for r in rows)/seconds,
                http_p95_ms=float(np.percentile([r["http_ms"] for r in rows],95)),
                compute_ttft_p95_ms=float(np.percentile([r["compute_ttft_ms"] for r in rows],95)),
                queue_p95_ms=float(np.percentile([r["queue_ms"] for r in rows],95)),
                hit_rate=sum(r["cache_hit"] for r in rows)/len(rows)), metrics


def run_configuration(args, policy, mode, concurrency, repeat, payloads, out):
    # Bind a loopback port chosen by the OS; no public tunnel.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(KV_POLICY=policy, KV_COPY_MODE=mode, KV_DEVICE=args.device,
               KV_TINY="1" if args.tiny else "0", KV_MODEL=args.model,
               KV_BUDGET_BYTES=str(int(args.budget_mib*1024**2)),
               KV_PREDICTOR=str(Path(args.predictor).resolve()),
               KV_MAX_PENDING=str(max(8,concurrency)), USE_TF="0", OMP_NUM_THREADS="1")
    label = f"{policy}-{mode}-c{concurrency}-r{repeat}"
    url = f"http://127.0.0.1:{port}"
    with (out/f"{label}.log").open("w") as log:
        process = subprocess.Popen([sys.executable,"-m","uvicorn","kv_cache.serving:create_app",
                                    "--factory","--host","127.0.0.1","--port",str(port)],
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(timeout=2, trust_env=False) as client:
                deadline = time.monotonic()+300
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited; inspect {label}.log")
                    try:
                        if client.get(url+"/readyz").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic()>deadline:
                        raise TimeoutError(f"Server startup timeout; inspect {label}.log")
                    time.sleep(.5)
                info = client.get(url+"/info").json()
            report, metrics = asyncio.run(load(url,payloads,concurrency))
            (out/f"{label}.prom").write_text(metrics)
            report.update(info=info, concurrency=concurrency, repeat=repeat, label=label)
            dump(out/f"{label}.json",report)
            return report
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictor",required=True)
    p.add_argument("--output",default="results/serving")
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--model",default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--tiny",action="store_true")
    p.add_argument("--requests",type=int,default=30)
    p.add_argument("--new-tokens",type=int,default=4)
    p.add_argument("--seed",type=int,default=99)
    p.add_argument("--workload",choices=("hot","shift","scan"),default="hot")
    p.add_argument("--budget-mib",type=float,default=32)
    p.add_argument("--concurrency",type=int,nargs="+",default=[1,4])
    p.add_argument("--repeats",type=int,default=1)
    p.add_argument("--policies",nargs="+",choices=("none","fifo","lru","lfu","learned"),default=["none","lru","learned"])
    p.add_argument("--copy-modes",nargs="+",choices=("baseline","sliced"),default=["baseline","sliced"])
    args = p.parse_args()
    if min(args.requests,args.repeats,*args.concurrency,args.new_tokens)<1 or args.new_tokens>64 or args.budget_mib<0:
        p.error("Invalid counts or budget")
    out = Path(args.output)
    if out.exists():
        p.error("Output directory already exists; choose a new path to preserve prior runs")
    out.mkdir(parents=True)
    # Preserve exact training artifact and config for reproducibility.
    shutil.copyfile(args.predictor,out/"predictor.json")
    if args.tiny:
        vocab = 128
    else:
        from transformers import AutoConfig
        vocab = AutoConfig.from_pretrained(args.model).vocab_size
    trace = generate(args.requests,args.seed,args.workload)
    payloads = [dict(prefix_ids=token_ids(r,vocab).tolist()[0],suffix_ids=list(range(3,11)),new_tokens=args.new_tokens) for r in trace]
    dump(out/"requests.json",payloads)
    manifest = dict(schema=1,status="running",config=vars(args),completed=[],
                    predictor_sha256=hashlib.sha256(Path(args.predictor).read_bytes()).hexdigest(),
                    trace_sha256=hashlib.sha256(json.dumps(payloads,sort_keys=True).encode()).hexdigest(),
                    note="Synthetic tokens; instrumented nonstreaming service. Concurrency changes arrival order, not model batching. Compare policy performance at concurrency=1; concurrent runs measure queueing and correctness.")
    try:
        manifest["git_commit"] = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
        manifest["git_dirty"] = bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip())
    except (OSError,subprocess.CalledProcessError):
        manifest["git_commit"] = None
    def checkpoint():
        dump(out/"manifest.json",manifest)
        shutil.make_archive(str(out),"zip",out)
    source_manifest = Path("source-manifest.json")
    if source_manifest.exists():
        shutil.copyfile(source_manifest, out/"source-manifest.json")
    checkpoint()
    reference = None
    try:
        for repeat in range(args.repeats):
            for concurrency in args.concurrency:
                configs = [(policy,mode) for policy in dict.fromkeys(["none"]+args.policies)
                           for mode in (["baseline"] if policy=="none" else args.copy_modes)]
                # Reference first; rotate remaining configurations to reduce fixed-order bias.
                tail = configs[1:]
                np.random.default_rng(args.seed+repeat+concurrency).shuffle(tail)
                for policy,mode in configs[:1]+tail:
                    print(f"Running {policy}/{mode}, concurrency={concurrency}, repeat={repeat}",flush=True)
                    report = run_configuration(args,policy,mode,concurrency,repeat,payloads,out)
                    outputs = [r["token_ids"] for r in report["requests"]]
                    if reference is None:
                        reference = outputs
                    mismatch = [i for i,(a,b) in enumerate(zip(reference,outputs)) if a!=b]
                    report["correctness"] = dict(status="failed" if mismatch else "passed",mismatch_indices=mismatch)
                    dump(out/(report["label"]+".json"),report)
                    manifest["completed"].append(dict(label=report["label"],correctness=report["correctness"],
                        hit_rate=report["hit_rate"],http_p95_ms=report["http_p95_ms"],requests_per_second=report["requests_per_second"]))
                    checkpoint()
                    if mismatch:
                        raise RuntimeError("Output mismatch; inspect report, do not bypass")
        manifest["status"] = "passed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        checkpoint()
        print(f"Saved {out}.zip",flush=True)


if __name__ == "__main__":
    main()
