"""Summarize completed serving comparisons without conflating timing modes."""
import argparse
import json
from pathlib import Path
from statistics import median


def summarize(directory):
    root = Path(directory)
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest["status"] != "passed":
        raise ValueError("Only summarize complete, correctness-passing experiments")
    groups, baseline = {}, {}
    metadata = set()
    for item in manifest["completed"]:
        row = json.loads((root/(item["label"]+".json")).read_text())
        if row["correctness"]["status"] != "passed":
            raise ValueError("Correctness failure in report")
        info = row["info"]
        metadata.add((info["model"],info.get("model_revision"),info["dtype"],
                      info["timing_is_instrumented"],info.get("logits_mode","all")))
        key = (info["policy"],info["copy_mode"],row["concurrency"])
        groups.setdefault(key,[]).append(row)
        if info["policy"] == "none":
            baseline[(row["concurrency"],row["repeat"])] = row
    if len(metadata) != 1:
        raise ValueError("Do not combine different models, precisions or timing modes")
    result = []
    for (policy,mode,concurrency), rows in sorted(groups.items()):
        paired = [100*(baseline[(concurrency,r["repeat"])]["http_p95_ms"]-r["http_p95_ms"])
                  /baseline[(concurrency,r["repeat"])]["http_p95_ms"] for r in rows]
        throughput_gain = [100*(r["requests_per_second"]/baseline[(concurrency,r["repeat"])]["requests_per_second"]-1) for r in rows]
        mean_reduction = [100*(1-r["http_mean_ms"]/baseline[(concurrency,r["repeat"])]["http_mean_ms"]) for r in rows] if all("http_mean_ms" in r and "http_mean_ms" in baseline[(concurrency,r["repeat"])] for r in rows) else None
        result.append(dict(policy=policy,copy_mode=mode,concurrency=concurrency,repeats=len(rows),
            median_run_p95_ms=median(r["http_p95_ms"] for r in rows),
            median_requests_per_second=median(r["requests_per_second"] for r in rows),
            median_throughput_gain_percent=median(throughput_gain),
            paired_throughput_gain_percent=throughput_gain,
            throughput_wins=sum(p>0 for p in throughput_gain),
            median_mean_latency_reduction_percent=median(mean_reduction) if mean_reduction is not None else None,
            paired_mean_latency_reduction_percent=mean_reduction,
            mean_latency_wins=sum(p>0 for p in mean_reduction) if mean_reduction is not None else None,
            median_hit_rate=median(r["hit_rate"] for r in rows),
            median_paired_p95_reduction_percent=median(paired),
            paired_p95_reduction_percent=paired,
            faster_than_no_cache_repeats=sum(p>0 for p in paired)))
    return dict(status="passed",comparison="same workload, precision, logit scope and timing mode",
                note="Median of per-run p95s, not pooled p95. Positive reduction means faster in that repeat. Few repeats and one synthetic seed do not establish general speedup.",groups=result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args=parser.parse_args()
    result=summarize(args.directory)
    (Path(args.directory)/"comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    for row in result["groups"]:
        print(f'{row["policy"]:8} {row["copy_mode"]:8} c={row["concurrency"]} '
              f'p95={row["median_run_p95_ms"]:.2f} ms '
              f'rate={row["median_requests_per_second"]:.2f}/s '
              f'paired p95 reduction={row["median_paired_p95_reduction_percent"]:.1f}% '
              f'({row["faster_than_no_cache_repeats"]}/{row["repeats"]} faster)')
    print(result["note"])


if __name__=="__main__":
    main()
