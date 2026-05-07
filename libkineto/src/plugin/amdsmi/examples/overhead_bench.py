"""Measure overhead of the kineto-amdsmi plugin vs. baseline torch.profiler.

Runs a fixed-work matmul loop multiple times under each of:
  baseline       -- no torch.profiler at all (raw GPU throughput)
  prof_only      -- torch.profiler, no plugin
  prof_plus_plug -- torch.profiler + kineto-amdsmi plugin

Reports wall-time of the workload region (excluding setup / trace export)
and matmul throughput.  The point is: does enabling the plugin slow the
workload itself down measurably?

Usage:
    python overhead_bench.py --mode baseline      [--out ...]
    python overhead_bench.py --mode prof_only     [--out ...]
    python overhead_bench.py --mode prof_plus_plug [--out ...]

The driver script overhead_bench.sh runs all three N times and prints a
table.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

MODES = {"baseline", "prof_only", "prof_plus_plug"}


def fixed_work(device, M=4096, n_iters=2000) -> float:
    """Run exactly n_iters matmuls of MxM fp16 and return wall seconds.
    Synchronizes both before and after, so the timing only measures the
    matmul stream.
    """
    import torch

    a = torch.randn(M, M, device=device, dtype=torch.float16)
    b = torch.randn(M, M, device=device, dtype=torch.float16)
    torch.cuda.synchronize(device=device)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        a = torch.matmul(a, b)
    torch.cuda.synchronize(device=device)
    t1 = time.perf_counter()

    return t1 - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=sorted(MODES), required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--out", default=None, help="optional trace path; only used for prof_* modes")
    ap.add_argument("--results-json", default=None, help="if set, append timings as JSON line")
    args = ap.parse_args()

    import torch
    if args.mode == "prof_plus_plug":
        import kineto_amdsmi  # noqa: F401

    if not torch.cuda.is_available():
        print("torch.cuda not available", file=sys.stderr)
        return 1

    device = torch.device("cuda:0")

    _ = torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)
    torch.cuda.synchronize(device=device)
    _ = fixed_work(device, M=args.m, n_iters=64)

    timings = []

    if args.mode == "baseline":
        for r in range(args.reps):
            t = fixed_work(device, M=args.m, n_iters=args.iters)
            timings.append(t)
            print(f"  rep {r}: {t:.4f}s ({args.iters / t:.1f} matmul/s)", flush=True)
    else:
        from torch.profiler import ProfilerActivity, profile

        for r in range(args.reps):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
            ) as prof:
                t = fixed_work(device, M=args.m, n_iters=args.iters)
            timings.append(t)
            print(f"  rep {r}: {t:.4f}s ({args.iters / t:.1f} matmul/s)", flush=True)

            if args.out:
                trace_path = args.out.replace(".json", f".rep{r}.json")
            else:
                trace_path = f"/tmp/overhead_{args.mode}_rep{r}.json"
            prof.export_chrome_trace(trace_path)

    avg = statistics.mean(timings)
    stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
    print(
        f"mode={args.mode} reps={args.reps} iters={args.iters} M={args.m} "
        f"avg={avg:.4f}s stdev={stdev:.4f}s throughput={args.iters / avg:.1f} matmul/s",
        flush=True,
    )

    if args.results_json:
        with open(args.results_json, "a") as f:
            f.write(
                json.dumps(
                    {
                        "mode": args.mode,
                        "reps": args.reps,
                        "iters": args.iters,
                        "m": args.m,
                        "timings_s": timings,
                        "avg_s": avg,
                        "stdev_s": stdev,
                        "throughput_matmul_per_s": args.iters / avg,
                    }
                )
                + "\n"
            )

    return 0


if __name__ == "__main__":
    rc = main()
    if not os.environ.get("KINETO_AMDSMI_GRACEFUL_EXIT"):
        os._exit(rc)
