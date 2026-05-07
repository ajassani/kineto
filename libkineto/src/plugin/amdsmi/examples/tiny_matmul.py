"""Tiny matmul workload for validating the kineto_amdsmi plugin.

Runs a short bursty matmul loop with intermittent sleeps so that GPU
busy/temp/power counters show clear movement in the resulting trace.

    python tiny_matmul.py [--out trace.json]

Open the produced JSON in https://ui.perfetto.dev to inspect counter tracks.
"""

import argparse
import os
import time

import torch
import kineto_amdsmi  # noqa: F401  -- side effect: registers plugin
from torch.profiler import ProfilerActivity, profile


def workload(device: torch.device, secs_per_burst: float = 0.6) -> None:
    M = 4096
    a = torch.randn(M, M, device=device, dtype=torch.float16)
    b = torch.randn(M, M, device=device, dtype=torch.float16)

    for cycle in range(4):
        # Burst: drive GFX busy and power up.
        t_end = time.perf_counter() + secs_per_burst
        i = 0
        while time.perf_counter() < t_end:
            a = torch.matmul(a, b)
            i += 1
        torch.cuda.synchronize(device=device)
        print(f"cycle {cycle}: burst did {i} matmuls")

        # Idle: let counters drop.
        time.sleep(0.4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="amdsmi_trace.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda not available; need a HIP device for this POC.")

    device = torch.device("cuda:0")
    print(
        f"torch={torch.__version__} hip={torch.version.hip} "
        f"device={torch.cuda.get_device_name(0)}"
    )

    # Warm up CUDA / hip caches; we don't want the first kernel launch in the
    # measured region.
    _ = torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(activities=activities, record_shapes=False) as prof:
        workload(device)
        torch.cuda.synchronize()

    out_path = os.path.abspath(args.out)
    prof.export_chrome_trace(out_path)
    print(f"wrote {out_path}")

    if os.environ.get("KINETO_AMDSMI_SKIP_POSTPROCESS"):
        print("postprocess skipped (KINETO_AMDSMI_SKIP_POSTPROCESS set)")
    else:
        # Convert kineto-amdsmi sample events into Chrome counter events so
        # they render as proper tracks in Perfetto.
        from kineto_amdsmi.postprocess import transform

        import json

        with open(out_path) as f:
            trace = json.load(f)
        transform(trace)
        with open(out_path, "w") as f:
            json.dump(trace, f)
        print(f"post-processed {out_path}")


if __name__ == "__main__":
    main()
    # Skip the slow / fragile libtorch teardown.  We see a glibc
    # "corrupted size vs prev_size in fastbins" warning at process exit
    # when libkineto_amdsmi.so is loaded -- the trace has been fully
    # flushed by this point, but a static fmt instance somewhere along
    # the libtorch_cpu / libfmt boundary mis-frees a buffer at C++ atexit
    # time.  Exiting via _exit skips C++ destructors entirely, which is
    # safe here because we've already json.dump'd the trace to disk.
    if not os.environ.get("KINETO_AMDSMI_GRACEFUL_EXIT"):
        os._exit(0)
