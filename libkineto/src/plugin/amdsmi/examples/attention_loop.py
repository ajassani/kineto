"""More realistic workload: scaled-dot-product attention loop with
varying sequence lengths.  Designed to produce visible variation in
GPU activity, power, and memory over the trace -- so reviewers can
verify the counter tracks actually correlate with kernel events.

    python attention_loop.py [--out trace.json]
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import kineto_amdsmi  # noqa: F401  -- side effect: register plugin
from torch.profiler import ProfilerActivity, profile


def workload(device: torch.device, dtype: torch.dtype = torch.float16) -> None:
    batch = 4
    heads = 16
    head_dim = 128

    # Three phases:
    #   1) short seq (cheap, low activity)
    #   2) long seq (expensive, high activity)
    #   3) bursty: alternate between a small and big seq
    #
    # The trace should show power / busy / temp ramping with phase 2 and
    # oscillating during phase 3.

    def attn_step(seq_len: int) -> None:
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        # On MI210 the AOTriton flash backend hits HSA_STATUS_ERROR_EXCEPTION
        # in some ROCm PyTorch images; force the math backend so the
        # workload runs reliably.
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )
        _ = out.sum().item()

    print("phase 1: short seq -- light load", flush=True)
    t_end = time.perf_counter() + 1.5
    n = 0
    while time.perf_counter() < t_end:
        attn_step(512)
        n += 1
    print(f"  {n} steps", flush=True)

    print("phase 2: long seq -- heavy load", flush=True)
    t_end = time.perf_counter() + 2.5
    n = 0
    while time.perf_counter() < t_end:
        attn_step(4096)
        n += 1
    print(f"  {n} steps", flush=True)

    print("phase 3: bursty -- alternating small/big seq", flush=True)
    for cycle in range(6):
        attn_step(4096)
        for _ in range(4):
            attn_step(512)
        print(f"  cycle {cycle} done", flush=True)

    torch.cuda.synchronize(device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="attention_trace.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda not available; need a HIP device.")

    device = torch.device("cuda:0")
    print(
        f"torch={torch.__version__} hip={torch.version.hip} "
        f"device={torch.cuda.get_device_name(0)}"
    )

    # Warm up.
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
        return

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
    if not os.environ.get("KINETO_AMDSMI_GRACEFUL_EXIT"):
        os._exit(0)
