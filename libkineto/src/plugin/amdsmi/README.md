# AMD SMI Kineto plugin POC

This directory contains a proof-of-concept Kineto child profiler that samples
AMD GPU device metrics through `amd-smi` and adds them to a
`torch.profiler` Chrome trace.

The implementation is intentionally standalone:

- `libkineto_amdsmi.so` registers an `IActivityProfiler` when loaded.
- A background thread samples `amdsmi_get_gpu_metrics_info` and VRAM usage.
- Current PyTorch/Kineto releases do not expose a native counter event API, so
  samples are first emitted as `CPU_INSTANT_EVENT` markers.
- `python/kineto_amdsmi/postprocess.py` rewrites those markers into Chrome
  counter events (`"ph": "C"`) under a separate `amd-smi metrics` process.

## Usage

```python
import torch
import kineto_amdsmi  # load after torch so the current device can be detected

from torch.profiler import ProfilerActivity, profile

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    workload()

prof.export_chrome_trace("trace.json")

from kineto_amdsmi.postprocess import transform
import json

with open("trace.json") as f:
    trace = json.load(f)
transform(trace)
with open("trace.json", "w") as f:
    json.dump(trace, f)
```

Open the resulting JSON in Perfetto. The normal Kineto CPU/GPU process layout
is preserved; SMI counters appear in a separate `amd-smi metrics` process.

## Configuration

Environment variables:

- `KINETO_AMDSMI_INTERVAL_MS=50`: sampler interval in milliseconds.
- `KINETO_AMDSMI_BDFS=0000:75:00.0[,..]`: sample only these PCI BDFs.
- `KINETO_AMDSMI_UUIDS=<uuid>[,..]`: sample only these HIP UUIDs.
- `KINETO_AMDSMI_METRICS=power,temp,vram,busy`: postprocess metric allowlist.
- `KINETO_AMDSMI_POSTPROCESS_KEEP=active|all`: keep active GPUs or all GPUs.
- `KINETO_AMDSMI_POSTPROCESS_DEVICES=4[,5]`: manually select SMI device ids.
- `KINETO_AMDSMI_DISABLE=1`: disable registration.

When `torch` is imported before `kineto_amdsmi`, the Python loader attempts to
derive the current CUDA/HIP device UUID and sets `KINETO_AMDSMI_UUIDS`
automatically. This keeps single-GPU PyTorch traces focused on the GPU used by
the process.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DROCM_PATH=/opt/rocm
cmake --build build -j
```

The standalone build expects:

- ROCm `amd-smi` headers and library.
- Kineto headers from a PyTorch installation or source tree.
- `libtorch_cpu.so` for the Kineto API symbols used by the plugin.

This POC is not wired into Kineto's top-level build yet. The next upstream step
would be to replace the postprocess bridge with native counter-event emission
and plumb configuration through `torch.profiler._ExperimentalConfig`.
