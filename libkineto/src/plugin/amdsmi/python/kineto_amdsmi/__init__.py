"""kineto_amdsmi: import-side loader for the amd-smi kineto plugin.

Importing this module dlopens libkineto_amdsmi.so so its constructor
registers the activity profiler with libkineto.  After this import,
torch.profiler runs will collect amd-smi counters automatically.

Usage:

    import torch
    import kineto_amdsmi  # noqa: F401  -- side effect: load .so
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        ... your code ...
    p.export_chrome_trace("trace.json")

The .so is searched for in this order:
    1. $KINETO_AMDSMI_LIB if set
    2. libkineto_amdsmi.so on the system loader path (LD_LIBRARY_PATH)
    3. ./libkineto_amdsmi.so (the build dir if running from source)
"""

import ctypes
import os
import sys


def _normalize_bdf(bdf: str) -> str:
    bdf = bdf.strip().lower()
    # PyTorch may report "75:00.0"; amd-smi uses "0000:75:00.0".
    if len(bdf) >= 7 and bdf[2] == ":":
        bdf = "0000:" + bdf
    return bdf


def _normalize_uuid(uuid: str) -> str:
    value = uuid.strip().lower().replace("-", "")
    if value.startswith("gpu"):
        value = value[3:]
    # ROCm PyTorch currently reports the HIP UUID suffix as ASCII bytes
    # encoded into UUID form.  Example:
    #   torch: 66326561-6435-3936-6237-386532333238
    #   smi:   GPU-f2ead596b78e2328
    # Decode that form when possible.
    if len(value) % 2 == 0:
        try:
            decoded = bytes.fromhex(value).decode("ascii")
            if decoded and all(c in "0123456789abcdef" for c in decoded):
                value = decoded
        except Exception:
            pass
    return value


def _maybe_set_current_torch_bdf_filter() -> None:
    """If torch is already imported, filter SMI sampling to current cuda device.

    This keeps the trace focused: for a single-GPU PyTorch process, the plugin
    samples only the physical GPU backing `torch.cuda.current_device()` instead
    of all GPUs on the node. Users can override explicitly with:

        KINETO_AMDSMI_BDFS=0000:75:00.0[,0000:e7:00.0]
        KINETO_AMDSMI_UUIDS=<torch cuda uuid>[,...]
    """
    if os.environ.get("KINETO_AMDSMI_BDFS") or os.environ.get("KINETO_AMDSMI_UUIDS"):
        return

    torch = sys.modules.get("torch")
    if torch is None:
        return

    try:
        if not torch.cuda.is_available():
            return
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        bdf = getattr(props, "pci_bus_id", None)
        uuid = getattr(props, "uuid", None)
        if bdf:
            os.environ["KINETO_AMDSMI_BDFS"] = _normalize_bdf(str(bdf))
            ident = f"BDF={os.environ['KINETO_AMDSMI_BDFS']}"
        elif uuid:
            os.environ["KINETO_AMDSMI_UUIDS"] = _normalize_uuid(str(uuid))
            ident = f"UUID={os.environ['KINETO_AMDSMI_UUIDS']}"
        else:
            return
        sys.stderr.write(
            "[kineto_amdsmi] sampling current torch cuda device "
            f"{dev} ({ident})\n"
        )
    except Exception as exc:  # pragma: no cover - best-effort convenience.
        sys.stderr.write(f"[kineto_amdsmi] could not derive torch BDF: {exc}\n")


def _candidates():
    env = os.environ.get("KINETO_AMDSMI_LIB")
    if env:
        yield env
    yield "libkineto_amdsmi.so"
    here = os.path.dirname(os.path.abspath(__file__))
    yield os.path.join(here, "..", "..", "build", "libkineto_amdsmi.so")
    yield os.path.join(here, "libkineto_amdsmi.so")


_loaded = None
_errors = []
_maybe_set_current_torch_bdf_filter()
# RTLD_NODELETE keeps the plugin in memory across `dlclose`, so when the
# Python interpreter unloads modules at process exit the plugin's vtable
# stays valid for libtorch / libkineto's atexit destructors.  Without
# this we hit a heap corruption warning at process teardown because
# kineto's GenericActivityProfiler dtor calls into our vtable after our
# .so has been unloaded.  Constant value: glibc's bits/dlfcn.h defines
# RTLD_NODELETE = 0x01000.
RTLD_NODELETE = 0x1000
mode = ctypes.RTLD_GLOBAL | RTLD_NODELETE
for cand in _candidates():
    try:
        _loaded = ctypes.CDLL(cand, mode=mode)
        break
    except OSError as e:
        _errors.append(f"{cand}: {e}")
        continue

if _loaded is None:
    sys.stderr.write(
        "[kineto_amdsmi] failed to dlopen plugin .so. Tried:\n  "
        + "\n  ".join(_errors)
        + "\nSet KINETO_AMDSMI_LIB to the full path of libkineto_amdsmi.so.\n"
    )
