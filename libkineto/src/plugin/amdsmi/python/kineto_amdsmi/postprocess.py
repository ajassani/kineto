"""Post-process a torch.profiler chrome trace to convert kineto-amdsmi
sample events into Chrome counter events.

The plugin emits its samples as CPU_INSTANT_EVENT activities named
"amdsmi.counters", with metric values stashed in args metadata.  Current
shipped versions of kineto don't yet have a typed counter event API
(addCounterValue / counterValues()).  This script is the minimal bridge
until that API ships in torch.

Input:  trace.json (from prof.export_chrome_trace)
Output: trace.json (rewritten in place by default), or --out PATH

Usage:
    python -m kineto_amdsmi.postprocess trace.json
    python -m kineto_amdsmi.postprocess trace.json --out trace.with_counters.json

Useful env knobs:
    KINETO_AMDSMI_METRICS=power,temp,vram
    KINETO_AMDSMI_METRICS=gpu_gfx_busy_pct,gpu_power_current_w
    KINETO_AMDSMI_POSTPROCESS_KEEP=active|all
    KINETO_AMDSMI_POSTPROCESS_DEVICES=4[,5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


SAMPLE_NAME = "amdsmi.counters"
DEVICE_INFO_NAME = "amdsmi.device_info"
AMDSMI_PID_BASE = 9_000_000
DEFAULT_METRICS = (
    "gpu_power_w",
    "gpu_power_current_w",
    "gpu_temp_hotspot_c",
    "gpu_temp_edge_c",
    "gpu_temp_mem_c",
    "gpu_gfx_busy_pct",
    "gpu_umc_busy_pct",
    "gpu_mm_busy_pct",
    "gpu_vram_used_mb",
)


def _is_amdsmi_sample(ev: dict[str, Any]) -> bool:
    if ev.get("name") != SAMPLE_NAME:
        return False
    # CPU_INSTANT_EVENT writes ph="i" (instant) in chrome trace format.
    # Be lenient: accept "i", "X" (complete), "I" -- different kineto
    # versions emit slightly differently.
    return ev.get("ph") in ("i", "I", "X")


def _is_amdsmi_device_info(ev: dict[str, Any]) -> bool:
    return (
        ev.get("name") == DEVICE_INFO_NAME and ev.get("ph") in ("i", "I", "X")
    )


def _metric_filter() -> set[str] | None:
    """Return metric allowlist from env, or None to keep all known metrics."""
    raw = os.environ.get("KINETO_AMDSMI_METRICS")
    if not raw:
        return None
    aliases = {
        "power": {"gpu_power_w", "gpu_power_current_w"},
        "temp": {"gpu_temp_hotspot_c", "gpu_temp_edge_c", "gpu_temp_mem_c"},
        "temperature": {"gpu_temp_hotspot_c", "gpu_temp_edge_c", "gpu_temp_mem_c"},
        "busy": {"gpu_gfx_busy_pct", "gpu_umc_busy_pct", "gpu_mm_busy_pct"},
        "activity": {"gpu_gfx_busy_pct", "gpu_umc_busy_pct", "gpu_mm_busy_pct"},
        "mem": {"gpu_vram_used_mb"},
        "memory": {"gpu_vram_used_mb"},
        "vram": {"gpu_vram_used_mb"},
        "all": set(DEFAULT_METRICS),
    }
    selected: set[str] = set()
    for item in raw.split(","):
        key = item.strip()
        if not key:
            continue
        selected.update(aliases.get(key, {key}))
    return selected


def _to_counter_events(
    ev: dict[str, Any],
    pid_map: dict[Any, int],
    metric_names: set[str] | None,
) -> list[dict[str, Any]]:
    """Turn one amdsmi.counters event into N counter ('ph':'C') events,
    one per metric.  Each gets the metric name as the event name and the
    value as the single arg, which is what Perfetto expects to render
    one track per counter.
    """
    args = ev.get("args", {}) or {}
    if not args:
        return []
    original_pid = ev.get("pid")
    if original_pid not in pid_map:
        return []
    pid = pid_map[original_pid]
    tid = ev.get("tid", 0)
    ts = ev.get("ts")
    cat = "amdsmi"
    out: list[dict[str, Any]] = []
    for key, val in args.items():
        if key not in DEFAULT_METRICS:
            continue
        if metric_names is not None and key not in metric_names:
            continue
        # Filter out non-numeric values (e.g. correlation IDs).
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "ph": "C",
                "cat": cat,
                "name": key,
                "pid": pid,
                "tid": tid,
                "ts": ts,
                "args": {"value": v},
            }
        )
    return out


def _device_label(args: dict[str, Any]) -> str:
    """Build a human-readable pid label from a device_info args dict.
    Example: 'amd-smi[2] AMD Instinct MI300X (0000:97:00.0)'.
    """
    idx = args.get("amdsmi_index")
    name = args.get("product_name") or "GPU"
    bdf = args.get("bdf") or ""
    uuid = args.get("hip_uuid") or ""
    parts = ["amd-smi metrics"]
    if idx is not None:
        parts.append(f"[{idx}]")
    parts.append(str(name))
    if bdf:
        parts.append(f"({bdf})")
    elif uuid:
        parts.append(f"(uuid={str(uuid)[:8]})")
    return " ".join(parts)


def _sample_stats(events: list[dict[str, Any]]) -> dict[Any, dict[str, float]]:
    stats: dict[Any, dict[str, float]] = {}
    for ev in events:
        pid = ev.get("pid")
        args = ev.get("args", {}) or {}
        row = stats.setdefault(pid, {})
        for key, val in args.items():
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            row[key] = max(row.get(key, float("-inf")), fval)
    return stats


def _selected_original_pids(
    sample_events: list[dict[str, Any]],
    pid_labels: dict[Any, str],
) -> list[Any]:
    """Choose which original amd-smi pids to keep in the final trace.

    Default policy is "active": keep only GPUs with workload-like movement.
    This avoids producing eight idle SMI processes for a single-GPU PyTorch
    run. Users can override with:

        KINETO_AMDSMI_POSTPROCESS_DEVICES=4[,5]
        KINETO_AMDSMI_POSTPROCESS_KEEP=all
    """
    explicit = os.environ.get("KINETO_AMDSMI_POSTPROCESS_DEVICES")
    if explicit:
        wanted = {p.strip() for p in explicit.split(",") if p.strip()}
        return [pid for pid in pid_labels if str(pid) in wanted]

    keep = os.environ.get("KINETO_AMDSMI_POSTPROCESS_KEEP", "active").lower()
    if keep == "all":
        return sorted(pid_labels, key=str)

    stats = _sample_stats(sample_events)
    if not stats:
        return sorted(pid_labels, key=str)

    max_vrams = [s.get("gpu_vram_used_mb", 0.0) for s in stats.values()]
    min_vram = min(max_vrams) if max_vrams else 0.0
    max_powers = [
        max(s.get("gpu_power_w", 0.0), s.get("gpu_power_current_w", 0.0))
        for s in stats.values()
    ]
    min_power = min(max_powers) if max_powers else 0.0

    selected: list[Any] = []
    for pid, s in stats.items():
        busy = max(s.get("gpu_gfx_busy_pct", 0.0), s.get("gpu_umc_busy_pct", 0.0))
        power = max(s.get("gpu_power_w", 0.0), s.get("gpu_power_current_w", 0.0))
        vram = s.get("gpu_vram_used_mb", 0.0)
        if busy > 5.0 or power > min_power + 20.0 or vram > min_vram + 128.0:
            selected.append(pid)

    # If the trace is too quiet to infer the active device, keep everything
    # rather than silently dropping data.
    return sorted(selected, key=str) if selected else sorted(pid_labels, key=str)


def transform(trace: dict[str, Any]) -> dict[str, Any]:
    events = trace.get("traceEvents", [])
    new_events: list[dict[str, Any]] = []
    seen = 0
    emitted = 0
    pid_labels: dict[Any, str] = {}
    sample_events: list[dict[str, Any]] = []
    metric_names = _metric_filter()
    for ev in events:
        if isinstance(ev, dict) and _is_amdsmi_sample(ev):
            sample_events.append(ev)
            seen += 1
            # We drop the original instant-event marker so it doesn't
            # clutter the trace; counter events fully replace it.
        elif isinstance(ev, dict) and _is_amdsmi_device_info(ev):
            pid = ev.get("pid")
            args = ev.get("args", {}) or {}
            if pid is not None and pid not in pid_labels:
                pid_labels[pid] = _device_label(args)
            # Don't keep the marker event in the output.
        else:
            new_events.append(ev)

    selected_pids = _selected_original_pids(sample_events, pid_labels)
    pid_map = {
        original_pid: AMDSMI_PID_BASE + i
        for i, original_pid in enumerate(selected_pids)
    }
    for ev in sample_events:
        counter_events = _to_counter_events(ev, pid_map, metric_names)
        new_events.extend(counter_events)
        emitted += len(counter_events)

    # Emit Chrome metadata events to label our own high, non-colliding pids.
    # This keeps normal torch CPU/GPU rows untouched and places SMI counters
    # into a separate Perfetto process.
    label_events: list[dict[str, Any]] = []
    for original_pid in selected_pids:
        pid = pid_map[original_pid]
        label = pid_labels.get(original_pid, f"amd-smi metrics [{original_pid}]")
        label_events.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": pid,
                "tid": 0,
                "args": {"name": label},
            }
        )
        label_events.append(
            {
                "ph": "M",
                "name": "process_sort_index",
                "pid": pid,
                "tid": 0,
                "args": {"sort_index": 5_500_000 + int(original_pid)},
            }
        )
    new_events = new_events + label_events

    trace["traceEvents"] = new_events
    print(
        f"kineto_amdsmi.postprocess: matched {seen} sample events, "
        f"emitted {emitted} counter events, "
        f"kept {len(selected_pids)}/{len(pid_labels)} devices in separate process(es)",
        file=sys.stderr,
    )
    return trace


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="trace JSON from torch.profiler")
    ap.add_argument(
        "--out",
        default=None,
        help="output path (default: rewrite input in place)",
    )
    args = ap.parse_args(argv)

    with open(args.input, "r") as f:
        trace = json.load(f)
    transform(trace)

    out = args.out if args.out is not None else args.input
    with open(out, "w") as f:
        json.dump(trace, f)
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
