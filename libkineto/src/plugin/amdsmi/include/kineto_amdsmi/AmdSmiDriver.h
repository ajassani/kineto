// kineto-amdsmi: amd-smi sampling plugin for kineto / torch.profiler
// SPDX-License-Identifier: MIT

#pragma once

#include <amd_smi/amdsmi.h>

#include <cstdint>
#include <string>
#include <vector>

namespace kineto_amdsmi {

// Per-device handle + identifying info, populated once at init().
struct DeviceInfo {
  // Stable index 0..N-1 in enumeration order, after we've sorted by BDF
  // to match HIP's enumeration.  Used as Perfetto pid.
  uint32_t index = 0;
  amdsmi_processor_handle handle = nullptr;
  // Best-effort -- populated if amdsmi_get_gpu_asic_info works.
  std::string product_name;
  // Full BDF as a single uint64: (domain << 32) | (bus << 16) | (dev << 8) | fn.
  // Compared as integers when sorting devices into HIP-matching order.
  uint64_t bdf_packed = 0;
  // BDF as a string ("0000:03:00.0") for user-facing metadata.
  std::string bdf_string;
  // HIP UUID as reported by amd-smi enumeration info. PyTorch ROCm exposes
  // this as torch.cuda.get_device_properties(i).uuid.
  std::string hip_uuid;
};

// One sample worth of metrics from amdsmi_get_gpu_metrics_info plus a memory
// usage read.  Members default to 0; consumers should treat 0 as "not
// available" for fields that physically can't be 0 on a healthy GPU
// (temperature, power).
struct DeviceSample {
  uint64_t timestamp_ns = 0;
  uint32_t device_index = 0;

  // Power (watts).
  double power_avg_w = 0.0;
  double power_current_w = 0.0;

  // Temperatures (deg C).
  double temp_hotspot_c = 0.0;
  double temp_edge_c = 0.0;
  double temp_mem_c = 0.0;

  // Activity (percent).
  double gfx_busy_pct = 0.0;
  double umc_busy_pct = 0.0;
  double mm_busy_pct = 0.0;

  // Memory.
  double vram_used_mb = 0.0;

  // Set false if either underlying amdsmi call failed; consumers may still
  // use partially populated fields but should know the read had errors.
  bool ok = true;
};

// Thin RAII wrapper around the subset of amd-smi we need.
//
// The interface is deliberately small (mirrors what
// rocm-systems/projects/rocprofiler-systems/source/lib/rocprof-sys/library/
// pmc/device_providers/amd_smi/drivers/driver.hpp does) so it can be unit
// tested with a fake driver in the future.
class AmdSmiDriver {
 public:
  AmdSmiDriver() = default;
  AmdSmiDriver(const AmdSmiDriver&) = delete;
  AmdSmiDriver& operator=(const AmdSmiDriver&) = delete;
  ~AmdSmiDriver();

  // Initializes amd-smi and enumerates devices.  Returns true on success.
  // On failure callers should not use any other method.
  bool initialize();

  // Tear down amd-smi cleanly.  Safe to call multiple times.
  void shutdown();

  // Read one sample for the given device.  Returns DeviceSample with .ok
  // reflecting whether reads succeeded.  ts_ns is filled in by the caller
  // (so we keep one timestamp for the whole sweep across devices).
  DeviceSample sample_device(const DeviceInfo& dev, uint64_t ts_ns) const;

  // Enumerated devices.  Empty until initialize() succeeds.
  const std::vector<DeviceInfo>& devices() const { return devices_; }

  // Library version reported by amdsmi_get_lib_version, or "unknown".
  const std::string& lib_version() const { return lib_version_; }

 private:
  bool initialized_ = false;
  std::vector<DeviceInfo> devices_;
  std::string lib_version_ = "unknown";
};

}  // namespace kineto_amdsmi
