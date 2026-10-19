// kineto-amdsmi: standalone smoke test for the amd-smi driver layer.
// Prints two samples one second apart for each enumerated GPU.
// SPDX-License-Identifier: MIT

#include "kineto_amdsmi/AmdSmiDriver.h"

#include <amd_smi/amdsmi.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using kineto_amdsmi::AmdSmiDriver;

static uint64_t now_ns() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock::now().time_since_epoch())
      .count();
}

static void print_sample(const kineto_amdsmi::DeviceSample& s) {
  std::printf(
      "  device[%u] ok=%d power_avg=%.1fW power_cur=%.1fW "
      "temp_hs=%.1fC temp_edge=%.1fC temp_mem=%.1fC "
      "gfx=%.1f%% umc=%.1f%% mm=%.1f%% vram_used=%.1fMB\n",
      s.device_index,
      static_cast<int>(s.ok),
      s.power_avg_w,
      s.power_current_w,
      s.temp_hotspot_c,
      s.temp_edge_c,
      s.temp_mem_c,
      s.gfx_busy_pct,
      s.umc_busy_pct,
      s.mm_busy_pct,
      s.vram_used_mb);
}

// Microbenchmark the per-sweep cost of sample_device across all visible
// devices.  Reports min / median / mean / p99 over many sweeps; useful to
// understand what overhead the sampler thread adds at a given interval.
static int run_bench(AmdSmiDriver& d, int sweeps) {
  const size_t ndev = d.devices().size();
  std::printf("benchmark: %d sweeps over %zu device(s)\n", sweeps, ndev);

  std::vector<double> sweep_ms;
  sweep_ms.reserve(sweeps);

  // Warm-up sweeps.
  for (int w = 0; w < 5; ++w) {
    for (const auto& dev : d.devices()) {
      (void)d.sample_device(dev, 0);
    }
  }

  for (int i = 0; i < sweeps; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    for (const auto& dev : d.devices()) {
      (void)d.sample_device(dev, 0);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    sweep_ms.push_back(ms);
  }

  std::sort(sweep_ms.begin(), sweep_ms.end());
  double sum = 0.0;
  for (double v : sweep_ms) sum += v;
  double mean = sum / sweep_ms.size();
  double p50 = sweep_ms[sweep_ms.size() / 2];
  double p99 = sweep_ms[(sweep_ms.size() * 99) / 100];
  double min_ms = sweep_ms.front();
  double max_ms = sweep_ms.back();

  std::printf(
      "  per-sweep: min=%.3fms p50=%.3fms mean=%.3fms p99=%.3fms max=%.3fms\n",
      min_ms,
      p50,
      mean,
      p99,
      max_ms);
  std::printf(
      "  per-device: ~%.3fms (sweep / %zu devices)\n",
      mean / static_cast<double>(ndev),
      ndev);
  // At a 50ms sample interval, the sampler thread spends `mean` ms out
  // of every 50 actually working.  The rest is sleeping.
  for (int interval : {5, 10, 50, 100, 1000}) {
    double pct = (mean / static_cast<double>(interval)) * 100.0;
    std::printf(
        "  at %dms interval: sampler thread = %.2f%% of one CPU\n",
        interval,
        pct);
  }
  return 0;
}

int main(int argc, char** argv) {
  AmdSmiDriver d;
  if (!d.initialize()) {
    std::fprintf(stderr, "init failed\n");
    return 1;
  }
  std::printf(
      "amd-smi lib %s, %zu devices\n",
      d.lib_version().c_str(),
      d.devices().size());

  if (argc > 1 && std::string(argv[1]) == "--bench") {
    int n = (argc > 2) ? std::atoi(argv[2]) : 1000;
    return run_bench(d, n);
  }

  // Print BDF info for index-mapping debugging.
  std::printf("BDFs:\n");
  for (const auto& dev : d.devices()) {
    amdsmi_bdf_t bdf{};
    if (amdsmi_get_gpu_device_bdf(dev.handle, &bdf) == AMDSMI_STATUS_SUCCESS) {
      std::printf(
          "  device[%u] BDF=%04llx:%02llx:%02llx.%llx UUID=%s (%s)\n",
          dev.index,
          static_cast<unsigned long long>(bdf.domain_number),
          static_cast<unsigned long long>(bdf.bus_number),
          static_cast<unsigned long long>(bdf.device_number),
          static_cast<unsigned long long>(bdf.function_number),
          dev.hip_uuid.empty() ? "?" : dev.hip_uuid.c_str(),
          dev.product_name.c_str());
    }
  }

  int reps = (argc > 1) ? std::atoi(argv[1]) : 2;
  for (int i = 0; i < reps; ++i) {
    auto ts = now_ns();
    std::printf("sample %d (t=%llu ns):\n", i, static_cast<unsigned long long>(ts));
    for (const auto& dev : d.devices()) {
      auto s = d.sample_device(dev, ts);
      print_sample(s);
    }
    if (i < reps - 1) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
  return 0;
}
