// kineto-amdsmi: amd-smi sampling plugin for kineto / torch.profiler
// SPDX-License-Identifier: MIT

#include "kineto_amdsmi/AmdSmiDriver.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <unordered_set>

namespace kineto_amdsmi {

namespace {

// Minimal local logger.  Plugin must not pull in spdlog or anything else
// that could conflict with libtorch's logging.  Always writes to stderr
// behind a level check.
enum class Lvl { Error = 0, Warn = 1, Info = 2, Debug = 3 };

Lvl current_level() {
  static Lvl lvl = []() {
    const char* s = std::getenv("KINETO_AMDSMI_LOG_LEVEL");
    if (!s) return Lvl::Warn;
    if (std::strcmp(s, "error") == 0) return Lvl::Error;
    if (std::strcmp(s, "warning") == 0 || std::strcmp(s, "warn") == 0)
      return Lvl::Warn;
    if (std::strcmp(s, "info") == 0) return Lvl::Info;
    if (std::strcmp(s, "debug") == 0) return Lvl::Debug;
    return Lvl::Warn;
  }();
  return lvl;
}

#define LOG(lvl, ...)                                                          \
  do {                                                                         \
    if (static_cast<int>(lvl) <= static_cast<int>(current_level())) {          \
      std::fprintf(stderr, "[kineto-amdsmi] " __VA_ARGS__);                    \
      std::fprintf(stderr, "\n");                                              \
    }                                                                          \
  } while (0)

const char* status_str(amdsmi_status_t s) {
  // amd-smi has amdsmi_status_string but it's awkwardly out-parameterised
  // and not always linked; sprintf the numeric value as a fallback.
  static thread_local char buf[32];
  std::snprintf(buf, sizeof(buf), "amdsmi=%d", static_cast<int>(s));
  return buf;
}

double pct_clamp(double v) {
  if (v < 0.0) return 0.0;
  if (v > 100.0) return 100.0;
  return v;
}

std::string normalize_bdf(std::string v) {
  // Accept both "75:00.0" and "0000:75:00.0"; amd-smi prints the latter.
  v.erase(
      std::remove_if(v.begin(), v.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
      }),
      v.end());
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (!v.empty() && v.find(':') == 2) {
    v = "0000:" + v;
  }
  return v;
}

std::unordered_set<std::string> requested_bdfs_from_env() {
  std::unordered_set<std::string> out;
  const char* s = std::getenv("KINETO_AMDSMI_BDFS");
  if (!s || !*s) return out;

  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    auto normalized = normalize_bdf(item);
    if (!normalized.empty()) out.insert(normalized);
  }
  return out;
}

std::string normalize_uuid(std::string v) {
  v.erase(
      std::remove_if(v.begin(), v.end(), [](unsigned char c) {
        return std::isspace(c) != 0 || c == '-';
      }),
      v.end());
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (v.rfind("gpu", 0) == 0) {
    v = v.substr(3);
  }
  return v;
}

std::unordered_set<std::string> requested_uuids_from_env() {
  std::unordered_set<std::string> out;
  const char* s = std::getenv("KINETO_AMDSMI_UUIDS");
  if (!s || !*s) return out;

  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    auto normalized = normalize_uuid(item);
    if (!normalized.empty()) out.insert(normalized);
  }
  return out;
}

}  // namespace

AmdSmiDriver::~AmdSmiDriver() {
  // By default we DO NOT call amdsmi_shut_down() from the destructor.
  // That call interacts badly with libtorch / fmt at process exit -- we
  // see a "corrupted size vs prev_size in fastbins" glibc warning even
  // though the trace has been fully written and flushed.  The OS reclaims
  // every amd-smi resource at exit anyway, so skipping shutdown is safe
  // for the typical "load, profile, exit" use case.  Set
  // KINETO_AMDSMI_FORCE_SHUTDOWN=1 to opt in to the explicit shutdown
  // (e.g. for valgrind / leak-check builds).
  if (std::getenv("KINETO_AMDSMI_FORCE_SHUTDOWN")) {
    shutdown();
    return;
  }
  initialized_ = false;
  devices_.clear();
}

bool AmdSmiDriver::initialize() {
  if (initialized_) return true;

  amdsmi_status_t s = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  if (s != AMDSMI_STATUS_SUCCESS) {
    LOG(Lvl::Error, "amdsmi_init failed: %s", status_str(s));
    return false;
  }

  // Library version (best-effort).
  amdsmi_version_t ver{};
  if (amdsmi_get_lib_version(&ver) == AMDSMI_STATUS_SUCCESS) {
    char vbuf[64];
    std::snprintf(
        vbuf, sizeof(vbuf), "%u.%u.%u", ver.year, ver.major, ver.minor);
    lib_version_ = vbuf;
  }

  // Enumerate sockets, then GPU processors per socket.
  uint32_t socket_count = 0;
  s = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (s != AMDSMI_STATUS_SUCCESS || socket_count == 0) {
    LOG(Lvl::Error,
        "amdsmi_get_socket_handles count failed: %s (sockets=%u)",
        status_str(s),
        socket_count);
    amdsmi_shut_down();
    return false;
  }
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  s = amdsmi_get_socket_handles(&socket_count, sockets.data());
  if (s != AMDSMI_STATUS_SUCCESS) {
    LOG(Lvl::Error, "amdsmi_get_socket_handles fill failed: %s", status_str(s));
    amdsmi_shut_down();
    return false;
  }

  for (auto sock : sockets) {
    uint32_t proc_count = 0;
    if (amdsmi_get_processor_handles(sock, &proc_count, nullptr) !=
            AMDSMI_STATUS_SUCCESS ||
        proc_count == 0) {
      continue;
    }
    std::vector<amdsmi_processor_handle> procs(proc_count);
    if (amdsmi_get_processor_handles(sock, &proc_count, procs.data()) !=
        AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    for (auto h : procs) {
      processor_type_t pt = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
      if (amdsmi_get_processor_type(h, &pt) != AMDSMI_STATUS_SUCCESS) continue;
      if (pt != AMDSMI_PROCESSOR_TYPE_AMD_GPU) continue;

      DeviceInfo d;
      d.handle = h;

      // ASIC info: name (best-effort).
      amdsmi_asic_info_t asic{};
      if (amdsmi_get_gpu_asic_info(h, &asic) == AMDSMI_STATUS_SUCCESS) {
        d.product_name = asic.market_name;
      }

      // BDF -- needed to match HIP enumeration order.
      amdsmi_bdf_t bdf{};
      if (amdsmi_get_gpu_device_bdf(h, &bdf) == AMDSMI_STATUS_SUCCESS) {
        d.bdf_packed =
            (static_cast<uint64_t>(bdf.domain_number) << 32) |
            (static_cast<uint64_t>(bdf.bus_number & 0xff) << 16) |
            (static_cast<uint64_t>(bdf.device_number & 0xff) << 8) |
            (static_cast<uint64_t>(bdf.function_number & 0xff));
        char bbuf[32];
        std::snprintf(
            bbuf,
            sizeof(bbuf),
            "%04x:%02x:%02x.%x",
            static_cast<unsigned>(bdf.domain_number),
            static_cast<unsigned>(bdf.bus_number),
            static_cast<unsigned>(bdf.device_number),
            static_cast<unsigned>(bdf.function_number));
        d.bdf_string = bbuf;
      }

      amdsmi_enumeration_info_t einfo{};
      if (amdsmi_get_gpu_enumeration_info(h, &einfo) == AMDSMI_STATUS_SUCCESS) {
        d.hip_uuid = einfo.hip_uuid;
      }

      devices_.push_back(d);
    }
  }

  // HIP enumerates GPUs in an order that, on the systems we've tested
  // (single-host AMD compute nodes), is BDF-descending --
  // that is, GPU 0 has the highest BDF.  Mirror that ordering so the
  // device_index we expose as Perfetto pid lines up with torch's
  // cuda:N device numbering.  Users can override via env var (todo).
  std::sort(
      devices_.begin(), devices_.end(), [](const auto& a, const auto& b) {
        return a.bdf_packed > b.bdf_packed;
      });
  for (uint32_t i = 0; i < devices_.size(); ++i) {
    devices_[i].index = i;
  }

  auto requested_bdfs = requested_bdfs_from_env();
  if (!requested_bdfs.empty()) {
    const auto before = devices_.size();
    devices_.erase(
        std::remove_if(
            devices_.begin(),
            devices_.end(),
            [&](const DeviceInfo& d) {
              return requested_bdfs.count(normalize_bdf(d.bdf_string)) == 0;
            }),
        devices_.end());
    LOG(Lvl::Info,
        "filtered devices via KINETO_AMDSMI_BDFS=%s (%zu -> %zu)",
        std::getenv("KINETO_AMDSMI_BDFS"),
        before,
        devices_.size());
  }

  auto requested_uuids = requested_uuids_from_env();
  if (!requested_uuids.empty()) {
    const auto before = devices_.size();
    devices_.erase(
        std::remove_if(
            devices_.begin(),
            devices_.end(),
            [&](const DeviceInfo& d) {
              return requested_uuids.count(normalize_uuid(d.hip_uuid)) == 0;
            }),
        devices_.end());
    LOG(Lvl::Info,
        "filtered devices via KINETO_AMDSMI_UUIDS=%s (%zu -> %zu)",
        std::getenv("KINETO_AMDSMI_UUIDS"),
        before,
        devices_.size());
  }

  if (devices_.empty()) {
    LOG(Lvl::Error, "no AMD GPU processors found via amd-smi");
    amdsmi_shut_down();
    return false;
  }

  initialized_ = true;
  LOG(Lvl::Info,
      "initialized; lib=%s devices=%zu",
      lib_version_.c_str(),
      devices_.size());
  for (const auto& d : devices_) {
    LOG(Lvl::Info,
        "  device[%u] %s bdf=%s",
        d.index,
        d.product_name.empty() ? "(unknown)" : d.product_name.c_str(),
        d.bdf_string.empty() ? "?" : d.bdf_string.c_str());
  }
  return true;
}

void AmdSmiDriver::shutdown() {
  if (!initialized_) return;
  amdsmi_shut_down();
  devices_.clear();
  initialized_ = false;
}

DeviceSample AmdSmiDriver::sample_device(
    const DeviceInfo& dev,
    uint64_t ts_ns) const {
  DeviceSample s;
  s.timestamp_ns = ts_ns;
  s.device_index = dev.index;
  s.ok = true;

  amdsmi_gpu_metrics_t m{};
  amdsmi_status_t st = amdsmi_get_gpu_metrics_info(dev.handle, &m);
  if (st != AMDSMI_STATUS_SUCCESS) {
    LOG(Lvl::Debug,
        "device[%u] amdsmi_get_gpu_metrics_info failed: %s",
        dev.index,
        status_str(st));
    s.ok = false;
  } else {
    // Power -- amd-smi reports milliwatts in some firmware versions and watts
    // in others, depending on header version.  amdsmi_gpu_metrics_t fields
    // average_socket_power and current_socket_power are documented as watts;
    // accept the value verbatim.  Some asics (e.g. MI210) don't populate
    // current_socket_power and return 0xffff -- treat that as "unavailable"
    // by reporting 0 instead, so we don't pollute Perfetto with a 65535W
    // reading.
    constexpr uint16_t kU16Sentinel = 0xffff;
    auto u16_or_zero = [&](uint16_t v) -> double {
      return v == kU16Sentinel ? 0.0 : static_cast<double>(v);
    };
    s.power_avg_w = u16_or_zero(m.average_socket_power);
    s.power_current_w = u16_or_zero(m.current_socket_power);

    // Temperatures (already in deg C in amdsmi_gpu_metrics_t).  The
    // 0xffff sentinel means "unavailable" on the firmware revision; treat
    // as 0.
    s.temp_hotspot_c = u16_or_zero(m.temperature_hotspot);
    s.temp_edge_c = u16_or_zero(m.temperature_edge);
    s.temp_mem_c = u16_or_zero(m.temperature_mem);

    // Activity (percent, 0..100).  Some firmwares return 0xffff as the
    // "unavailable" sentinel for these fields too.
    auto activity_or_zero = [&](uint16_t v) -> double {
      return v == kU16Sentinel ? 0.0 : pct_clamp(v);
    };
    s.gfx_busy_pct = activity_or_zero(m.average_gfx_activity);
    s.umc_busy_pct = activity_or_zero(m.average_umc_activity);
    s.mm_busy_pct = activity_or_zero(m.average_mm_activity);
  }

  // Memory usage (separate amd-smi call).
  uint64_t vram_used = 0;
  st = amdsmi_get_gpu_memory_usage(dev.handle, AMDSMI_MEM_TYPE_VRAM, &vram_used);
  if (st == AMDSMI_STATUS_SUCCESS) {
    s.vram_used_mb = static_cast<double>(vram_used) / (1024.0 * 1024.0);
  } else {
    LOG(Lvl::Debug,
        "device[%u] amdsmi_get_gpu_memory_usage failed: %s",
        dev.index,
        status_str(st));
    s.ok = false;
  }

  return s;
}

}  // namespace kineto_amdsmi
