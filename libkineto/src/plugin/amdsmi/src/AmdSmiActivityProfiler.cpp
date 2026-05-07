// kineto-amdsmi: amd-smi sampling plugin for kineto / torch.profiler
// SPDX-License-Identifier: MIT

#include "AmdSmiActivityProfiler.h"
#include "AmdSmiActivityProfilerSession.h"

#include <ActivityType.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace kineto_amdsmi {

namespace {

std::chrono::nanoseconds period_from_env() {
  const char* s = std::getenv("KINETO_AMDSMI_INTERVAL_MS");
  long ms = 100;  // default 100ms
  if (s) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end != s && v >= 1 && v < 60000) ms = v;
  }
  return std::chrono::milliseconds(ms);
}

}  // namespace

AmdSmiActivityProfiler::AmdSmiActivityProfiler()
    : driver_(std::make_unique<AmdSmiDriver>()),
      period_(period_from_env()) {}

[[noreturn]] const std::set<libkineto::ActivityType>&
AmdSmiActivityProfiler::availableActivities() const {
  // Mirrors XpuptiActivityProfiler -- the legacy method should never be
  // called by modern kineto.
  throw std::runtime_error(
      "availableActivities is a legacy method and should not be called "
      "by kineto");
}

bool AmdSmiActivityProfiler::ensure_initialized() {
  if (init_attempted_) return init_ok_;
  init_attempted_ = true;
  init_ok_ = driver_->initialize();
  if (init_ok_) {
    std::fprintf(
        stderr,
        "[kineto-amdsmi] active; lib=%s devices=%zu interval=%lldms\n",
        driver_->lib_version().c_str(),
        driver_->devices().size(),
        static_cast<long long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(period_)
                .count()));
  } else {
    std::fprintf(
        stderr,
        "[kineto-amdsmi] init failed; plugin will be inactive for this "
        "profile\n");
  }
  return init_ok_;
}

std::unique_ptr<libkineto::IActivityProfilerSession>
AmdSmiActivityProfiler::configure(
    const std::set<libkineto::ActivityType>& /*activity_types*/,
    const libkineto::Config& /*config*/) {
  if (!ensure_initialized()) {
    // Returning nullptr tells kineto's GenericActivityProfiler not to run
    // this child profiler (see configureChildProfilers()).
    return nullptr;
  }
  return std::make_unique<AmdSmiActivityProfilerSession>(
      driver_.get(), period_, name_);
}

std::unique_ptr<libkineto::IActivityProfilerSession>
AmdSmiActivityProfiler::configure(
    int64_t /*ts_ms*/,
    int64_t /*duration_ms*/,
    const std::set<libkineto::ActivityType>& activity_types,
    const libkineto::Config& config) {
  return configure(activity_types, config);
}

}  // namespace kineto_amdsmi
