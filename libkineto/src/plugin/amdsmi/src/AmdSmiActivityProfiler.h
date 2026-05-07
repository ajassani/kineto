// kineto-amdsmi: amd-smi sampling plugin for kineto / torch.profiler
// SPDX-License-Identifier: MIT

#pragma once

#include "kineto_amdsmi/AmdSmiDriver.h"

#include <IActivityProfiler.h>

#include <chrono>
#include <memory>
#include <set>
#include <string>

namespace kineto_amdsmi {

// Top-level kineto plugin entry.  Lives for the lifetime of the process;
// hands out one AmdSmiActivityProfilerSession per profile run.
class AmdSmiActivityProfiler : public libkineto::IActivityProfiler {
 public:
  AmdSmiActivityProfiler();
  ~AmdSmiActivityProfiler() override = default;

  AmdSmiActivityProfiler(const AmdSmiActivityProfiler&) = delete;
  AmdSmiActivityProfiler& operator=(const AmdSmiActivityProfiler&) = delete;

  const std::string& name() const override { return name_; }

  [[noreturn]] const std::set<libkineto::ActivityType>& availableActivities()
      const override;

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override;

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t ts_ms,
      int64_t duration_ms,
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override;

 private:
  // Initialize amd-smi if not already initialized.  Returns true on success.
  bool ensure_initialized();

  std::string name_{"__amdsmi_profiler__"};
  std::unique_ptr<AmdSmiDriver> driver_;
  bool init_attempted_ = false;
  bool init_ok_ = false;
  std::chrono::nanoseconds period_;
};

}  // namespace kineto_amdsmi
