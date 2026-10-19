// kineto-amdsmi: amd-smi sampling plugin for kineto / torch.profiler
// SPDX-License-Identifier: MIT

#pragma once

#include "AmdSmiSampler.h"

#include <IActivityProfiler.h>
#include <Config.h>
// libkineto.h gives us the full CpuTraceBuffer definition needed for
// std::unique_ptr<CpuTraceBuffer>.  IActivityProfiler.h only forward-
// declares it, which makes the unique_ptr destructor uninstantiable at
// the point of class definition.
#include <libkineto.h>

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace kineto_amdsmi {

// One per-profile-run session.  Owns a sampler, drains it at processTrace()
// time, and emits one GenericTraceActivity per sample to the kineto logger.
class AmdSmiActivityProfilerSession
    : public libkineto::IActivityProfilerSession {
 public:
  AmdSmiActivityProfilerSession(
      AmdSmiDriver* driver,
      std::chrono::nanoseconds period,
      const std::string& name);

  ~AmdSmiActivityProfilerSession() override;

  AmdSmiActivityProfilerSession(const AmdSmiActivityProfilerSession&) = delete;
  AmdSmiActivityProfilerSession& operator=(
      const AmdSmiActivityProfilerSession&) = delete;

  // ====== IActivityProfilerSession overrides ======

  void start() override;
  void stop() override;

  std::vector<std::string> errors() override { return errors_; }

  void processTrace(libkineto::ActivityLogger& logger) override;
  void processTrace(
      libkineto::ActivityLogger& logger,
      libkineto::getLinkedActivityCallback get_linked_activity,
      int64_t captureWindowStartTime,
      int64_t captureWindowEndTime) override;

  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override {
    return {};
  }

  std::vector<libkineto::ResourceInfo> getResourceInfos() override {
    return {};
  }

  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override {
    return {};
  }

  // No correlation IDs for sampler-only counter events.
  void pushCorrelationId(uint64_t) override {}
  void popCorrelationId() override {}
  void pushUserCorrelationId(uint64_t) override {}
  void popUserCorrelationId() override {}

 private:
  AmdSmiDriver* driver_ = nullptr;
  std::chrono::nanoseconds period_;
  std::string name_;
  std::unique_ptr<AmdSmiSampler> sampler_;
  std::vector<std::string> errors_;
  bool process_metadata_emitted_ = false;
};

}  // namespace kineto_amdsmi
