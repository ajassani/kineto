/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef KINETO_ENABLE_PERFETTO

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "GenericTraceActivity.h"
#include "output_base.h"

namespace perfetto {
class TracingSession;
class Track;
} // namespace perfetto

namespace KINETO_NAMESPACE {

class Config;
struct TraceSpan;

class PerfettoTraceLogger : public libkineto::ActivityLogger {
 public:
  explicit PerfettoTraceLogger(const std::string& traceFileName);
  ~PerfettoTraceLogger() override;

  void handleDeviceInfo(const DeviceInfo& info, int64_t time) override;
  void handleOverheadInfo(const OverheadInfo& info, int64_t time) override;
  void handleResourceInfo(const ResourceInfo& info, int64_t time) override;
  void handleTraceSpan(const TraceSpan& span) override;
  void handleActivity(const ITraceActivity& activity) override;
  void handleGenericActivity(const GenericTraceActivity& activity) override;

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata,
      const std::string& device_properties) override;

  void finalizeTrace(
      const Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime,
      std::unordered_map<std::string, std::vector<std::string>>& metadata) override;

  void finalizeMemoryTrace(const std::string&, const Config&) override;

 private:
  const perfetto::Track& trackForActivity(const ITraceActivity& activity);
  const perfetto::Track& trackForDevice(int64_t deviceId, std::string name);
  void emitFlowIfNeeded(const ITraceActivity& activity);
  void writeTraceFile();

  std::string fileName_;
  bool started_{false};
  bool finalized_{false};
  std::unique_ptr<perfetto::TracingSession> tracingSession_;
  std::unordered_map<std::string, std::unique_ptr<perfetto::Track>> tracks_;
};

} // namespace KINETO_NAMESPACE

#endif // KINETO_ENABLE_PERFETTO
