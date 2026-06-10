/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef KINETO_ENABLE_PERFETTO
namespace libkineto {
namespace {
[[maybe_unused]] constexpr int kPerfettoTraceLoggerDisabled = 0;
} // namespace
} // namespace libkineto
#else

#include "output_perfetto.h"

#include <perfetto.h>

#include <fmt/format.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ActivityType.h"
#include "Config.h"
#include "Logger.h"
#include "ThreadUtil.h"
#include "TraceSpan.h"

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("cpu_op").SetDescription("CPU operator"),
    perfetto::Category("user_annotation").SetDescription("User annotation"),
    perfetto::Category("gpu_user_annotation").SetDescription("GPU user annotation"),
    perfetto::Category("gpu_memcpy").SetDescription("GPU memory copy"),
    perfetto::Category("gpu_memset").SetDescription("GPU memory set"),
    perfetto::Category("kernel").SetDescription("GPU kernel"),
    perfetto::Category("external_correlation").SetDescription("External correlation"),
    perfetto::Category("cuda_runtime").SetDescription("CUDA/HIP runtime event"),
    perfetto::Category("cuda_driver").SetDescription("CUDA/HIP driver event"),
    perfetto::Category("cpu_instant_event").SetDescription("CPU instant event"),
    perfetto::Category("python_function").SetDescription("Python function"),
    perfetto::Category("overhead").SetDescription("Profiler overhead"),
    perfetto::Category("mtia_runtime").SetDescription("MTIA runtime event"),
    perfetto::Category("mtia_ccp_events").SetDescription("MTIA CCP event"),
    perfetto::Category("mtia_insight").SetDescription("MTIA insight event"),
    perfetto::Category("cuda_sync").SetDescription("CUDA/HIP synchronization"),
    perfetto::Category("cuda_event").SetDescription("CUDA/HIP event"),
    perfetto::Category("mtia_counters").SetDescription("MTIA counter"),
    perfetto::Category("glow_runtime").SetDescription("Glow runtime event"),
    perfetto::Category("cuda_profiler_range").SetDescription("CUDA profiler range"),
    perfetto::Category("hpu_op").SetDescription("HPU operation"),
    perfetto::Category("xpu_runtime").SetDescription("XPU runtime event"),
    perfetto::Category("xpu_driver").SetDescription("XPU driver event"),
    perfetto::Category("collective_comm").SetDescription("Collective communication"),
    perfetto::Category("privateuse1_runtime").SetDescription("PrivateUse1 runtime event"),
    perfetto::Category("privateuse1_driver").SetDescription("PrivateUse1 driver event"),
    perfetto::Category("xpu_scope_profiler").SetDescription("XPU scope profiler"));

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace KINETO_NAMESPACE {
namespace {

constexpr size_t kBufferSizeKb = 102400;
constexpr size_t kSharedMemorySizeKb = 64;

#ifdef __linux__
constexpr const char* kDefaultLogFileFmt = "/tmp/libkineto_activities_{}.pftrace";
#else
constexpr const char* kDefaultLogFileFmt = "libkineto_activities_{}.pftrace";
#endif

uint64_t stableTrackId(const std::string& key) {
  return std::hash<std::string>{}(key);
}

std::string makeTrackKey(std::string_view prefix, int64_t device, int64_t resource) {
  std::ostringstream os;
  os << prefix << ':' << device << ':' << resource;
  return os.str();
}

bool isDeviceActivity(ActivityType type) {
  switch (type) {
    case ActivityType::GPU_USER_ANNOTATION:
    case ActivityType::GPU_MEMCPY:
    case ActivityType::GPU_MEMSET:
    case ActivityType::CONCURRENT_KERNEL:
    case ActivityType::MTIA_CCP_EVENTS:
    case ActivityType::MTIA_INSIGHT:
    case ActivityType::MTIA_COUNTERS:
    case ActivityType::XPU_SCOPE_PROFILER:
      return true;
    default:
      return false;
  }
}

void addDebugAnnotation(
    perfetto::EventContext& ctx,
    const std::string& name,
    const std::string& value) {
  if (value.empty()) {
    return;
  }
  auto* annotation = ctx.event()->add_debug_annotations();
  annotation->set_name(name);
  annotation->set_string_value(value);
}

void addDebugAnnotation(
    perfetto::EventContext& ctx,
    const std::string& name,
    int64_t value) {
  auto* annotation = ctx.event()->add_debug_annotations();
  annotation->set_name(name);
  annotation->set_int_value(value);
}

void addActivityDebugAnnotations(
    perfetto::EventContext& ctx,
    const ITraceActivity& activity,
    int64_t beginNs,
    int64_t endNs) {
  addDebugAnnotation(ctx, "activity_type", toString(activity.type()));
  addDebugAnnotation(ctx, "begin_ns", beginNs);
  addDebugAnnotation(ctx, "end_ns", endNs);
  addDebugAnnotation(ctx, "device", activity.deviceId());
  addDebugAnnotation(ctx, "resource", activity.resourceId());
  addDebugAnnotation(ctx, "thread_id", activity.getThreadId());
  addDebugAnnotation(ctx, "correlation_id", activity.correlationId());
  addDebugAnnotation(ctx, "metadata_json", activity.metadataJson());
}

void initializePerfettoOnce() {
  static std::once_flag once;
  std::call_once(once, []() {
    perfetto::TracingInitArgs args;
    args.backends |= perfetto::kInProcessBackend;
    args.shmem_size_hint_kb = kSharedMemorySizeKb;
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();
  });
}

std::string defaultFileName() {
  return fmt::format(kDefaultLogFileFmt, processId());
}

} // namespace

PerfettoTraceLogger::PerfettoTraceLogger(const std::string& traceFileName)
    : fileName_(traceFileName.empty() ? defaultFileName() : traceFileName) {}

PerfettoTraceLogger::~PerfettoTraceLogger() {
  if (started_ && !finalized_) {
    try {
      writeTraceFile();
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to finalize Perfetto trace: " << e.what();
    }
  }
}

void PerfettoTraceLogger::handleTraceStart(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::string& device_properties) {
  if (started_) {
    return;
  }

  initializePerfettoOnce();

  perfetto::TraceConfig cfg;
  auto* buffer = cfg.add_buffers();
  buffer->set_size_kb(kBufferSizeKb);
  buffer->set_fill_policy(
      perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_DISCARD);

  perfetto::protos::gen::TrackEventConfig trackEventConfig;
  auto* dataSource = cfg.add_data_sources()->mutable_config();
  dataSource->set_name("track_event");
  dataSource->set_track_event_config_raw(trackEventConfig.SerializeAsString());

  tracingSession_ = perfetto::Tracing::NewTrace();
  tracingSession_->Setup(cfg);
  tracingSession_->StartBlocking();
  started_ = true;

  auto metadataTrack = perfetto::Track{stableTrackId("kineto:metadata")};
  auto descriptor = metadataTrack.Serialize();
  descriptor.set_name("Kineto metadata");
  perfetto::TrackEvent::SetTrackDescriptor(metadataTrack, descriptor);
  tracks_.emplace(
      "kineto:metadata", std::make_unique<perfetto::Track>(metadataTrack));

  TRACE_EVENT_BEGIN(
      "user_annotation",
      "Kineto trace metadata",
      metadataTrack,
      static_cast<uint64_t>(0),
      [&](perfetto::EventContext ctx) {
        addDebugAnnotation(ctx, "device_properties", device_properties);
        for (const auto& [key, value] : metadata) {
          addDebugAnnotation(ctx, key, value);
        }
      });
  TRACE_EVENT_END("user_annotation", metadataTrack, static_cast<uint64_t>(1));
}

void PerfettoTraceLogger::handleDeviceInfo(const DeviceInfo& info, int64_t /*time*/) {
  trackForDevice(info.id, info.name);
}

void PerfettoTraceLogger::handleResourceInfo(
    const ResourceInfo& info,
    int64_t /*time*/) {
  std::ostringstream name;
  name << info.name << " (device " << info.deviceId << ", resource " << info.id << ")";
  trackForDevice(info.id, name.str());
}

void PerfettoTraceLogger::handleOverheadInfo(
    const OverheadInfo& info,
    int64_t time) {
  auto track = perfetto::Track{stableTrackId("kineto:overhead")};
  TRACE_EVENT_BEGIN(
      "overhead",
      perfetto::DynamicString(info.name.c_str()),
      track,
      static_cast<uint64_t>(time));
  TRACE_EVENT_END("overhead", track, static_cast<uint64_t>(time + 1));
}

void PerfettoTraceLogger::handleTraceSpan(const TraceSpan& span) {
  auto track = perfetto::Track{stableTrackId("kineto:trace_span")};
  auto descriptor = track.Serialize();
  descriptor.set_name("Kineto trace spans");
  perfetto::TrackEvent::SetTrackDescriptor(track, descriptor);
  TRACE_EVENT_BEGIN(
      "user_annotation",
      perfetto::DynamicString(span.name.c_str()),
      track,
      static_cast<uint64_t>(span.startTime),
      "iteration",
      span.iteration);
  TRACE_EVENT_END("user_annotation", track, static_cast<uint64_t>(span.endTime));
}

void PerfettoTraceLogger::handleGenericActivity(
    const GenericTraceActivity& activity) {
  handleActivity(activity);
}

void PerfettoTraceLogger::handleActivity(const ITraceActivity& activity) {
  if (!started_) {
    handleTraceStart({}, "");
  }

  auto name = activity.name();
  auto beginNs = activity.timestamp();
  auto endNs = beginNs + std::max<int64_t>(activity.duration(), 1);
  const auto& track = trackForActivity(activity);

  const auto flowId = activity.flowId() != 0 ? activity.flowId()
      : activity.linkedActivity() ? activity.linkedActivity()->correlationId()
      : 0;

#define KINETO_TRACE_ACTIVITY_FOR_CATEGORY(category_literal)                 \
  do {                                                                       \
    if (flowId != 0) {                                                       \
      TRACE_EVENT_BEGIN(                                                     \
          category_literal,                                                  \
          perfetto::DynamicString(name.c_str()),                             \
          track,                                                             \
          static_cast<uint64_t>(beginNs),                                    \
          perfetto::Flow::ProcessScoped(flowId),                             \
          [&](perfetto::EventContext ctx) {                                  \
            addActivityDebugAnnotations(ctx, activity, beginNs, endNs);      \
          });                                                                \
    } else {                                                                 \
      TRACE_EVENT_BEGIN(                                                     \
          category_literal,                                                  \
          perfetto::DynamicString(name.c_str()),                             \
          track,                                                             \
          static_cast<uint64_t>(beginNs),                                    \
          [&](perfetto::EventContext ctx) {                                  \
            addActivityDebugAnnotations(ctx, activity, beginNs, endNs);      \
          });                                                                \
    }                                                                        \
    TRACE_EVENT_END(category_literal, track, static_cast<uint64_t>(endNs));  \
  } while (false)

  switch (activity.type()) {
    case ActivityType::CPU_OP:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("cpu_op");
      break;
    case ActivityType::USER_ANNOTATION:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("user_annotation");
      break;
    case ActivityType::GPU_USER_ANNOTATION:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("gpu_user_annotation");
      break;
    case ActivityType::GPU_MEMCPY:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("gpu_memcpy");
      break;
    case ActivityType::GPU_MEMSET:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("gpu_memset");
      break;
    case ActivityType::CONCURRENT_KERNEL:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("kernel");
      break;
    case ActivityType::EXTERNAL_CORRELATION:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("external_correlation");
      break;
    case ActivityType::CUDA_RUNTIME:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("cuda_runtime");
      break;
    case ActivityType::CUDA_DRIVER:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("cuda_driver");
      break;
    case ActivityType::CPU_INSTANT_EVENT:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("cpu_instant_event");
      break;
    case ActivityType::PYTHON_FUNCTION:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("python_function");
      break;
    case ActivityType::OVERHEAD:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("overhead");
      break;
    case ActivityType::MTIA_RUNTIME:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("mtia_runtime");
      break;
    case ActivityType::MTIA_CCP_EVENTS:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("mtia_ccp_events");
      break;
    case ActivityType::MTIA_INSIGHT:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("mtia_insight");
      break;
    case ActivityType::CUDA_SYNC:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("cuda_sync");
      break;
    case ActivityType::CUDA_EVENT:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("cuda_event");
      break;
    case ActivityType::MTIA_COUNTERS:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("mtia_counters");
      break;
    case ActivityType::GLOW_RUNTIME:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("glow_runtime");
      break;
    case ActivityType::CUDA_PROFILER_RANGE:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("cuda_profiler_range");
      break;
    case ActivityType::HPU_OP:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("hpu_op");
      break;
    case ActivityType::XPU_RUNTIME:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("xpu_runtime");
      break;
    case ActivityType::XPU_DRIVER:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("xpu_driver");
      break;
    case ActivityType::COLLECTIVE_COMM:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("collective_comm");
      break;
    case ActivityType::PRIVATEUSE1_RUNTIME:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("privateuse1_runtime");
      break;
    case ActivityType::PRIVATEUSE1_DRIVER:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("privateuse1_driver");
      break;
    case ActivityType::XPU_SCOPE_PROFILER:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("xpu_scope_profiler");
      break;
    case ActivityType::ENUM_COUNT:
      KINETO_TRACE_ACTIVITY_FOR_CATEGORY("user_annotation");
      break;
  }

#undef KINETO_TRACE_ACTIVITY_FOR_CATEGORY
}

const perfetto::Track& PerfettoTraceLogger::trackForActivity(
    const ITraceActivity& activity) {
  const auto device = activity.deviceId();
  const auto resource = activity.resourceId();
  const auto deviceActivity = isDeviceActivity(activity.type());
  auto key = makeTrackKey(deviceActivity ? "device" : "thread", device, resource);
  auto it = tracks_.find(key);
  if (it != tracks_.end()) {
    return *it->second;
  }

  std::ostringstream name;
  if (deviceActivity) {
    name << "Device " << device << " resource " << resource;
  } else {
    name << "Thread " << activity.getThreadId() << " (device " << device
         << ", resource " << resource << ")";
  }

  auto track = std::make_unique<perfetto::Track>(stableTrackId(key));
  auto descriptor = track->Serialize();
  descriptor.set_name(name.str());
  perfetto::TrackEvent::SetTrackDescriptor(*track, descriptor);
  auto emplaceResult = tracks_.emplace(std::move(key), std::move(track));
  return *emplaceResult.first->second;
}

const perfetto::Track& PerfettoTraceLogger::trackForDevice(
    int64_t deviceId,
    std::string name) {
  auto key = fmt::format("device-info:{}", deviceId);
  auto it = tracks_.find(key);
  if (it != tracks_.end()) {
    return *it->second;
  }

  auto track = std::make_unique<perfetto::Track>(stableTrackId(key));
  auto descriptor = track->Serialize();
  descriptor.set_name(std::move(name));
  perfetto::TrackEvent::SetTrackDescriptor(*track, descriptor);
  auto emplaceResult = tracks_.emplace(std::move(key), std::move(track));
  return *emplaceResult.first->second;
}

void PerfettoTraceLogger::finalizeTrace(
    const Config& /*config*/,
    std::unique_ptr<ActivityBuffers> /*buffers*/,
    int64_t /*endTime*/,
    std::unordered_map<std::string, std::vector<std::string>>& /*metadata*/) {
  if (!started_ || finalized_) {
    return;
  }
  writeTraceFile();
}

void PerfettoTraceLogger::finalizeMemoryTrace(
    const std::string&,
    const Config&) {
  LOG(INFO) << "finalizeMemoryTrace not implemented for PerfettoTraceLogger";
}

void PerfettoTraceLogger::emitFlowIfNeeded(const ITraceActivity& /*activity*/) {}

void PerfettoTraceLogger::writeTraceFile() {
  perfetto::TrackEvent::Flush();
  tracingSession_->FlushBlocking();
  tracingSession_->StopBlocking();
  auto traceData = std::vector<char>{tracingSession_->ReadTraceBlocking()};
  if (traceData.empty()) {
    throw std::runtime_error("Perfetto trace contains no data");
  }

  auto out = std::ofstream{fileName_, std::ios::out | std::ios::binary | std::ios::trunc};
  if (!out) {
    throw std::runtime_error(fmt::format("Failed to open {}", fileName_));
  }
  out.write(traceData.data(), static_cast<std::streamsize>(traceData.size()));
  finalized_ = true;
}

} // namespace KINETO_NAMESPACE

#endif // KINETO_ENABLE_PERFETTO
