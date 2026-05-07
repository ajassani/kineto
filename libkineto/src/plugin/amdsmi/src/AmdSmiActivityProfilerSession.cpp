// kineto-amdsmi: amd-smi sampling plugin for kineto / torch.profiler
// SPDX-License-Identifier: MIT

#include "AmdSmiActivityProfilerSession.h"

#include <ActivityType.h>
#include <GenericTraceActivity.h>
#include <libkineto.h>  // CpuTraceBuffer full type for unique_ptr<>
#include <output_base.h>

#include <cstdio>
#include <cstdlib>

namespace kineto_amdsmi {

AmdSmiActivityProfilerSession::AmdSmiActivityProfilerSession(
    AmdSmiDriver* driver,
    std::chrono::nanoseconds period,
    const std::string& name)
    : driver_(driver), period_(period), name_(name) {
  sampler_ = std::make_unique<AmdSmiSampler>(driver_);
}

AmdSmiActivityProfilerSession::~AmdSmiActivityProfilerSession() {
  if (sampler_) sampler_->stop();
}

void AmdSmiActivityProfilerSession::start() {
  if (!sampler_->start(period_)) {
    errors_.push_back(
        "amdsmi sampler failed to start (no devices or already running)");
  }
}

void AmdSmiActivityProfilerSession::stop() {
  sampler_->stop();
}

void AmdSmiActivityProfilerSession::processTrace(
    libkineto::ActivityLogger& logger) {
  auto samples = sampler_->drain();

  // The kineto API in current torch ships does NOT expose a typed counter
  // event mechanism (no ActivityType::MTIA_COUNTERS, no addCounterValue()
  // -- those were added on kineto main but haven't shipped yet).  So we
  // emit each sample as a CPU_INSTANT_EVENT with metric values stashed in
  // metadata.  A small Python post-processor (kineto_amdsmi_postprocess.py)
  // converts these into Chrome counter events ("ph": "C") so they render
  // as proper counter tracks in Perfetto.
  //
  // Recognition is by activityName == "amdsmi.counters".
  std::fprintf(
      stderr,
      "[kineto-amdsmi] processTrace: emitting %zu samples\n",
      samples.size());

  // Allow disabling the metadata emission to bisect heap issues; when
  // KINETO_AMDSMI_NO_METADATA is set, we still emit instant events at
  // the right timestamps but skip addMetadata, which is the only path
  // that calls fmt::format.
  const bool emit_metadata =
      std::getenv("KINETO_AMDSMI_NO_METADATA") == nullptr;

  // Emit one device_info marker per device at the earliest sample time,
  // carrying BDF + product name in metadata.  The post-processor uses these
  // to label each pid in the Chrome trace as e.g. "amd-smi[2] MI300X (0000:97:00.0)".
  if (emit_metadata && !samples.empty()) {
    int64_t t0 = static_cast<int64_t>(samples.front().timestamp_ns);
    for (const auto& d : driver_->devices()) {
      libkineto::GenericTraceActivity info;
      info.activityType = libkineto::ActivityType::CPU_INSTANT_EVENT;
      info.startTime = t0;
      info.endTime = t0;
      info.device = static_cast<int32_t>(d.index);
      info.resource = 0;
      info.activityName = "amdsmi.device_info";
      info.addMetadataQuoted("bdf", d.bdf_string);
      info.addMetadataQuoted("hip_uuid", d.hip_uuid);
      info.addMetadataQuoted("product_name", d.product_name);
      info.addMetadata("amdsmi_index", static_cast<int>(d.index));
      logger.handleGenericActivity(info);
    }
  }

  for (const auto& s : samples) {
    libkineto::GenericTraceActivity ev;
    ev.activityType = libkineto::ActivityType::CPU_INSTANT_EVENT;
    ev.startTime = static_cast<int64_t>(s.timestamp_ns);
    ev.endTime = static_cast<int64_t>(s.timestamp_ns);
    ev.device = static_cast<int32_t>(s.device_index);
    ev.resource = 0;
    ev.activityName = "amdsmi.counters";

    if (emit_metadata) {
      ev.addMetadata("gpu_power_w", s.power_avg_w);
      ev.addMetadata("gpu_power_current_w", s.power_current_w);
      ev.addMetadata("gpu_temp_hotspot_c", s.temp_hotspot_c);
      ev.addMetadata("gpu_temp_edge_c", s.temp_edge_c);
      ev.addMetadata("gpu_temp_mem_c", s.temp_mem_c);
      ev.addMetadata("gpu_gfx_busy_pct", s.gfx_busy_pct);
      ev.addMetadata("gpu_umc_busy_pct", s.umc_busy_pct);
      ev.addMetadata("gpu_mm_busy_pct", s.mm_busy_pct);
      ev.addMetadata("gpu_vram_used_mb", s.vram_used_mb);
    }

    logger.handleGenericActivity(ev);
  }
}

void AmdSmiActivityProfilerSession::processTrace(
    libkineto::ActivityLogger& logger,
    libkineto::getLinkedActivityCallback /*get_linked_activity*/,
    int64_t /*captureWindowStartTime*/,
    int64_t /*captureWindowEndTime*/) {
  processTrace(logger);
}

}  // namespace kineto_amdsmi
