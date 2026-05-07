// kineto-amdsmi: amd-smi sampling plugin for kineto / torch.profiler
// SPDX-License-Identifier: MIT

#pragma once

#include "kineto_amdsmi/AmdSmiDriver.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

namespace kineto_amdsmi {

// Owns a background sampler thread that periodically polls amd-smi metrics
// and accumulates DeviceSample records into a buffer for later draining.
//
// Lifecycle:
//   AmdSmiSampler s(driver);
//   s.start(period);
//   ... profile runs ...
//   s.stop();
//   auto samples = s.drain();
//
// Thread-safe: drain() can be called only after stop() returns.  start()
// must not be called twice.
class AmdSmiSampler {
 public:
  // The driver pointer must outlive the sampler.
  explicit AmdSmiSampler(AmdSmiDriver* driver);
  ~AmdSmiSampler();

  AmdSmiSampler(const AmdSmiSampler&) = delete;
  AmdSmiSampler& operator=(const AmdSmiSampler&) = delete;

  // Start the polling thread with the given period.  Returns false if the
  // sampler is already running or the driver is not initialized.
  bool start(std::chrono::nanoseconds period);

  // Signal the polling thread to stop and join it.  Idempotent.
  void stop();

  // Take ownership of accumulated samples.  Must be called after stop().
  std::vector<DeviceSample> drain();

  // Approximate count of samples buffered (best-effort, no lock).
  size_t buffered_size_hint() const { return buffered_size_hint_.load(); }

 private:
  void poll_loop(std::chrono::nanoseconds period);

  AmdSmiDriver* driver_ = nullptr;
  std::atomic<bool> stop_flag_{false};
  std::atomic<bool> running_{false};
  std::thread thread_;

  std::mutex mu_;
  std::vector<DeviceSample> samples_;  // guarded by mu_
  std::atomic<size_t> buffered_size_hint_{0};
};

}  // namespace kineto_amdsmi
