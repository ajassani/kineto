// kineto-amdsmi: amd-smi sampling plugin for kineto / torch.profiler
// SPDX-License-Identifier: MIT

#include "AmdSmiSampler.h"

#include <pthread.h>
#include <time.h>

#include <cerrno>
#include <cstdio>

namespace kineto_amdsmi {

namespace {

uint64_t now_ns_for_kineto() {
  // Kineto's ChromeTraceBaseTime uses system_clock (timeSinceEpoch), and
  // it subtracts that base from each event's timestamp via
  // transToRelativeTime.  If we use CLOCK_MONOTONIC here, our values are
  // tens of orders of magnitude smaller than the system_clock base and
  // get clamped to 0.  Use system_clock to match.
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
          .count());
}

}  // namespace

AmdSmiSampler::AmdSmiSampler(AmdSmiDriver* driver) : driver_(driver) {}

AmdSmiSampler::~AmdSmiSampler() {
  stop();
}

bool AmdSmiSampler::start(std::chrono::nanoseconds period) {
  if (running_.load()) return false;
  if (!driver_ || driver_->devices().empty()) return false;

  stop_flag_.store(false);
  running_.store(true);
  thread_ = std::thread(&AmdSmiSampler::poll_loop, this, period);
  return true;
}

void AmdSmiSampler::stop() {
  if (!running_.load()) return;
  stop_flag_.store(true);
  if (thread_.joinable()) thread_.join();
  running_.store(false);
}

std::vector<DeviceSample> AmdSmiSampler::drain() {
  std::lock_guard<std::mutex> g(mu_);
  std::vector<DeviceSample> out;
  out.swap(samples_);
  buffered_size_hint_.store(0);
  return out;
}

void AmdSmiSampler::poll_loop(std::chrono::nanoseconds period) {
  // Name the thread for easier debugging.
  pthread_setname_np(pthread_self(), "amdsmi.sampler");

  using clock = std::chrono::steady_clock;
  auto next = clock::now();

  while (!stop_flag_.load(std::memory_order_acquire)) {
    auto ts = now_ns_for_kineto();

    // Sweep all devices on this tick under one timestamp.  Each device
    // becomes its own DeviceSample; consumers group by device_index.
    std::vector<DeviceSample> tick;
    tick.reserve(driver_->devices().size());
    for (const auto& dev : driver_->devices()) {
      tick.push_back(driver_->sample_device(dev, ts));
    }

    {
      std::lock_guard<std::mutex> g(mu_);
      for (auto& s : tick) {
        samples_.push_back(std::move(s));
      }
      buffered_size_hint_.store(samples_.size(), std::memory_order_relaxed);
    }

    next += period;
    auto now = clock::now();
    if (next < now) {
      // We fell behind -- skip ahead so we don't oscillate.
      next = now + period;
      continue;
    }
    std::this_thread::sleep_until(next);
  }
}

}  // namespace kineto_amdsmi
