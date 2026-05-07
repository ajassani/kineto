// kineto-amdsmi: amd-smi sampling plugin for kineto / torch.profiler
// SPDX-License-Identifier: MIT
//
// Registration entry point.  When this shared object is loaded (via
// LD_PRELOAD or the Python ctypes shim), the constructor below registers
// the AmdSmiActivityProfiler as a kineto child activity profiler.

#include "AmdSmiActivityProfiler.h"

#include <libkineto.h>

#include <cstdio>
#include <cstdlib>

namespace {

bool plugin_disabled() {
  const char* s = std::getenv("KINETO_AMDSMI_DISABLE");
  return s != nullptr && s[0] != '\0' && s[0] != '0';
}

[[gnu::constructor]] void kineto_amdsmi_register() {
  if (plugin_disabled()) {
    std::fprintf(
        stderr, "[kineto-amdsmi] disabled via KINETO_AMDSMI_DISABLE\n");
    return;
  }
  std::fprintf(
      stderr, "[kineto-amdsmi] registering with libkineto\n");
  libkineto::api().registerProfilerFactory([]() {
    return std::unique_ptr<libkineto::IActivityProfiler>(
        new kineto_amdsmi::AmdSmiActivityProfiler());
  });
}

}  // namespace
