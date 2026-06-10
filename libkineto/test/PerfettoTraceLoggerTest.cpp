/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "ActivityBuffers.h"
#include "ActivityType.h"
#include "Config.h"
#include "GenericTraceActivity.h"
#include "ThreadUtil.h"
#include "TraceSpan.h"
#include "output_perfetto.h"

namespace libkineto {
namespace {

TEST(PerfettoTraceLoggerTest, WritesNonEmptyPftraceFile) {
  auto tracePath = std::filesystem::temp_directory_path() /
      ("kineto_perfetto_logger_test_" + std::to_string(processId()) + ".pftrace");
  std::filesystem::remove(tracePath);

  {
    PerfettoTraceLogger logger(tracePath.string());
    logger.handleTraceStart({{"unit_test", "true"}}, "");

    TraceSpan span(1000, 5000, "Test trace");
    GenericTraceActivity op(span, ActivityType::CPU_OP, "aten::add");
    op.startTime = 1000;
    op.endTime = 3000;
    op.device = processId();
    op.resource = systemThreadId();
    op.threadId = systemThreadId();
    op.id = 7;
    op.addMetadataQuoted("Input type", "float");

    logger.handleActivity(op);

    Config cfg;
    std::unordered_map<std::string, std::vector<std::string>> metadata;
    logger.finalizeTrace(cfg, nullptr, 5000, metadata);
  }

  ASSERT_TRUE(std::filesystem::exists(tracePath));
  EXPECT_GT(std::filesystem::file_size(tracePath), 0);
  std::filesystem::remove(tracePath);
}

} // namespace
} // namespace libkineto
