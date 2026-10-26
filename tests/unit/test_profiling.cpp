// Unit tests for the profiling layer: NoProfilingContract is a zero-cost
// no-op, CPUProfilingContract records wall time into SpmvMetrics.

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "prof/cpu_time_scope.hpp"
#include "prof/no_prof.hpp"
#include "prof/spmv_metrics.hpp"
#include "spmv_test_utils.hpp"

namespace spmv {

using bsmp::prof::CPUProfilingContract;
using bsmp::prof::NoProfilingContract;

TEST(SpmvProfiling, NoProfilingIsDisabledByDefault) {
    EXPECT_EQ(static_cast<int>(NoProfilingContract::enabled), 0);
}

TEST(SpmvProfiling, NoProfilingScopeIsNoop) {
    NoProfilingContract policy;
    auto scope = policy.makeScope();
    EXPECT_NO_THROW(scope.stop());
}

TEST(SpmvProfiling, CpuProfilingIsEnabled) {
    EXPECT_EQ(static_cast<int>(CPUProfilingContract::enabled), 1);
}

TEST(SpmvProfiling, CpuScopeRecordsNonNegativeElapsed) {
    CPUProfilingContract policy;
    auto scope = policy.makeScope();
    scope.stop();
    EXPECT_GE(policy.last().elapsed_time, 0.0);
}

TEST(SpmvProfiling, CpuScopeRecordsPositiveElapsedAfterWait) {
    CPUProfilingContract policy;
    auto scope = policy.makeScope();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    scope.stop();
    EXPECT_GT(policy.last().elapsed_time, 0.0);
}

TEST(SpmvProfiling, ConsecutiveScopesOverwriteMetrics) {
    CPUProfilingContract policy;
    {
        auto scope = policy.makeScope();
        scope.stop();
    }
    const double first = policy.last().elapsed_time;
    {
        auto scope = policy.makeScope();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        scope.stop();
    }
    EXPECT_GE(first, 0.0);
    EXPECT_GE(policy.last().elapsed_time, 0.0);
}

TEST(SpmvProfiling, MetricsFactoryBuildsValidMetrics) {
    // make_metrics currently lives in the global namespace (spmv_metrics.hpp).
    const auto m = make_metrics(1.25);
    EXPECT_DOUBLE_EQ(m.elapsed_time, 1.25);
}

}  // namespace spmv