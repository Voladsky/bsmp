// Unit tests for DeviceWorkspace: named vector reuse and first-request-wins
// semantics (try_emplace does not resize existing entries).

#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "mem/device_allocator.hpp"
#include "mem/device_workspace.hpp"
#include "spmv_test_utils.hpp"

namespace spmv {

using bsmp::mem::DeviceWorkspace;
using bsmp::mem::SpanAllocator;

TEST(SpmvWorkspace, CreatesNamedVector) {
    DeviceWorkspace ws(bsmp::mem::default_allocator());
    auto& v = ws.vector("residual", 16);
    EXPECT_EQ(v.size(), 16u);
    EXPECT_NE(v.data(), nullptr);
}

TEST(SpmvWorkspace, SameNameReturnsSameBuffer) {
    DeviceWorkspace ws(bsmp::mem::default_allocator());
    auto& a = ws.vector("tmp", 16);
    auto& b = ws.vector("tmp", 16);
    EXPECT_EQ(&a, &b);
    EXPECT_EQ(a.data(), b.data());
    EXPECT_EQ(a.size(), 16u);
}

TEST(SpmvWorkspace, FirstRequestWinsSize) {
    // try_emplace: later requests for the same name must not resize.
    DeviceWorkspace ws(bsmp::mem::default_allocator());
    ws.vector("v", 8);
    auto& v = ws.vector("v", 64);
    EXPECT_EQ(v.size(), 8u);
}

TEST(SpmvWorkspace, DistinctNamesGiveDistinctBuffers) {
    DeviceWorkspace ws(bsmp::mem::default_allocator());
    auto& a = ws.vector("a", 16);
    auto& b = ws.vector("b", 16);
    EXPECT_NE(&a, &b);
    EXPECT_NE(a.data(), b.data());
}

TEST(SpmvWorkspace, WorksWithSpanAllocator) {
    auto span = std::make_shared<SpanAllocator>(1 << 20);
    DeviceWorkspace ws(span);
    auto& a = ws.vector("a", 128);
    auto& b = ws.vector("b", 128);
    EXPECT_NE(a.data(), nullptr);
    EXPECT_NE(b.data(), nullptr);
    EXPECT_NE(a.data(), b.data());
}

TEST(SpmvWorkspace, MovePreservesNamedBuffers) {
    DeviceWorkspace ws(bsmp::mem::default_allocator());
    auto& v = ws.vector("x", 8);
    float* ptr = v.data();

    DeviceWorkspace moved(std::move(ws));
    auto& v2 = moved.vector("x", 8);
    EXPECT_EQ(v2.data(), ptr);
    EXPECT_EQ(v2.size(), 8u);
}

}  // namespace spmv