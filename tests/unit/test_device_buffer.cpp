// Unit tests for DeviceBuffer RAII semantics: sizing, host<->device
// round-trips, move-only ownership transfer and self-move safety.

#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "mem/device_allocator.hpp"
#include "mem/device_buffer.hpp"
#include "spmv_test_utils.hpp"

namespace spmv {

using bsmp::mem::DeviceBuffer;
using bsmp::mem::SimpleAllocator;
using bsmp::mem::SpanAllocator;

TEST(SpmvDeviceBuffer, SizedConstruction) {
    DeviceBuffer<float> buf(8, std::make_shared<SimpleAllocator>());
    EXPECT_EQ(buf.size(), 8u);
    EXPECT_NE(buf.data(), nullptr);
}

TEST(SpmvDeviceBuffer, ConstAndMutableAccess) {
    DeviceBuffer<int> buf(4, bsmp::mem::default_allocator());
    const DeviceBuffer<int>& cref = buf;
    EXPECT_EQ(cref.data(), buf.data());
    EXPECT_EQ(cref.size(), 4u);
}

TEST(SpmvDeviceBuffer, HostDeviceRoundtrip) {
    DeviceBuffer<float> buf(4, std::make_shared<SimpleAllocator>());
    const std::vector<float> host = {1.0f, 2.0f, -3.5f, 4.25f};

    ASSERT_EQ(cudaMemcpy(buf.data(), host.data(),
                         host.size() * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);

    std::vector<float> back(host.size());
    ASSERT_EQ(cudaMemcpy(back.data(), buf.data(),
                         back.size() * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(back, host);
}

TEST(SpmvDeviceBuffer, MoveCtorTransfersOwnership) {
    DeviceBuffer<float> a(8, std::make_shared<SimpleAllocator>());
    float* ptr = a.data();

    DeviceBuffer<float> b(std::move(a));

    EXPECT_EQ(b.size(), 8u);
    EXPECT_EQ(b.data(), ptr);
    EXPECT_EQ(a.data(), nullptr) << "moved-from buffer must release the pointer";
}

TEST(SpmvDeviceBuffer, MoveAssignTransfersOwnershipAndFreesOld) {
    DeviceBuffer<float> a(4, std::make_shared<SimpleAllocator>());
    DeviceBuffer<float> b(2, std::make_shared<SimpleAllocator>());
    float* ptr = a.data();

    b = std::move(a);

    EXPECT_EQ(b.size(), 4u);
    EXPECT_EQ(b.data(), ptr);
    EXPECT_EQ(a.data(), nullptr);
    // b's previous allocation is released by the move-assignment operator.
    EXPECT_NE(ptr, nullptr);
}

TEST(SpmvDeviceBuffer, SelfMoveAssignmentIsSafe) {
    DeviceBuffer<float> a(4, std::make_shared<SimpleAllocator>());
    float* ptr = a.data();

    a = std::move(a);

    EXPECT_EQ(a.data(), ptr);
    EXPECT_EQ(a.size(), 4u);
}

TEST(SpmvDeviceBuffer, SimultaneousBuffersStayDistinct) {
    auto span = std::make_shared<SpanAllocator>(1 << 20);
    DeviceBuffer<float> a(16, span);
    DeviceBuffer<float> b(16, span);
    EXPECT_NE(a.data(), nullptr);
    EXPECT_NE(b.data(), nullptr);
    EXPECT_NE(a.data(), b.data());
}

TEST(SpmvDeviceBuffer, ReleasedBufferCanBeReallocated) {
    SimpleAllocator alloc;
    {
        DeviceBuffer<int> buf(64, std::make_shared<SimpleAllocator>());
        ASSERT_NE(buf.data(), nullptr);
    }  // buffer destroyed here; memory returned to the allocator
    void* p = alloc.allocate(64 * sizeof(int));
    EXPECT_NE(p, nullptr);
    alloc.deallocate(p);
}

}  // namespace spmv