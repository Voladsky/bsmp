// Unit tests for the memory layer: SimpleAllocator, SpanAllocator and the
// process-wide default allocator singleton.

#include <cstdint>
#include <new>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "mem/device_allocator.hpp"
#include "spmv_test_utils.hpp"

namespace spmv {

using bsmp::mem::DeviceAllocator;
using bsmp::mem::SimpleAllocator;
using bsmp::mem::SpanAllocator;

TEST(SpmvAllocators, SimpleAllocatorAllocatesUsableMemory) {
    SimpleAllocator alloc;
    void* p1 = alloc.allocate(64);
    void* p2 = alloc.allocate(128);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);
    EXPECT_NE(p1, p2);
    EXPECT_EQ(cudaMemset(p1, 0xAB, 64), cudaSuccess);
    EXPECT_EQ(cudaMemset(p2, 0xCD, 128), cudaSuccess);
    alloc.deallocate(p1);
    alloc.deallocate(p2);
}

TEST(SpmvAllocators, SimpleAllocatorReallocatesAfterFree) {
    SimpleAllocator alloc;
    void* p = alloc.allocate(32);
    ASSERT_NE(p, nullptr);
    alloc.deallocate(p);
    void* q = alloc.allocate(32);
    ASSERT_NE(q, nullptr);
    alloc.deallocate(q);
}

TEST(SpmvAllocators, SpanAllocatorGrowsSequentially) {
    SpanAllocator alloc(1024);
    void* p1 = alloc.allocate(256);
    void* p2 = alloc.allocate(512);
    void* p3 = alloc.allocate(256);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);
    ASSERT_NE(p3, nullptr);
    auto* b1 = static_cast<uint8_t*>(p1);
    auto* b2 = static_cast<uint8_t*>(p2);
    auto* b3 = static_cast<uint8_t*>(p3);
    EXPECT_EQ(b2 - b1, 256);
    EXPECT_EQ(b3 - b2, 512);
    EXPECT_EQ(cudaMemset(p1, 0, 256), cudaSuccess);
}

TEST(SpmvAllocators, SpanAllocatorThrowsWhenExhausted) {
    SpanAllocator alloc(512);
    void* p = alloc.allocate(512);
    ASSERT_NE(p, nullptr);
    EXPECT_THROW((void)alloc.allocate(1), std::bad_alloc);
    alloc.deallocate(p);  // no-op, span offset is unaffected
    EXPECT_THROW((void)alloc.allocate(1), std::bad_alloc);
}

TEST(SpmvAllocators, SpanAllocatorRejectsOversizedRequest) {
    SpanAllocator alloc(64);
    EXPECT_THROW((void)alloc.allocate(65), std::bad_alloc);
}

TEST(SpmvAllocators, DefaultAllocatorIsSingleton) {
    std::shared_ptr<DeviceAllocator>& a = bsmp::mem::default_allocator();
    std::shared_ptr<DeviceAllocator>& b = bsmp::mem::default_allocator();
    EXPECT_EQ(&a, &b);
    EXPECT_NE(a.get(), nullptr);
}

TEST(SpmvAllocators, DefaultAllocatorIsSwappableAndRestorable) {
    std::shared_ptr<DeviceAllocator>& slot = bsmp::mem::default_allocator();
    const std::shared_ptr<DeviceAllocator> original = slot;
    ASSERT_NE(original.get(), nullptr);

    // RAII guard: restore the singleton even if an assertion fires.
    struct RestoreGuard {
        std::shared_ptr<DeviceAllocator>& slot_;
        std::shared_ptr<DeviceAllocator> saved_;
        ~RestoreGuard() { slot_ = std::move(saved_); }
    } guard{slot, original};

    std::shared_ptr<DeviceAllocator> span =
        std::make_shared<SpanAllocator>(1 << 20);
    DeviceAllocator* span_raw = span.get();
    slot.swap(span);  // both shared_ptr<DeviceAllocator>: swaps in the arena
    EXPECT_EQ(bsmp::mem::default_allocator().get(), span_raw);

    // The swapped-in allocator is actually used for device allocations.
    void* p = bsmp::mem::default_allocator()->allocate(128);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(cudaMemset(p, 0, 128), cudaSuccess);
    bsmp::mem::default_allocator()->deallocate(p);

    slot = original;  // restore before guard destructor
    EXPECT_EQ(bsmp::mem::default_allocator().get(), original.get());
}

TEST(SpmvAllocators, DefaultAllocatorServesSimpleAllocator) {
    // The default instantiated allocator must be a SimpleAllocator (full
    // cudaMalloc/free), not an arena.
    auto* simple =
        dynamic_cast<SimpleAllocator*>(bsmp::mem::default_allocator().get());
    EXPECT_NE(simple, nullptr);
}

}  // namespace spmv