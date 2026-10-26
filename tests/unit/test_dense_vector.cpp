// Unit tests for DenseVector: construction (size / host range), ownership
// transfer, copy_to download and the pack() convenience API.

#include <iterator>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "core/dense_vector.hpp"
#include "mem/device_allocator.hpp"
#include "spmv_test_utils.hpp"

namespace spmv {

using bsmp::core::DenseVector;
using bsmp::mem::SpanAllocator;

TEST(SpmvDenseVector, SizedConstruction) {
    DenseVector v(6);
    EXPECT_EQ(v.size(), 6u);
    EXPECT_NE(v.data(), nullptr);
}

TEST(SpmvDenseVector, RangeCtorCopiesHostValues) {
    const std::vector<float> host = {1.0f, 2.5f, -3.25f, 0.0f};
    DenseVector v(host.begin(), host.end());
    ASSERT_EQ(v.size(), host.size());

    Vec back(host.size(), 0.0f);
    v.copy_to(back.data());
    EXPECT_EQ(back, host);
}

TEST(SpmvDenseVector, RangeCtorAcceptsMoveIterators) {
    std::vector<float> host = {4.0f, 5.0f, 6.0f};
    DenseVector v(std::make_move_iterator(host.begin()),
                  std::make_move_iterator(host.end()));
    ASSERT_EQ(v.size(), host.size());

    Vec back(host.size(), 0.0f);
    v.copy_to(back.data());
    EXPECT_EQ(back, std::vector<float>({4.0f, 5.0f, 6.0f}));
}

TEST(SpmvDenseVector, CustomAllocatorInjection) {
    auto span = std::make_shared<SpanAllocator>(1 << 20);
    DenseVector v(4, span);
    EXPECT_EQ(v.size(), 4u);
    EXPECT_NE(v.data(), nullptr);
}

TEST(SpmvDenseVector, CopyToDownloadsAllElements) {
    const std::vector<float> host = {0.5f, -1.5f, 2.75f, 3.125f, -4.0f, 8.0f};
    DenseVector v(host.begin(), host.end());

    Vec dst(v.size(), -1.0f);
    v.copy_to(dst.data());
    EXPECT_EQ(dst, host);
}

TEST(SpmvDenseVector, MoveCtorTransfersOwnership) {
    DenseVector a(6);
    float* ptr = a.data();

    DenseVector b(std::move(a));

    EXPECT_EQ(b.size(), 6u);
    EXPECT_EQ(b.data(), ptr);
    EXPECT_EQ(a.data(), nullptr) << "moved-from vector must release the pointer";
}

TEST(SpmvDenseVector, MoveAssignmentTransfersOwnership) {
    DenseVector a(6);
    DenseVector b(2);
    float* ptr = a.data();

    b = std::move(a);

    EXPECT_EQ(b.size(), 6u);
    EXPECT_EQ(b.data(), ptr);
    EXPECT_EQ(a.data(), nullptr);
}

TEST(SpmvDenseVector, ConstDataPointerMatches) {
    const std::vector<float> host = {1.0f, 2.0f, 3.0f};
    DenseVector v(host.begin(), host.end());

    const DenseVector& cref = v;
    EXPECT_EQ(cref.data(), v.data());
    EXPECT_EQ(cref.size(), v.size());
}

TEST(SpmvDenseVector, PackExposesRawPointers) {
    const std::vector<float> host_x = {1.0f, 2.0f, 3.0f};
    const std::vector<float> host_y = {0.0f, 0.0f, 0.0f};
    DenseVector x(host_x.begin(), host_x.end());
    DenseVector y(host_y.begin(), host_y.end());

    const auto args = DenseVector::pack(x, y);

    EXPECT_EQ(args.x, x.data());
    EXPECT_EQ(args.y, y.data());
}

TEST(SpmvDenseVector, UploadDownloadRoundTrip) {
    Vec host(256);
    Lcg rng(0x5EED);
    for (float& v : host) {
        v = rng.next_float01() * 2.0f - 1.0f;
    }

    DenseVector v(host.begin(), host.end());
    Vec back(host.size(), 0.0f);
    v.copy_to(back.data());

    expect_vec_close(host, back, 1e-7);
}

}  // namespace spmv