// Core correctness tests for SparseMatrix<COOFormat>: every GPU multiply is
// checked against a host reference build from the same block data.

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "core/matrix_formats.hpp"
#include "core/sparse_matrix.hpp"
#include "io/parsers_utilities.hpp"
#include "mem/device_allocator.hpp"
#include "prof/cpu_time_scope.hpp"
#include "prof/no_prof.hpp"
#include "spmv_test_utils.hpp"

namespace spmv {

using bsmp::core::DenseVector;
using bsmp::core::SparseMatrix;
using bsmp::core::formats::COOFormat;
using bsmp::io::parser_utilities::HostCOO;
using bsmp::io::parser_utilities::to_coo;
using bsmp::prof::CPUProfilingContract;
using bsmp::prof::NoProfilingContract;

using CpuMatrix = SparseMatrix<COOFormat, CPUProfilingContract>;
using BareMatrix = SparseMatrix<COOFormat, NoProfilingContract>;

// Build a matrix from HostCOO block data via the public to_coo path.
inline CpuMatrix make_cpu_matrix(const HostCOO& coo) {
    return CpuMatrix(to_coo(coo));
}

// Run A*x on the device and return the downloaded result. The kernel only
// defines rows [0, num_rows); the buffer is allocated at x.size(), so the
// result is truncated to the meaningful prefix before comparing.
inline Vec run_multiply(const HostCOO& coo, const Vec& x) {
    auto A = make_cpu_matrix(coo);
    DenseVector dx = to_device(x);
    DenseVector dy = A * dx;
    Vec full = from_device(dy);
    full.resize(static_cast<size_t>(coo.num_rows));
    return full;
}

// Input vector sized to the maximum dimension so that both kernel reads
// (x[col < num_cols]) and writes (y[row < num_rows]) stay in bounds.
inline Vec make_input(const HostCOO& coo, uint64_t seed) {
    const size_t n =
        static_cast<size_t>(std::max(coo.num_rows, coo.num_cols));
    Vec x(n);
    Lcg rng(seed);
    for (float& v : x) {
        v = rng.next_float01() * 2.0f - 1.0f;
    }
    return x;
}

// ---------------------------------------------------------------------------
// Exact small cases (integer values, exact float math)
// ---------------------------------------------------------------------------

TEST(SpmvMatrix, SingleIdentityBlock) {
    const HostCOO coo = make_identity_coo(3);
    const Vec x = {1.0f, 2.0f, 3.0f};
    const Vec y = run_multiply(coo, x);
    EXPECT_EQ(y, x);
}

TEST(SpmvMatrix, TwoIdentityBlocks) {
    const HostCOO coo = make_identity_coo(6);
    const Vec x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const Vec y = run_multiply(coo, x);
    EXPECT_EQ(y, x);
}

TEST(SpmvMatrix, KnownBlockMatchesHostReference) {
    HostCOO coo{};
    coo.num_rows = 3;
    coo.num_cols = 3;
    coo.block_size = 3;
    coo.block_rows = {0};
    coo.block_cols = {0};
    // Row-major 3x3 block: [[1,2,3],[4,5,6],[7,8,9]]
    coo.block_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    const Vec x = {1.0f, 1.0f, 1.0f};
    const Vec y = run_multiply(coo, x);

    const Vec expected = {6.0f, 15.0f, 24.0f};  // row sums
    EXPECT_EQ(y, expected);

    Vec ref;
    host_spmv(coo, x, ref);
    EXPECT_EQ(ref, expected);
}

TEST(SpmvMatrix, DiagonalValuesScaleInput) {
    HostCOO coo{};
    coo.num_rows = 3;
    coo.num_cols = 3;
    coo.block_size = 3;
    coo.block_rows = {0};
    coo.block_cols = {0};
    coo.block_data = {2.0f, 0, 0, 0, 3.0f, 0, 0, 0, 4.0f};

    const Vec x = {1.0f, 2.0f, 3.0f};
    const Vec y = run_multiply(coo, x);

    const Vec expected = {2.0f, 6.0f, 12.0f};
    EXPECT_EQ(y, expected);
}

// ---------------------------------------------------------------------------
// Partial / edge blocks and rectangular matrices
// ---------------------------------------------------------------------------

TEST(SpmvMatrix, PartialEdgeBlockSevenBySeven) {
    // 7x7 matrix: last block (2,2) partially covers rows/cols 6..8, only 6
    // is in bounds; kernel must mask rows/cols beyond num_rows/num_cols.
    const HostCOO coo = make_identity_coo(7);
    const Vec x = make_input(coo, 0xABC);
    const Vec y = run_multiply(coo, x);

    Vec ref;
    host_spmv(coo, x, ref);
    expect_vec_close(ref, y, 1e-6);
}

TEST(SpmvMatrix, MatrixSmallerThanBlock) {
    // 1x1 matrix still uses a 3x3 block; all but one row/col are masked.
    HostCOO coo{};
    coo.num_rows = 1;
    coo.num_cols = 1;
    coo.block_size = 3;
    coo.block_rows = {0};
    coo.block_cols = {0};
    coo.block_data = {5.0f, 0, 0, 0, 0, 0, 0, 0, 0};

    const Vec x = {2.0f};
    const Vec y = run_multiply(coo, x);

    const Vec expected = {10.0f};
    EXPECT_EQ(y, expected);
}

TEST(SpmvMatrix, RectangularMatrixWiderThanTall) {
    // 4 rows x 6 cols: block (1,1) covers rows 3..5 / cols 3..5; only row 3
    // is in bounds, all cols in bounds. The block's first row is all ones, so
    // every in-bounds column of that row contributes.
    HostCOO coo{};
    coo.num_rows = 4;
    coo.num_cols = 6;
    coo.block_size = 3;
    coo.block_rows = {1};
    coo.block_cols = {1};
    coo.block_data = {1.0f, 1.0f, 1.0f, 0, 0, 0, 0, 0, 0};

    const Vec x = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    const Vec y = run_multiply(coo, x);

    // y[3] = x[3] + x[4] + x[5] = 6; rows 0..2 stay 0; row 4+ does not exist.
    const Vec expected = {0.0f, 0.0f, 0.0f, 6.0f};
    EXPECT_EQ(y, expected);
}

TEST(SpmvMatrix, MultipleBlocksAccumulateInRow) {
    // Two blocks in the same block row: rows accumulate atomicAdd results.
    HostCOO coo{};
    coo.num_rows = 3;
    coo.num_cols = 6;
    coo.block_size = 3;
    coo.block_rows = {0, 0};
    coo.block_cols = {0, 1};
    coo.block_data = {1.0f, 0, 0, 0, 1.0f, 0, 0, 0, 1.0f,   // block (0,0)
                      2.0f, 0, 0, 0, 2.0f, 0, 0, 0, 2.0f};  // block (0,1)

    const Vec x = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const Vec y = run_multiply(coo, x);

    const Vec expected = {3.0f, 3.0f, 3.0f};  // diag(1)+2*diag(1) = 3
    EXPECT_EQ(y, expected);
}

// ---------------------------------------------------------------------------
// Multiply API contract
// ---------------------------------------------------------------------------

TEST(SpmvMatrix, OverloadedMultiplyRunsDirtyY) {
    // spmv_coo zeroes y internally, so a polluted y is still correct.
    const HostCOO coo = make_identity_coo(3);
    auto A = make_cpu_matrix(coo);

    const Vec x = {1.0f, 2.0f, 3.0f};
    DenseVector dx = to_device(x);
    // Deliberately dirty y: the kernel must zero it before accumulating.
    const std::vector<float> dirty = {123.0f, 123.0f, 123.0f};
    DenseVector dy(dirty.begin(), dirty.end());

    A.multiply(dx, dy);
    const Vec y = from_device(dy);

    EXPECT_EQ(y, x);
}

TEST(SpmvMatrix, OperatorStarEqualsTwoArgMultiply) {
    const HostCOO coo = make_random_coo(24, 0.1, 0x111);
    auto A = make_cpu_matrix(coo);
    const Vec x = make_input(coo, 0x222);

    DenseVector dx = to_device(x);
    DenseVector dy(x.size());
    A.multiply(dx, dy);

    const Vec via_two_arg = from_device(dy);
    const Vec via_star = from_device(A * dx);

    expect_vec_close(via_two_arg, via_star, 1e-6);
}

TEST(SpmvMatrix, InputVectorIsNotModified) {
    const HostCOO coo = make_random_coo(24, 0.1, 0x333);
    auto A = make_cpu_matrix(coo);
    const Vec x = make_input(coo, 0x444);

    DenseVector dx = to_device(x);
    const DenseVector dy = A * dx;

    const Vec x_after = from_device(dx);
    EXPECT_EQ(x_after, x);
    EXPECT_EQ(dy.size(), x.size());
}

TEST(SpmvMatrix, RepeatedMultipliesAreConsistent) {
    const HostCOO coo = make_random_coo(24, 0.1, 0x555);
    auto A = make_cpu_matrix(coo);
    const Vec x = make_input(coo, 0x666);

    const Vec y1 = from_device(A * to_device(x));
    const Vec y2 = from_device(A * to_device(x));

    expect_vec_close(y1, y2, 1e-6);
}

// ---------------------------------------------------------------------------
// Profiling and policy variants
// ---------------------------------------------------------------------------

TEST(SpmvMatrix, CpuProfilingExposesLastMetrics) {
    const HostCOO coo = make_identity_coo(3);
    auto A = make_cpu_matrix(coo);
    const Vec x = {1.0f, 1.0f, 1.0f};

    DenseVector dx = to_device(x);
    DenseVector dy(3);
    A.multiply(dx, dy);

    EXPECT_GE(A.lastMetrics().elapsed_time, 0.0);
}

TEST(SpmvMatrix, NoProfilingVariantStillComputes) {
    // The same math must run with the zero-cost profiler.
    const HostCOO coo = make_random_coo(24, 0.1, 0x777);
    BareMatrix A(to_coo(coo));
    const Vec x = make_input(coo, 0x888);

    DenseVector dx = to_device(x);
    DenseVector dy(x.size());
    A.multiply(dx, dy);

    Vec ref;
    host_spmv(coo, x, ref);
    expect_vec_close(ref, from_device(dy), 1e-5);
}

TEST(SpmvMatrix, MoveSemanticsPreserveFunctionality) {
    const HostCOO coo = make_random_coo(24, 0.1, 0x999);
    auto A = make_cpu_matrix(coo);
    const Vec x = make_input(coo, 0xAAA);

    CpuMatrix B(std::move(A));
    Vec ref;
    host_spmv(coo, x, ref);

    const Vec y = from_device(B * to_device(x));
    expect_vec_close(ref, y, 1e-5);
}

// ---------------------------------------------------------------------------
// Randomized differential testing against the host reference
// ---------------------------------------------------------------------------

TEST(SpmvMatrix, RandomMatrixMatchesHostReference) {
    const HostCOO coo = make_random_coo(64, 0.05, 0xBEEF);
    const Vec x = make_input(coo, 0xCAFE);
    const Vec y = run_multiply(coo, x);

    Vec ref;
    host_spmv(coo, x, ref);
    expect_vec_close(ref, y, 1e-4);
}

TEST(SpmvMatrix, RandomDenseMatrixMatchesHostReference) {
    const HostCOO coo = make_random_coo(32, 0.4, 0xDEAD);
    const Vec x = make_input(coo, 0xFEED);
    const Vec y = run_multiply(coo, x);

    Vec ref;
    host_spmv(coo, x, ref);
    expect_vec_close(ref, y, 1e-4);
}

TEST(SpmvMatrix, RandomRectangularMatrixMatchesHostReference) {
    // 40 rows x 60 cols, deliberately not square.
    const HostCOO coo = make_random_coo(40, 60, 0.03, 0xF00D);
    const Vec x = make_input(coo, 0x1234);
    const Vec y = run_multiply(coo, x);

    Vec ref;
    host_spmv(coo, x, ref);
    expect_vec_close(ref, y, 1e-4);
}

TEST(SpmvMatrix, RandomPrimeSizeMatrixMatchesHostReference) {
    // 47 is prime: no dimension divides the block size, so all edge blocks
    // are partial on every boundary.
    const HostCOO coo = make_random_coo(47, 0.08, 0x5678);
    const Vec x = make_input(coo, 0x9ABC);
    const Vec y = run_multiply(coo, x);

    Vec ref;
    host_spmv(coo, x, ref);
    expect_vec_close(ref, y, 1e-4);
}

// ---------------------------------------------------------------------------
// Allocator injection
// ---------------------------------------------------------------------------

TEST(SpmvMatrix, SpanAllocatorBackedMultiplyWorks) {
    const HostCOO coo = make_random_coo(24, 0.1, 0x1357);
    auto span = std::make_shared<bsmp::mem::SpanAllocator>(1 << 20);

    CpuMatrix A(to_coo(coo, span), span);
    const Vec x = make_input(coo, 0x2468);

    Vec ref;
    host_spmv(coo, x, ref);
    const Vec y = from_device(A * to_device(x));
    expect_vec_close(ref, y, 1e-5);
}

}  // namespace spmv