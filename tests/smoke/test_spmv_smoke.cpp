// Integration smoke tests: the full framework pipeline
//   random element triplets -> parse_coo -> to_coo -> SparseMatrix::multiply
// compared against an independent element-level host reference built straight
// from the raw triplets (so parser, storage and kernel bugs all surface).

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "core/matrix_formats.hpp"
#include "core/sparse_matrix.hpp"
#include "core/cuda/spmv.cuh"
#include "io/parsers_utilities.hpp"
#include "prof/cpu_time_scope.hpp"
#include "prof/no_prof.hpp"
#include "spmv_test_utils.hpp"

namespace spmv {

using bsmp::core::DenseVector;
using bsmp::core::SparseMatrix;
using bsmp::core::formats::COOFormat;
using bsmp::io::parser_utilities::parse_coo;
using bsmp::io::parser_utilities::to_coo;
using bsmp::prof::CPUProfilingContract;
using bsmp::prof::NoProfilingContract;

using Matrix = SparseMatrix<COOFormat, CPUProfilingContract>;
using BareMatrix = SparseMatrix<COOFormat, NoProfilingContract>;

namespace {

struct RawTriplet {
    int row;
    int col;
    float value;
};

struct PipelineResult {
    std::vector<RawTriplet> triplets;
    std::vector<float> x;
    std::vector<float> y_ref;  // element-level reference
    std::vector<float> y_gpu;  // downloaded kernel result
    double relative_error = 0.0;
    size_t num_blocks = 0;
};

// Generate a random triplet problem, run it through the whole framework
// pipeline and compute the reference on the host at element level.
PipelineResult run_pipeline(int rows, int cols, double density, uint64_t seed,
                            bool use_cpu_profiler = true) {
    PipelineResult out;
    Lcg rng(seed);
    std::ostringstream oss;

    const int num_triplets = static_cast<int>(density * rows * cols);
    out.triplets.reserve(static_cast<size_t>(num_triplets));
    for (int k = 0; k < num_triplets; ++k) {
        const int i = rng.next_int(0, rows);
        const int j = rng.next_int(0, cols);
        const float v = rng.next_float01() * 2.0f - 1.0f;
        out.triplets.push_back({i, j, v});
        oss << i << ' ' << j << ' ' << v << '\n';
    }

    // x must be sized to the maximum dimension: the kernel reads
    // x[col < num_cols] and the result buffer is allocated at x.size() while
    // num_rows rows are written.
    const int x_len = std::max(rows, cols);
    out.x.resize(static_cast<size_t>(x_len));
    for (float& v : out.x) {
        v = rng.next_float01() * 2.0f - 1.0f;
    }
    out.y_ref.assign(static_cast<size_t>(rows), 0.0f);
    for (const RawTriplet& t : out.triplets) {
        out.y_ref[static_cast<size_t>(t.row)] +=
            t.value * out.x[static_cast<size_t>(t.col)];
    }

    // Framework pipeline.
    std::istringstream iss(oss.str());
    const auto coo = parse_coo(iss);
    out.num_blocks = coo.block_rows.size();
    const int parsed_rows = coo.num_rows;  // may be < nominal rows (sparse)
    out.y_ref.resize(static_cast<size_t>(parsed_rows));

    COOFormat::Data device = to_coo(coo);
    DenseVector x(out.x.begin(), out.x.end());

    if (use_cpu_profiler) {
        Matrix A(std::move(device));
        DenseVector y = A * x;
        out.y_gpu.resize(y.size());  // y is sized x.size() == x_len
        y.copy_to(out.y_gpu.data());
        EXPECT_GE(A.lastMetrics().elapsed_time, 0.0);
    } else {
        BareMatrix A(std::move(device));
        DenseVector y = A * x;
        out.y_gpu.resize(y.size());
        y.copy_to(out.y_gpu.data());
    }

    // Only rows [0, num_rows) are defined by the kernel; everything beyond is
    // uninitialized device memory and must be dropped before comparing.
    out.y_gpu.resize(static_cast<size_t>(parsed_rows));

    double num = 0.0, den = 0.0;
    for (int i = 0; i < parsed_rows; ++i) {
        const double d = static_cast<double>(out.y_ref[static_cast<size_t>(i)]) -
                         static_cast<double>(out.y_gpu[static_cast<size_t>(i)]);
        num += d * d;
        den += static_cast<double>(out.y_ref[static_cast<size_t>(i)]) *
               static_cast<double>(out.y_ref[static_cast<size_t>(i)]);
    }
    out.relative_error = std::sqrt(num / (den > 0.0 ? den : 1.0));
    return out;
}

}  // namespace

TEST(SpmvSmoke, SquareMatrixPipelineMatchesElementReference) {
    const auto result = run_pipeline(512, 512, 0.01, 0x5EED);
    EXPECT_GT(result.triplets.size(), 0u);
    EXPECT_GT(result.num_blocks, 0u);
    EXPECT_LT(result.relative_error, 1e-4);
    std::cout << "[smoke] square 512x512 triplets=" << result.triplets.size()
              << " blocks=" << result.num_blocks
              << " rel_err=" << result.relative_error << '\n';
}

TEST(SpmvSmoke, RectangularMatrixPipelineMatchesElementReference) {
    // 256 rows x 448 cols: exercises partial edge blocks on both axes.
    const auto result = run_pipeline(256, 448, 0.02, 0xABCD);
    EXPECT_GT(result.num_blocks, 0u);
    EXPECT_LT(result.relative_error, 1e-4);
    std::cout << "[smoke] rect 256x448 blocks=" << result.num_blocks
              << " rel_err=" << result.relative_error << '\n';
}

TEST(SpmvSmoke, PrimeSizeMatrixPipelineMatchesElementReference) {
    // 97 is prime: every edge block is partial in at least one dimension.
    const auto result = run_pipeline(97, 97, 0.05, 0x1234);
    EXPECT_GT(result.num_blocks, 0u);
    EXPECT_LT(result.relative_error, 1e-4);
}

TEST(SpmvSmoke, NoProfilingVariantMatchesElementReference) {
    const auto result = run_pipeline(128, 128, 0.05, 0x7777,
                                     /*use_cpu_profiler=*/false);
    EXPECT_GT(result.num_blocks, 0u);
    EXPECT_LT(result.relative_error, 1e-4);
}

TEST(SpmvSmoke, DeviceSynchronizeCompletesAfterWork) {
    EXPECT_NO_THROW(bsmp::cuda::device_synchronize());
    const auto result = run_pipeline(64, 64, 0.1, 0x99);
    EXPECT_LT(result.relative_error, 1e-4);
    EXPECT_NO_THROW(bsmp::cuda::device_synchronize());
}

TEST(SpmvSmoke, DeterministicAcrossRuns) {
    const auto a = run_pipeline(128, 128, 0.04, 0x4242);
    const auto b = run_pipeline(128, 128, 0.04, 0x4242);
    // Host reference is bitwise reproducible; the GPU result is stable well
    // within float accumulation tolerance (atomicAdd ordering may vary).
    EXPECT_EQ(a.y_ref, b.y_ref);
    expect_vec_close(a.y_gpu, b.y_gpu, 5e-5);
    EXPECT_NEAR(a.relative_error, b.relative_error, 1e-6);
}

}  // namespace spmv