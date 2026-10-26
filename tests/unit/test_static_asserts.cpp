// Compile-time contract checks for the bsmp framework.
// These guards fail the build the moment a contract is violated, which is
// the cheapest and earliest possible feedback for a CUDA/HPC codebase.

#include <type_traits>

#include <gtest/gtest.h>

#include "core/constants.hpp"
#include "core/dense_vector.hpp"
#include "core/matrix_formats.hpp"
#include "core/sparse_matrix.hpp"
#include "core/cuda/spmv.cuh"
#include "mem/device_allocator.hpp"
#include "mem/device_buffer.hpp"
#include "prof/cpu_time_scope.hpp"
#include "prof/no_prof.hpp"
#include "prof/prof_contract.hpp"
#include "prof/prof_holder.hpp"

namespace spmv {

using bsmp::core::constants::BLOCK_SIZE;
using bsmp::core::DenseVector;
using bsmp::core::SparseMatrix;
using bsmp::core::formats::COOFormat;
using bsmp::core::formats::is_format_contract;
using bsmp::cuda::SpmvCOOArgs;
using bsmp::mem::DeviceBuffer;
using bsmp::prof::CPUProfilingContract;
using bsmp::prof::is_profiling_contract;
using bsmp::prof::NoProfilingContract;
using bsmp::prof::ProfHolder;

// ---------------------------------------------------------------------------
// Format contract
// ---------------------------------------------------------------------------
static_assert(is_format_contract<COOFormat>::value,
              "COOFormat must satisfy the MatrixFormat contract");
static_assert(!is_format_contract<int>::value,
              "int must not satisfy the format contract");
static_assert(!is_format_contract<void>::value,
              "void must not satisfy the format contract");

struct MissingData {
    using KernelArgs = int;
    static void spmv(int, const DenseVector&, DenseVector&) {}
};
static_assert(!is_format_contract<MissingData>::value,
              "format without a Data type must be rejected");

struct WrongSignature {
    struct Data {};
    using KernelArgs = int;
    static int spmv(const Data&, const DenseVector&, DenseVector&) { return 0; }
};
static_assert(!is_format_contract<WrongSignature>::value,
              "format whose spmv does not return void must be rejected");

struct ValidCustomFormat {
    struct Data {};
    using KernelArgs = int;
    static void spmv(const Data&, const DenseVector&, DenseVector&) {}
};
static_assert(is_format_contract<ValidCustomFormat>::value,
              "any format matching the contract must be accepted");

// COOFormat::Data layout: three device buffers plus metadata.
static_assert(std::is_same_v<decltype(COOFormat::Data::block_rows),
                             DeviceBuffer<int>>,
              "block_rows must be DeviceBuffer<int>");
static_assert(std::is_same_v<decltype(COOFormat::Data::block_cols),
                             DeviceBuffer<int>>,
              "block_cols must be DeviceBuffer<int>");
static_assert(std::is_same_v<decltype(COOFormat::Data::block_data),
                             DeviceBuffer<float>>,
              "block_data must be DeviceBuffer<float>");
static_assert(std::is_same_v<decltype(COOFormat::Data::nnz), int>,
              "nnz must be int");
static_assert(std::is_same_v<decltype(COOFormat::Data::num_rows), int>,
              "num_rows must be int");
static_assert(std::is_same_v<decltype(COOFormat::Data::num_cols), int>,
              "num_cols must be int");

// Kernel argument layout is the ABI bridge to the CUDA kernels.
static_assert(std::is_same_v<decltype(SpmvCOOArgs::block_rows), const int*>,
              "SpmvCOOArgs::block_rows must be const int*");
static_assert(std::is_same_v<decltype(SpmvCOOArgs::block_cols), const int*>,
              "SpmvCOOArgs::block_cols must be const int*");
static_assert(std::is_same_v<decltype(SpmvCOOArgs::block_data), const float*>,
              "SpmvCOOArgs::block_data must be const float*");
static_assert(std::is_same_v<decltype(SpmvCOOArgs::x), const float*>,
              "SpmvCOOArgs::x must be const float*");
static_assert(std::is_same_v<decltype(SpmvCOOArgs::y), float*>,
              "SpmvCOOArgs::y must be float*");
static_assert(std::is_same_v<decltype(SpmvCOOArgs::num_nonzero_blocks), int>,
              "SpmvCOOArgs::num_nonzero_blocks must be int");
static_assert(std::is_same_v<decltype(SpmvCOOArgs::block_size), int>,
              "SpmvCOOArgs::block_size must be int");
static_assert(std::is_same_v<decltype(SpmvCOOArgs::num_rows), int>,
              "SpmvCOOArgs::num_rows must be int");
static_assert(std::is_same_v<decltype(SpmvCOOArgs::num_cols), int>,
              "SpmvCOOArgs::num_cols must be int");
static_assert(std::is_trivially_copyable_v<SpmvCOOArgs>,
              "SpmvCOOArgs must remain trivially copyable for kernel launch");

// ---------------------------------------------------------------------------
// Profiling contract
// ---------------------------------------------------------------------------
static_assert(is_profiling_contract<NoProfilingContract>::value,
              "NoProfilingContract must satisfy the profiling contract");
static_assert(is_profiling_contract<CPUProfilingContract>::value,
              "CPUProfilingContract must satisfy the profiling contract");
static_assert(!is_profiling_contract<int>::value,
              "int must not satisfy the profiling contract");

static_assert(NoProfilingContract::enabled == false,
              "NoProfilingContract must be disabled");
static_assert(CPUProfilingContract::enabled == true,
              "CPUProfilingContract must be enabled");

static_assert(noexcept(std::declval<NoProfilingContract::Scope&>().stop()),
              "NoProfilingContract::Scope::stop must be noexcept");
static_assert(noexcept(std::declval<CPUProfilingContract::Scope&>().stop()),
              "CPUProfilingContract::Scope::stop must be noexcept");

// Policy holder: empty policies must not add any storage (EBO).
static_assert(std::is_empty_v<ProfHolder<NoProfilingContract>>,
              "ProfHolder with an empty policy must be empty (EBO)");
static_assert(!std::is_empty_v<ProfHolder<CPUProfilingContract>>,
              "ProfHolder with a stateful policy must hold the policy object");

// ---------------------------------------------------------------------------
// Resource semantics: everything device-touching is move-only.
// ---------------------------------------------------------------------------
using SpmvMatrixType =
    SparseMatrix<COOFormat, bsmp::prof::NoProfilingContract>;

static_assert(std::is_move_constructible_v<SpmvMatrixType>,
              "SparseMatrix must be move-constructible");
static_assert(std::is_move_assignable_v<SpmvMatrixType>,
              "SparseMatrix must be move-assignable");
static_assert(!std::is_copy_constructible_v<SpmvMatrixType>,
              "SparseMatrix must not be copyable (owns device memory)");
static_assert(!std::is_copy_assignable_v<SpmvMatrixType>,
              "SparseMatrix must not be copy-assignable");

static_assert(std::is_move_constructible_v<DenseVector>,
              "DenseVector must be move-constructible");
static_assert(!std::is_copy_constructible_v<DenseVector>,
              "DenseVector must not be copyable");
static_assert(!std::is_copy_assignable_v<DenseVector>,
              "DenseVector must not be copy-assignable");

static_assert(std::is_move_constructible_v<DeviceBuffer<float>>,
              "DeviceBuffer must be move-constructible");
static_assert(!std::is_copy_constructible_v<DeviceBuffer<float>>,
              "DeviceBuffer must not be copyable");

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static_assert(BLOCK_SIZE > 0, "BLOCK_SIZE must be positive");
static_assert(BLOCK_SIZE == 3, "framework block size is 3 (block-COO kernels)");

// ---------------------------------------------------------------------------
// lastMetrics() availability is gated by the profiler's enabled flag.
// ---------------------------------------------------------------------------
template <typename T, typename = void>
struct has_last_metrics : std::false_type {};
template <typename T>
struct has_last_metrics<T, std::void_t<decltype(std::declval<const T&>().lastMetrics())>>
    : std::true_type {};

using ProfiledMatrix = SparseMatrix<COOFormat, CPUProfilingContract>;
using BareMatrix = SparseMatrix<COOFormat, NoProfilingContract>;

static_assert(has_last_metrics<ProfiledMatrix>::value,
              "enabled profiler must expose lastMetrics()");
static_assert(!has_last_metrics<BareMatrix>::value,
              "disabled profiler must hide lastMetrics()");

// Runtime mirror of the compile-time facts, so the test runner reports them.
TEST(SpmvStaticAsserts, ContractsHoldAtRuntime) {
    static_assert(is_format_contract<COOFormat>::value);
    static_assert(is_profiling_contract<NoProfilingContract>::value);
    static_assert(is_profiling_contract<CPUProfilingContract>::value);
    EXPECT_TRUE(has_last_metrics<ProfiledMatrix>::value);
    EXPECT_FALSE(has_last_metrics<BareMatrix>::value);
    EXPECT_EQ(static_cast<int>(NoProfilingContract::enabled), 0);
    EXPECT_EQ(static_cast<int>(CPUProfilingContract::enabled), 1);
}

TEST(SpmvStaticAsserts, BlockSizeIsSmallPositive) {
    EXPECT_GT(BLOCK_SIZE, 0);
    EXPECT_LE(BLOCK_SIZE, 8);  // kernels assume small dense blocks
}

}  // namespace spmv