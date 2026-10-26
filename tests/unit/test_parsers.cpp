// Unit tests for the triplet/block parser pipeline:
//   element triplets -> parse_coo (HostCOO) -> to_coo (COOFormat::Data)

#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "core/constants.hpp"
#include "core/matrix_formats.hpp"
#include "io/parsers_utilities.hpp"
#include "mem/device_allocator.hpp"
#include "spmv_test_utils.hpp"

namespace spmv {

using bsmp::core::constants::BLOCK_SIZE;
using bsmp::io::parser_utilities::HostCOO;
using bsmp::io::parser_utilities::parse_coo;
using bsmp::io::parser_utilities::to_coo;
using bsmp::mem::DeviceAllocator;
using bsmp::mem::SpanAllocator;

TEST(SpmvParsers, EmptyInputYieldsEmptyMatrix) {
    std::istringstream iss("");
    const HostCOO coo = parse_coo(iss);
    EXPECT_EQ(coo.num_rows, 0);
    EXPECT_EQ(coo.num_cols, 0);
    EXPECT_EQ(coo.block_size, BLOCK_SIZE);
    EXPECT_TRUE(coo.block_rows.empty());
    EXPECT_TRUE(coo.block_cols.empty());
    EXPECT_TRUE(coo.block_data.empty());
}

TEST(SpmvParsers, CommentsAndBlankLinesAreSkipped) {
    std::istringstream iss(
        "# comment line\n"
        "\n"
        "    \n"
        "1 2 3.5\n"
        "# trailing comment\n");
    const HostCOO coo = parse_coo(iss);
    ASSERT_EQ(coo.num_rows, 2);
    ASSERT_EQ(coo.num_cols, 3);
    ASSERT_EQ(coo.block_rows.size(), 1u);
    EXPECT_EQ(coo.block_rows[0], 0);
    EXPECT_EQ(coo.block_cols[0], 0);
}

TEST(SpmvParsers, WhitespaceSeparatedTriplets) {
    std::istringstream iss("0 0 1.0\n1 1 2.0\n2 2 3.0\n");
    const HostCOO coo = parse_coo(iss);
    ASSERT_EQ(coo.num_rows, 3);
    ASSERT_EQ(coo.num_cols, 3);
    ASSERT_EQ(coo.block_rows.size(), 1u);
    ASSERT_EQ(coo.block_data.size(), static_cast<size_t>(BLOCK_SIZE) * BLOCK_SIZE);
    const int bs = BLOCK_SIZE;
    EXPECT_FLOAT_EQ(coo.block_data[0 * bs + 0], 1.0f);
    EXPECT_FLOAT_EQ(coo.block_data[1 * bs + 1], 2.0f);
    EXPECT_FLOAT_EQ(coo.block_data[2 * bs + 2], 3.0f);
}

TEST(SpmvParsers, CommaSeparatedTriplets) {
    std::istringstream iss("0,0,1.5\n1,2,-2.25\n");
    const HostCOO coo = parse_coo(iss);
    ASSERT_EQ(coo.num_rows, 2);
    ASSERT_EQ(coo.num_cols, 3);
    ASSERT_EQ(coo.block_rows.size(), 1u);
    const int bs = BLOCK_SIZE;
    EXPECT_FLOAT_EQ(coo.block_data[0 * bs + 0], 1.5f);
    EXPECT_FLOAT_EQ(coo.block_data[1 * bs + 2], -2.25f);
}

TEST(SpmvParsers, MixedSeparatorsAccepted) {
    // Both forms are documented: "i,j,v" and "i, j, v" (spaces after commas).
    // Duplicate (0,0) entries accumulate in the same block slot.
    std::istringstream iss("0, 0, 1.0\n0,0,2.0\n");
    const HostCOO coo = parse_coo(iss);
    ASSERT_EQ(coo.block_rows.size(), 1u);
    const int bs = BLOCK_SIZE;
    EXPECT_FLOAT_EQ(coo.block_data[0 * bs + 0], 3.0f);
}

TEST(SpmvParsers, DuplicateTripletsAccumulate) {
    std::istringstream iss("0 0 1.0\n0 0 2.0\n0 0 -0.5\n");
    const HostCOO coo = parse_coo(iss);
    ASSERT_EQ(coo.block_rows.size(), 1u);
    EXPECT_FLOAT_EQ(coo.block_data[0], 2.5f);
}

TEST(SpmvParsers, ElementsOutsideFirstBlockCreateMoreBlocks) {
    // BLOCK_SIZE=3: element (3,3) starts a new block row/column pair.
    std::istringstream iss("0 0 5.0\n3 3 7.0\n");
    const HostCOO coo = parse_coo(iss);
    ASSERT_EQ(coo.num_rows, 4);
    ASSERT_EQ(coo.num_cols, 4);
    ASSERT_EQ(coo.block_rows.size(), 2u);
    // std::map orders blocks by (row, col): (0,0) then (1,1).
    EXPECT_EQ(coo.block_rows[0], 0);
    EXPECT_EQ(coo.block_cols[0], 0);
    EXPECT_EQ(coo.block_rows[1], 1);
    EXPECT_EQ(coo.block_cols[1], 1);
    EXPECT_FLOAT_EQ(coo.block_data[0], 5.0f);
    EXPECT_FLOAT_EQ(coo.block_data[0 + 9], 7.0f);
}

TEST(SpmvParsers, BlockReuseAcrossDifferentRows) {
    // Elements (0,3) and (1,4) live in the same block column pair (0,1).
    std::istringstream iss("0 3 1.0\n1 4 2.0\n");
    const HostCOO coo = parse_coo(iss);
    ASSERT_EQ(coo.block_rows.size(), 1u);
    EXPECT_EQ(coo.block_rows[0], 0);
    EXPECT_EQ(coo.block_cols[0], 1);
    const int bs = BLOCK_SIZE;
    EXPECT_FLOAT_EQ(coo.block_data[0 * bs + 0], 1.0f);
    EXPECT_FLOAT_EQ(coo.block_data[1 * bs + 1], 2.0f);
}

TEST(SpmvParsers, NegativeIndicesIgnored) {
    std::istringstream iss("-1 0 5.0\n0 -2 6.0\n0 0 1.0\n");
    const HostCOO coo = parse_coo(iss);
    ASSERT_EQ(coo.num_rows, 1);
    ASSERT_EQ(coo.num_cols, 1);
    ASSERT_EQ(coo.block_rows.size(), 1u);
    EXPECT_FLOAT_EQ(coo.block_data[0], 1.0f);
}

TEST(SpmvParsers, GarbageLinesSkipped) {
    std::istringstream iss("not a triplet\n0 0 1.0\n1.5 2 3\n");
    const HostCOO coo = parse_coo(iss);
    ASSERT_EQ(coo.num_rows, 1);
    ASSERT_EQ(coo.num_cols, 1);
    ASSERT_EQ(coo.block_rows.size(), 1u);
    EXPECT_FLOAT_EQ(coo.block_data[0], 1.0f);
}

TEST(SpmvParsers, ToCooUploadsHostData) {
    HostCOO host{};
    host.num_rows = 2 * BLOCK_SIZE;
    host.num_cols = 2 * BLOCK_SIZE;
    host.block_size = BLOCK_SIZE;
    host.block_rows = {0, 1};
    host.block_cols = {0, 1};
    const int bs = BLOCK_SIZE;
    host.block_data.assign(2 * static_cast<size_t>(bs) * bs, 0.0f);
    host.block_data[0] = 1.0f;
    host.block_data[bs * bs + bs + 1] = 2.0f;

    const auto dev = to_coo(host);

    EXPECT_EQ(dev.nnz, 2);
    EXPECT_EQ(dev.num_rows, 2 * bs);
    EXPECT_EQ(dev.num_cols, 2 * bs);

    std::vector<int> rows(2), cols(2);
    std::vector<float> data(2 * static_cast<size_t>(bs) * bs);
    ASSERT_EQ(cudaMemcpy(rows.data(), dev.block_rows.data(),
                         rows.size() * sizeof(int), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(cols.data(), dev.block_cols.data(),
                         cols.size() * sizeof(int), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(data.data(), dev.block_data.data(),
                         data.size() * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(rows, host.block_rows);
    EXPECT_EQ(cols, host.block_cols);
    EXPECT_EQ(data, host.block_data);
}

TEST(SpmvParsers, ToCooWithInjectedAllocator) {
    HostCOO host{};
    host.num_rows = BLOCK_SIZE;
    host.num_cols = BLOCK_SIZE;
    host.block_size = BLOCK_SIZE;
    host.block_rows = {0};
    host.block_cols = {0};
    const int bs = BLOCK_SIZE;
    host.block_data.assign(static_cast<size_t>(bs) * bs, 0.0f);
    host.block_data[0] = 42.0f;

    auto span = std::make_shared<SpanAllocator>(1 << 20);
    const auto dev = to_coo(host, span);

    std::vector<float> data(static_cast<size_t>(bs) * bs);
    ASSERT_EQ(cudaMemcpy(data.data(), dev.block_data.data(),
                         data.size() * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_FLOAT_EQ(data[0], 42.0f);
}

}  // namespace spmv