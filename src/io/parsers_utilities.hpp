#ifndef BSMP_PARSERS_UTILITIES_HPP
#define BSMP_PARSERS_UTILITIES_HPP

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/constants.hpp"
#include "core/matrix_formats.hpp"
#include "mem/device_allocator.hpp"
#include "mem/device_buffer.hpp"

namespace bsmp {
namespace io {
namespace parser_utilities {

struct HostCOO {
    int num_rows;    // matrix rows (max_i + 1)
    int num_cols;    // matrix cols (max_j + 1)
    int block_size;  // block edge length
    // nnz is block_rows size;
    std::vector<int> block_rows;  // block-level row index  (i / block_size)
    std::vector<int> block_cols;  // block-level col index  (j / block_size)
    std::vector<float>
        block_data;  // flattened, block_size*block_size per block
};

/// Parse element-level triplets and group them into dense blocks.
/// Lines starting with '#' and blank lines are skipped.
/// Accepts comma- and/or whitespace-separated (i, j, v) lines.
inline HostCOO parse_coo(std::istream& is) {
    int block_size = core::constants::BLOCK_SIZE;
    if (block_size <= 0) {
        throw std::invalid_argument("parse_coo: block_size must be positive");
    }

    HostCOO result{};
    result.block_size = block_size;

    int max_row = -1, max_col = -1;
    std::map<std::pair<int, int>, std::vector<float>> blocks;

    std::string line;
    while (std::getline(is, line)) {
        const size_t p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos) continue;
        if (line[p] == '#') continue;

        long long i = 0, j = 0;
        double v = 0.0;
        std::istringstream ss(line);

        char d1 = 0, d2 = 0;
        bool ok = false;
        if ((ss >> i >> d1 >> j >> d2 >> v) &&
            ((d1 == ',' && (d2 == ',' || d2 == ' ')) ||
             (d1 == ' ' && (d2 == ',' || d2 == ' ')))) {
            ok = true;
        } else {
            ss.clear();
            ss.str(line);
            if (ss >> i >> j >> v) ok = true;
        }
        if (!ok) {
            std::cerr << "Warning: could not parse line: '" << line << "'\n";
            continue;
        }
        if (i < 0 || j < 0) {
            std::cerr << "Warning: negative indices ignored: " << i << "," << j
                      << "\n";
            continue;
        }

        max_row = std::max(max_row, static_cast<int>(i));
        max_col = std::max(max_col, static_cast<int>(j));

        const int brow = static_cast<int>(i) / block_size;
        const int bcol = static_cast<int>(j) / block_size;
        const int local_i = static_cast<int>(i) % block_size;
        const int local_j = static_cast<int>(j) % block_size;

        auto key = std::make_pair(brow, bcol);
        auto it = blocks.find(key);
        if (it == blocks.end()) {
            std::vector<float> data(
                static_cast<size_t>(block_size) * block_size, 0.0f);
            data[static_cast<size_t>(local_i) * block_size + local_j] =
                static_cast<float>(v);
            blocks.emplace(key, std::move(data));
        } else {
            it->second[static_cast<size_t>(local_i) * block_size + local_j] +=
                static_cast<float>(v);
        }
    }

    if (max_row < 0 || max_col < 0) {
        // empty matrix
        return result;  // num_rows = num_cols = 0, vectors empty
    }

    result.num_rows = max_row + 1;
    result.num_cols = max_col + 1;

    result.block_rows.reserve(blocks.size());
    result.block_cols.reserve(blocks.size());
    result.block_data.reserve(blocks.size() * static_cast<size_t>(block_size) *
                              block_size);

    for (const auto& kv : blocks) {
        result.block_rows.push_back(kv.first.first);
        result.block_cols.push_back(kv.first.second);
        result.block_data.insert(result.block_data.end(), kv.second.begin(),
                                 kv.second.end());
    }
    return result;
}

/// Upload a parsed HostCOO to device memory as COOFormat::Data.
inline bsmp::core::formats::COOFormat::Data to_coo(
    const HostCOO& coo,
    std::shared_ptr<mem::DeviceAllocator> alloc = mem::default_allocator()) {
    using COOFormat = bsmp::core::formats::COOFormat;

    const size_t num_blocks = coo.block_rows.size();
    const size_t block_size = static_cast<size_t>(coo.block_size);

    COOFormat::Data device_data{
        mem::DeviceBuffer<int>(num_blocks, alloc),
        mem::DeviceBuffer<int>(num_blocks, alloc),
        mem::DeviceBuffer<float>(num_blocks * block_size * block_size, alloc),
        (int)num_blocks,
        coo.num_rows,
        coo.num_cols};

    if (num_blocks == 0) {
        return device_data;
    }
    cudaError_t err =
        cudaMemcpy(device_data.block_rows.data(), coo.block_rows.data(),
                   num_blocks * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaMemcpy(device_data.block_cols.data(), coo.block_cols.data(),
                     num_blocks * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaMemcpy(device_data.block_data.data(), coo.block_data.data(),
                     coo.block_data.size() * sizeof(float),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return device_data;
}

}  // namespace parser_utilities
}  // namespace io
}  // namespace bsmp

#endif  // BSMP_PARSERS_UTILITIES_HPP
