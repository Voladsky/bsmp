#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

#include "spmv.cuh"

namespace bsmp {

namespace cuda {

/// @brief Helper macro for CUDA errors
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl;   \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

__global__ void block_sparse_matvec_kernel(
    const int* block_rows,
    const int* block_cols,
    const float* block_data,
    const float* x,
    float* y,
    int num_nonzero_blocks,
    int block_size,
    int num_rows,
    int num_cols) {
    // Our blocks are small, we place multiple blocks per CUDA block
    int blocks_per_cuda_block = blockDim.x / block_size;
    int local_block_id = threadIdx.x / block_size;
    int local_row = threadIdx.x % block_size;

    int matrix_block_idx = blockIdx.x * blocks_per_cuda_block + local_block_id;
    if (matrix_block_idx >= num_nonzero_blocks)
        return;

    int row_block = block_rows[matrix_block_idx];
    int col_block = block_cols[matrix_block_idx];
    const float* block = block_data + matrix_block_idx * block_size * block_size;

    int global_row = row_block * block_size + local_row;
    if (global_row >= num_rows)
        return;

    // scalar product of the row of the block with vector x
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < block_size; j++) {
        int global_col = col_block * block_size + j;
        if (global_col < num_cols) {
            sum += block[local_row * block_size + j] * x[global_col];
        }
    }

    // Blocks have no overlap in rows or columns
    atomicAdd(&y[global_row], sum);
}

/// @brief Transpose multiplication kernel
__global__ void block_sparse_matvec_transpose_kernel(
    const int* block_rows,
    const int* block_cols,
    const float* block_data,
    const float* x,
    float* y,
    int num_nonzero_blocks,
    int block_size,
    int num_rows,
    int num_cols) {
    int matrix_block_idx = blockIdx.x;
    if (matrix_block_idx >= num_nonzero_blocks)
        return;

    int row_block = block_rows[matrix_block_idx];
    int col_block = block_cols[matrix_block_idx];
    const float* block = block_data + matrix_block_idx * block_size * block_size;

    // Each thread computes one column of the block
    int local_col = threadIdx.x;
    if (local_col >= block_size)
        return;

    int global_col = col_block * block_size + local_col;
    if (global_col >= num_cols)
        return;

    float sum = 0.0f;
    for (int i = 0; i < block_size; i++) {
        int global_row = row_block * block_size + i;
        if (global_row < num_rows) {
            float matrix_val = block[i * block_size + local_col];  // Транспонирование
            float vector_val = x[global_row];
            sum += matrix_val * vector_val;
        }
    }

    atomicAdd(&y[global_col], sum);
}

static void spmv_coo(const SpmvCOOArgs& args) noexcept {
    int threads = 128;
    threads = (threads / config_.block_size) * config_.block_size;
    if (threads < config_.block_size) threads = config_.block_size;
    int matrix_blocks_per_cuda_block = threads / config_.block_size;
    if (matrix_blocks_per_cuda_block < 1) matrix_blocks_per_cuda_block = 1;
    int cuda_blocks =
        (config_.num_nonzero_blocks + matrix_blocks_per_cuda_block - 1) /
        matrix_blocks_per_cuda_block;
    block_sparse_matvec_kernel<<<cuda_blocks, threads, 0, stream>>>(
        args.block_rows,
        args.block_cols,
        args.block_data,
        args.x,
        args.y,
        args.num_nonzero_blocks,
        args.block_size,
        args.num_rows,
        args.num_col;),
        CHECK_CUDA(cudaGetLastError());
}

}  // namespace cuda

}  // namespace bsmp
