#ifndef BSMP_SPMV_CUH
#define BSMP_SPMV_CUH

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace bsmp {

namespace cuda {

struct SpmvCOOArgs {
    const int* block_rows;
    const int* block_cols;
    const float* block_data;
    const float* x;
    float* y;
    int num_nonzero_blocks;
    int block_size;
    int num_rows;
    int num_cols;
};

void spmv_coo(const SpmvCOOArgs& args) noexcept;

// Blocks the host until all previously launched device work completes,
// throwing std::runtime_error on failure.
inline void device_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("device_synchronize: ") + cudaGetErrorString(err));
    }
}

}  // namespace cuda

}  // namespace bsmp

#endif // BSMP_SPMV_CUH
