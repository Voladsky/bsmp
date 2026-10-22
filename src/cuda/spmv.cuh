#ifndef BSMP_SPMV_CUH
#define BSMP_SPMV_CUH

#include <cuda_runtime.h>

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
    int num_col;
};

static void spmv_coo(const SpmvCOOArgs& args) noexcept;

}  // namespace cuda

}  // namespace bsmp

#endif // BSMP_SPMV_CUH
