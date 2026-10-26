#ifndef BSMP_MATRIX_FORMAT_CONTRACT_HPP
#define BSMP_MATRIX_FORMAT_CONTRACT_HPP

#include <tuple>
#include <type_traits>
#include <utility>

#include "cuda/spmv.cuh"
#include "dense_vector.hpp"
#include "constants.hpp"

namespace bsmp {

namespace core {

namespace formats {

template <class, class = void>
struct is_format_contract : std::false_type {};

template <class T>
struct is_format_contract<
    T,
    std::void_t<typename T::Data, typename T::KernelArgs, decltype(&T::spmv)>>
    : std::is_same<decltype(&T::spmv),
                   void (*)(const typename T::Data&, const DenseVector&,
                            DenseVector&)> {};

// ----------------------------------------------------------------------------

struct COOFormat {
    struct Data {
        mem::DeviceBuffer<int> block_rows;
        mem::DeviceBuffer<int> block_cols;
        mem::DeviceBuffer<float> block_data;
        int nnz;
        int num_rows;
        int num_cols;
    };
    using KernelArgs = cuda::SpmvCOOArgs;

    static void spmv(const Data& matrix, const DenseVector& x, DenseVector& y) {
        cuda::spmv_coo({matrix.block_rows.data(), matrix.block_cols.data(),
                        matrix.block_data.data(), x.data(), y.data(),
                        matrix.nnz, bsmp::core::constants::BLOCK_SIZE,
                        matrix.num_rows,
                        matrix.num_cols});
        cuda::device_synchronize();
    }

    static_assert(is_format_contract<COOFormat>(),
                  "COOFormat does not satisfy the format contract");
};

}  // namespace formats
}  // namespace core
}  // namespace bsmp

#endif  // BSMP_MATRIX_FORMAT_CONTRACT_HPP
