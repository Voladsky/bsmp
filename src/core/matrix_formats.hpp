#ifndef BSMP_MATRIX_FORMAT_CONTRACT_HPP
#define BSMP_MATRIX_FORMAT_CONTRACT_HPP

#include <tuple>
#include <type_traits>
#include <utility>
#include "dense_vector.hpp"

#include "cuda/spmv.cuh"

namespace bsmp {

namespace core {

namespace formats {

template <class, class = void>
struct is_format_contract : std::false_type {};

template <class T>
struct is_format_contract<
    T, std::void_t<typename T::Buffers, typename T::KernelArgs>>
    : std::true_type {};

// ----------------------------------------------------------------------------

struct COOFormat {
    using Buffers = std::tuple<mem::DeviceBuffer<int>, mem::DeviceBuffer<int>,
                               mem::DeviceBuffer<float>>;
    using Parameters = std::tuple<int, int, int, int>;
    using KernelArgs = bsmp::cuda::SpmvCOOArgs;
    struct MatrixArgs {
        const int* row_indices;
        const int* col_indices;
        const float* values;
        size_t nnz;
    };
    static MatrixArgs pack(const Buffers& buffers) {
        return MatrixArgs{
            std::get<0>(buffers).data(), std::get<1>(buffers).data(),
            std::get<2>(buffers).data(), std::get<2>(buffers).size()};
    };

    static void launch(const MatrixArgs& args,
                       const DenseVector::VectorArgs& vec_args,
                       const Parameters& params) {
        cuda::spmv_coo({args.row_indices, args.col_indices, args.values,
                        vec_args.x, vec_args.y, (int)args.nnz, 16, 1024, 1024});
    }

    static_assert(is_format_contract<COOFormat>(),
                  "COOFormat does not satisfy the format contract");

}  // namespace formats
}  // namespace core
}  // namespace bsmp

#endif  // BSMP_MATRIX_FORMAT_CONTRACT_HPP
