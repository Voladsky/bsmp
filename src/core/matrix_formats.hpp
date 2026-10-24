#ifndef BSMP_MATRIX_FORMAT_CONTRACT_HPP
#define BSMP_MATRIX_FORMAT_CONTRACT_HPP

#include <tuple>
#include <type_traits>
#include <utility>

#include "cuda/spmv.cuh"
#include "dense_vector.hpp"

namespace bsmp {

namespace core {

namespace formats {

template <class, class = void>
struct is_format_contract : std::false_type {};

template <class T>
struct is_format_contract<
    T, std::void_t<typename T::Buffers, typename T::KernelArgs,
                   decltype(&T::launch)>>
    : std::is_same<decltype(&T::launch),
                   void (*)(const typename T::MatrixArgs&,
                            const DenseVector::VectorArgs&)> {};

// ----------------------------------------------------------------------------

struct COOFormat {
    using Buffers = std::tuple<mem::DeviceBuffer<int>, mem::DeviceBuffer<int>,
                               mem::DeviceBuffer<float>>;
    using KernelArgs = cuda::SpmvCOOArgs;
    struct MatrixArgs {
        const int* row_indices;
        const int* col_indices;
        const float* values;
        size_t nnz;
    };
    static inline MatrixArgs pack(const Buffers& buffers) {
        return MatrixArgs{
            std::get<0>(buffers).data(), std::get<1>(buffers).data(),
            std::get<2>(buffers).data(), std::get<0>(buffers).size()};
    };

    static void launch(const MatrixArgs& args,
                       const DenseVector::VectorArgs& vec_args) {
        cuda::spmv_coo({args.row_indices, args.col_indices, args.values,
                        vec_args.x, vec_args.y, (int)args.nnz, core::constants::BLOCK_SIZE, 1024, 1024});
        cuda::device_synchronize();
    }

    static_assert(is_format_contract<COOFormat>(),
                  "COOFormat does not satisfy the format contract");
};

}  // namespace formats
}  // namespace core
}  // namespace bsmp

#endif  // BSMP_MATRIX_FORMAT_CONTRACT_HPP
