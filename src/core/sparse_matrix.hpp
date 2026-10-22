#ifndef BSMP_SPARSE_MATRIX_HPP
#define BSMP_SPARSE_MATRIX_HPP

#include "dense_vector.hpp"
#include "matrix_formats.hpp"
#include "prof/no_prof.hpp"

namespace bsmp {

namespace core {

template <typename MatrixFormat,
          typename Profiler = bsmp::prof::NoProfilingContract>
class SparseMatrix {
    static_assert(formats::is_format_contract<MatrixFormat>(),
                  "MatrixFormat does not satisfy the format contract");
    static_assert(prof::is_profiling_contract<Profiler>(),
                  "Profiler does not satisfy the profiling contract");)
   public:
    SparseMatrix(typename MatrixFormat::Buffers&&) = default;
    SparseMatrix(SparseMatrix&&) = default;
    SparseMatrix(const SparseMatrix&) = delete;
    SparseMatrix& operator=(SparseMatrix&&) = default;
    SparseMatrix& operator=(const SparseMatrix&) = delete;
    ~SparseMatrix() = default;
    DenseVector multiply(const DenseVector& x) const {
        [[maybeunused]] auto scope = profiler_.makeScope();

        if constexpr (Profiler::enabled) {
            scope.stop();
        }
    }
    void multiply(const DenseVector& x, DenseVector& y) const;
    DenseVector operator*(const DenseVector& x) const { return multiply(x); }

   private:
    Profiler profiler_;
    typename MatrixFormat::Buffers buffers_;
    typename MatrixFormat::Parameters parameters_;
};

}  // namespace core
}  // namespace bsmp

#endif  // BSMP_SPARSE_MATRIX_HPP
