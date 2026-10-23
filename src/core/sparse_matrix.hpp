#ifndef BSMP_SPARSE_MATRIX_HPP
#define BSMP_SPARSE_MATRIX_HPP

#include <memory>
#include <utility>

#include "dense_vector.hpp"
#include "matrix_formats.hpp"
#include "mem/device_allocator.hpp"
#include "prof/no_prof.hpp"
#include "prof/prof_holder.hpp"

namespace bsmp {

namespace core {

template <typename MatrixFormat,
          typename Profiler = bsmp::prof::NoProfilingContract>
class SparseMatrix : private prof::InstrHolder<Profiler> {
    static_assert(formats::is_format_contract<MatrixFormat>(),
                  "MatrixFormat does not satisfy the format contract");
    static_assert(prof::is_profiling_contract<Profiler>(),
                  "Profiler does not satisfy the profiling contract");

   public:
    SparseMatrix(typename MatrixFormat::Buffers&& buffers,
                 std::shared_ptr<mem::DeviceAllocator> alloc =
                     mem::default_allocator())
        : buffers_(std::move(buffers)), alloc_(std::move(alloc)) {}
    SparseMatrix(SparseMatrix&&) = default;
    SparseMatrix(const SparseMatrix&) = delete;
    SparseMatrix& operator=(SparseMatrix&&) = default;
    SparseMatrix& operator=(const SparseMatrix&) = delete;
    ~SparseMatrix() = default;

    DenseVector multiply(const DenseVector& x) const {
        DenseVector y(x.size(), alloc_);
        // The COO kernel accumulates with atomicAdd: zero the output first.
        cudaMemset(y.data(), 0, y.size() * sizeof(float));
        multiply(x, y);
        return y;
    }

    // Pre-zeroed y required (the kernel accumulates with atomicAdd).
    void multiply(const DenseVector& x, DenseVector& y) const { launch(x, y); }

    DenseVector operator*(const DenseVector& x) const { return multiply(x); }

   private:
    void launch(const DenseVector& x, DenseVector& y) const {
        [[maybe_unused]] auto scope = this->policy().makeScope();
        auto matrix_args = MatrixFormat::pack(this->buffers_);
        auto vector_args = DenseVector::pack(x, y);
        MatrixFormat::launch(matrix_args, vector_args);
        if constexpr (profiler_.enabled) {
            scope.stop();
        }
    }

    Profiler profiler_;
    typename MatrixFormat::Buffers buffers_;
    std::shared_ptr<mem::DeviceAllocator> alloc_;
};

}  // namespace core
}  // namespace bsmp

#endif  // BSMP_SPARSE_MATRIX_HPP
