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
class SparseMatrix : private prof::ProfHolder<Profiler> {
    static_assert(formats::is_format_contract<MatrixFormat>(),
                  "MatrixFormat does not satisfy the format contract");
    static_assert(prof::is_profiling_contract<Profiler>(),
                  "Profiler does not satisfy the profiling contract");

   public:
    SparseMatrix(typename MatrixFormat::Data&& data,
                 std::shared_ptr<mem::DeviceAllocator> alloc =
                     mem::default_allocator())
        : data_(std::move(data)), alloc_(std::move(alloc)) {}
    SparseMatrix(SparseMatrix&&) = default;
    SparseMatrix(const SparseMatrix&) = delete;
    SparseMatrix& operator=(SparseMatrix&&) = default;
    SparseMatrix& operator=(const SparseMatrix&) = delete;
    ~SparseMatrix() = default;

    DenseVector multiply(const DenseVector& x) const {
        DenseVector y(x.size(), alloc_);
        multiply(x, y);
        return y;
    }

    // Pre-zeroed y required (the kernel accumulates with atomicAdd).
    void multiply(const DenseVector& x, DenseVector& y) const { spmv(x, y); }

    DenseVector operator*(const DenseVector& x) const { return multiply(x); }

   private:
    void spmv(const DenseVector& x, DenseVector& y) const {
        [[maybe_unused]] auto scope = this->policy().makeScope();
        MatrixFormat::spmv(data_, x, y);
        if constexpr (this->enabled) {
            scope.stop();
        }
    }

    typename MatrixFormat::Data data_;
    std::shared_ptr<mem::DeviceAllocator> alloc_;
};

}  // namespace core
}  // namespace bsmp

#endif  // BSMP_SPARSE_MATRIX_HPP
