#ifndef BSMP_SPARSE_MATRIX_HPP
#define BSMP_SPARSE_MATRIX_HPP

#include <memory>
#include <utility>

#include "dense_vector.hpp"
#include "matrix_formats.hpp"
#include "mem/device_allocator.hpp"
#include "prof/no_prof.hpp"
#include "prof/prof_holder.hpp"
#include "prof/spmv_metrics.hpp"

namespace bsmp {

namespace core {

template <typename MatrixFormat,
          typename Profiler = bsmp::prof::NoProfilingContract>
class SparseMatrix : private prof::ProfHolder<Profiler>,
                     public prof::MetricsAccessor<Profiler> {
    static_assert(formats::is_format_contract<MatrixFormat>(),
                  "MatrixFormat does not satisfy the format contract");
    static_assert(prof::is_profiling_contract<Profiler>(),
                  "Profiler does not satisfy the profiling contract");

   public:
    SparseMatrix(typename MatrixFormat::Data&& data,
                 std::shared_ptr<mem::DeviceAllocator> alloc =
                     mem::default_allocator())
        : prof::ProfHolder<Profiler>(),
          // The accessor points at the policy subobject of *this, which is
          // fully constructed at this point (declaration order).
          prof::MetricsAccessor<Profiler>(&this->policy()),
          data_(std::move(data)),
          alloc_(std::move(alloc)) {}

    // Explicit move: the defaulted one would copy the accessor's raw policy
    // pointer, leaving it dangling at the moved-from object's address.
    SparseMatrix(SparseMatrix&& other) noexcept
        : prof::ProfHolder<Profiler>(std::move(other)),
          prof::MetricsAccessor<Profiler>(&this->policy()),
          data_(std::move(other.data_)),
          alloc_(std::move(other.alloc_)) {}

    SparseMatrix(const SparseMatrix&) = delete;
    // Not noexcept: DeviceBuffer's move-assignment calls the virtual
    // deallocate and is not marked noexcept (same as the original = default).
    SparseMatrix& operator=(SparseMatrix&& other) {
        if (this != &other) {
            static_cast<prof::ProfHolder<Profiler>&>(*this) =
                std::move(other);
            // The policy subobject address is unchanged (it is a base of
            // *this); rewire() keeps the accessor coherent.
            this->rewire(&this->policy());
            data_ = std::move(other.data_);
            alloc_ = std::move(other.alloc_);
        }
        return *this;
    }
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

    // lastMetrics() is inherited from MetricsAccessor and therefore only
    // declared for enabled profilers (no SFINAE on this class's members).

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
