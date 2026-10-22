#ifndef BSMP_DENSE_VECTOR_HPP
#define BSMP_DENSE_VECTOR_HPP

#include <memory>
#include "mem/device_buffer.hpp"
#include "mem/device_allocator.hpp"

namespace bsmp {

namespace core {

class DenseVector {
   public:
    DenseVector(size_t n, std::shared_ptr<mem::DeviceAllocator>);
    DenseVector(DenseVector&&) = default;
    DenseVector(const DenseVector&) = delete;
    DenseVector& operator=(DenseVector&&) = default;
    DenseVector& operator=(const DenseVector&) = delete;
    ~DenseVector() = default;
    [[nodiscard]] float* data();
    [[nodiscard]] const float* data() const;
    [[nodiscard]] size_t size() const;

    struct VectorArgs {
        const float* x;
        float* y;
    };

    static VectorArgs pack(const DenseVector& x, DenseVector& y) noexcept {
        return VectorArgs{x.data(), y.data()};
    }

   private:
    mem::DeviceBuffer<float> data_;
};

}  // namespace core
}  // namespace bsmp

#endif  // BSMP_DENSE_VECTOR_HPP
