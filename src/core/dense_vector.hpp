#ifndef BSMP_DENSE_VECTOR_HPP
#define BSMP_DENSE_VECTOR_HPP

#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>

#include "mem/device_allocator.hpp"
#include "mem/device_buffer.hpp"

namespace bsmp {

namespace core {

class DenseVector {
   public:
    DenseVector(size_t n,
                std::shared_ptr<mem::DeviceAllocator> alloc = mem::default_allocator());
    // Constructs from a contiguous host range [first, last) and uploads it to
    // device memory. Pass std::make_move_iterator(...) to move elements out of
    // the source instead of copying them.
    template <typename InputIt>
    DenseVector(InputIt first, InputIt last,
                std::shared_ptr<mem::DeviceAllocator> alloc = mem::default_allocator())
        : data_(static_cast<size_t>(std::distance(first, last)), std::move(alloc)) {
        if (data_.size() > 0) {
            auto&& first_elem = *first;
            upload_from_host(data_.data(), std::addressof(first_elem), data_.size());
        }
    }
    DenseVector(DenseVector&&) = default;
    DenseVector(const DenseVector&) = delete;
    DenseVector& operator=(DenseVector&&) = default;
    DenseVector& operator=(const DenseVector&) = delete;
    ~DenseVector() = default;
    [[nodiscard]] float* data();
    [[nodiscard]] const float* data() const;
    [[nodiscard]] size_t size() const;
    // Copies size() elements into the host buffer dst (synchronous).
    void copy_to(float* dst) const;

    struct VectorArgs {
        const float* x;
        float* y;
    };

    static inline VectorArgs pack(const DenseVector& x, DenseVector& y) noexcept {
        return VectorArgs{x.data(), y.data()};
    }

   private:
    static void upload_from_host(float* dst, const void* src, size_t count);
    static void download_to_host(float* dst, const float* src, size_t count);

    mem::DeviceBuffer<float> data_;
};

}  // namespace core
}  // namespace bsmp

#endif  // BSMP_DENSE_VECTOR_HPP
