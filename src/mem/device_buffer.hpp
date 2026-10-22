#ifndef BSMP_DEVICE_BUFFER_HPP
#define BSMP_DEVICE_BUFFER_HPP

#include <memory>

#include "device_allocator.hpp"

namespace bsmp {

namespace mem {

template <typename T>
class DeviceBuffer {
   public:
    DeviceBuffer(size_t n, std::shared_ptr<DeviceAllocator> alloc_ptr) : n_(n), alloc_(alloc_ptr) {
        d_ptr_ = reinterpret_cast<T*>(alloc_->allocate(n_));
    }
    DeviceBuffer(DeviceBuffer&& other) noexcept : d_ptr_(other.d_ptr_), n_(other.n_), alloc_(other.alloc_) {
        other.d_ptr_ = nullptr;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this == &other) return *this;

        if (d_ptr_ != nullptr) {
            alloc_->deallocate(d_ptr_);
        }

        d_ptr_ = other.d_ptr_;
        n_ = other.n_;
        alloc_ = std::move(other.alloc_);

        other.d_ptr_ = nullptr;

        return *this;
    }
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { alloc_->deallocate(d_ptr_); }
    [[nodiscard]] T* data() noexcept { return d_ptr_; };
    [[nodiscard]] const T* data() const noexcept { return d_ptr_; }
    [[nodiscard]] size_t size() const noexcept { return n_; }

   private:
    T* d_ptr_;
    size_t n_;
    std::shared_ptr<DeviceAllocator> alloc_;
};

}  // namespace mem
}  // namespace bsmp

#endif  // BSMP_DEVICE_BUFFER_HPP
