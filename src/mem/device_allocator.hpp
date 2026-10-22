#ifndef BSMP_DEVICE_ALLOCATOR_HPP
#define BSMP_DEVICE_ALLOCATOR_HPP

#include <cstddef>

namespace bsmp {

namespace mem {

class DeviceAllocator {
   public:
    [[nodiscard]] virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual ~DeviceAllocator() = 0;
};

class SimpleAllocator : DeviceAllocator {
   public:
    [[nodiscard]] void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    ~SimpleAllocator() override;
};

class SpanAllocator : DeviceAllocator {
   public:
    SpanAllocator(size_t capacity);
    [[nodiscard]] void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    ~SpanAllocator() override;

   private:
    std::byte* d_ptr_;
    size_t capacity_;
    size_t offset_;
};

}  // namespace mem

}  // namespace bsmp
#endif // BSMP_DEVICE_ALLOCATOR_HPP
