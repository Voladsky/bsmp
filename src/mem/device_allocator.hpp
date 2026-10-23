#ifndef BSMP_DEVICE_ALLOCATOR_HPP
#define BSMP_DEVICE_ALLOCATOR_HPP

#include <cstddef>
#include <memory>
#include <cstdint>

namespace bsmp {

namespace mem {

class DeviceAllocator {
   public:
    [[nodiscard]] virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual ~DeviceAllocator() = default;
};

class SimpleAllocator : public DeviceAllocator {
   public:
    [[nodiscard]] void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    ~SimpleAllocator() override;
};

class SpanAllocator : public DeviceAllocator {
   public:
    SpanAllocator(size_t capacity);
    [[nodiscard]] void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    ~SpanAllocator() override;

   private:
    uint8_t* d_ptr_;
    size_t capacity_;
    size_t offset_;
};

// One process-wide default; built on first use (no static-init-order fiasco)
// and swappable by tests via the returned reference.
inline std::shared_ptr<DeviceAllocator>& default_allocator() {
    static std::shared_ptr<DeviceAllocator> alloc =
        std::make_shared<SimpleAllocator>();
    return alloc;
}

}  // namespace mem

}  // namespace bsmp
#endif // BSMP_DEVICE_ALLOCATOR_HPP
