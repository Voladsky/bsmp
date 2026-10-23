#include <cuda_runtime.h>
#include <stdexcept>

#include "device_allocator.hpp"

namespace bsmp {

namespace mem {

SpanAllocator::SpanAllocator(size_t capacity) : offset_(0), capacity_(capacity) {
    cudaMalloc(&d_ptr_, capacity);
}

void* SpanAllocator::allocate(size_t bytes) {
    if (offset_ + bytes > capacity_) throw std::bad_alloc();
    void* ptr = reinterpret_cast<void*>(d_ptr_ + offset_);
    offset_ += bytes;
    return ptr;
}

void SpanAllocator::deallocate(void* ptr) {
    // do nothing
}

SpanAllocator::~SpanAllocator() {
    cudaFree(d_ptr_);
}

}  // namespace mem
}  // namespace bsmp
