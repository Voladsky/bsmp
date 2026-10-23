#include "device_allocator.hpp"
#include <cuda_runtime.h>

namespace bsmp {

namespace mem {

void* SimpleAllocator::allocate(size_t bytes) {
    void* ptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

void SimpleAllocator::deallocate(void* ptr) {
    cudaFree(ptr);
}

SimpleAllocator::~SimpleAllocator() {
    // do nothing
}

}  // namespace mem
}  // namespace bsmp
