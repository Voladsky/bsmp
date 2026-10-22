#ifndef BSMP_DEVICE_WORKSPACE_HPP
#define BSMP_DEVICE_WORKSPACE_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include "device_allocator.hpp"
#include "device_buffer.hpp"

namespace bsmp {
namespace mem {

class DeviceWorkspace {
   public:
    DeviceWorkspace(std::shared_ptr<DeviceAllocator>);
    DeviceBuffer<float>& vector(std::string name, size_t n);

   private:
    std::shared_ptr<DeviceAllocator> alloc_;
    std::unordered_map<std::string, DeviceBuffer<float>> hash_map_;
};

}  // namespace mem
}  // namespace bsmp

#endif
