#include "device_workspace.hpp"

namespace bsmp {
namespace mem {

DeviceWorkspace::DeviceWorkspace(std::shared_ptr<DeviceAllocator> alloc_ptr) : alloc_(alloc_ptr) {}
DeviceBuffer<float>& DeviceWorkspace::vector(std::string name, size_t n) {
    auto [it, inserted] = hash_map_.try_emplace(name, n, alloc_);
    return it->second;
}

}  // namespace mem
}  // namespace bsmp
