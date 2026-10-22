#include "dense_vector.hpp"

bsmp::core::DenseVector::DenseVector(size_t n,
    std::shared_ptr<mem::DeviceAllocator> alloc): data_(n, alloc) {}

float* bsmp::core::DenseVector::data() { return data_.data(); }

const float* bsmp::core::DenseVector::data() const { return data_.data(); }

size_t bsmp::core::DenseVector::size() const { return data_.size(); }
