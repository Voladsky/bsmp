#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "dense_vector.hpp"

bsmp::core::DenseVector::DenseVector(size_t n,
    std::shared_ptr<mem::DeviceAllocator> alloc): data_(n, alloc) {}

void bsmp::core::DenseVector::upload_from_host(float* dst, const void* src, size_t count) {
    cudaError_t err =
        cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("DenseVector upload: ") + cudaGetErrorString(err));
    }
}

void bsmp::core::DenseVector::download_to_host(float* dst, const float* src, size_t count) {
    cudaError_t err =
        cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("DenseVector download: ") + cudaGetErrorString(err));
    }
}

void bsmp::core::DenseVector::copy_to(float* dst) const {
    download_to_host(dst, data_.data(), data_.size());
}

float* bsmp::core::DenseVector::data() { return data_.data(); }

const float* bsmp::core::DenseVector::data() const { return data_.data(); }

size_t bsmp::core::DenseVector::size() const { return data_.size(); }
