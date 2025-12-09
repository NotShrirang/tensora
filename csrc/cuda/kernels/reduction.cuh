#pragma once

#include <cstdint>

namespace tensora {

// Kernel declarations
__global__ void reduce_sum_kernel(const float* input, float* output, int64_t size, int64_t reduce_size);
__global__ void reduce_max_kernel(const float* input, float* output, int64_t size, int64_t reduce_size);

// Wrapper function declarations
void reduce_sum_cuda(const float* input, float* output, int64_t size, int64_t reduce_size);
void reduce_max_cuda(const float* input, float* output, int64_t size, int64_t reduce_size);

} // namespace tensora
