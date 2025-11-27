#include "../cuda/cuda_utils.cuh"
#include "../tensor_ops.h"

namespace tensora {
namespace cuda {

__global__ void add_kernel(const float* a, const float* b, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_kernel(const float* a, const float* b, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_kernel(const float* a, const float* b, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_kernel(const float* a, const float* b, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void relu_kernel(const float* input, float* output, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void sigmoid_kernel(const float* input, float* output, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void tanh_kernel(const float* input, float* output, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void sqrt_kernel(const float* input, float* output, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = sqrtf(input[idx]);
    }
}

} // namespace cuda

// Host-side wrappers
void add_cuda(const float* a, const float* b, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::add_kernel<<<grid, block>>>(a, b, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sub_cuda(const float* a, const float* b, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::sub_kernel<<<grid, block>>>(a, b, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void mul_cuda(const float* a, const float* b, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::mul_kernel<<<grid, block>>>(a, b, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void div_cuda(const float* a, const float* b, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::div_kernel<<<grid, block>>>(a, b, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void relu_cuda(const float* in, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::relu_kernel<<<grid, block>>>(in, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sigmoid_cuda(const float* in, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::sigmoid_kernel<<<grid, block>>>(in, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void tanh_cuda(const float* in, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::tanh_kernel<<<grid, block>>>(in, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sqrt_cuda(const float* in, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::sqrt_kernel<<<grid, block>>>(in, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace tensora
