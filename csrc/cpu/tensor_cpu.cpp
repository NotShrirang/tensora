#include "../tensor_ops.h"
#include <cmath>
#include <algorithm>

namespace tensora
{

    void add_cpu(const float *a, const float *b, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = a[i] + b[i];
        }
    }

    void broadcasting_add_cpu(const float *a, const float *b, float *out, 
                              const std::vector<int64_t> &shape_a, 
                              const std::vector<int64_t> &shape_b,
                              const std::vector<int64_t> &shape_out)
    {
        int64_t ndim_out = shape_out.size();
        int64_t ndim_a = shape_a.size();
        int64_t ndim_b = shape_b.size();

        std::vector<int64_t> stride_a_orig(ndim_a, 1);
        for (int64_t i = ndim_a - 2; i >= 0; --i) {
            stride_a_orig[i] = stride_a_orig[i + 1] * shape_a[i + 1];
        }
        
        std::vector<int64_t> stride_b_orig(ndim_b, 1);
        for (int64_t i = ndim_b - 2; i >= 0; --i) {
            stride_b_orig[i] = stride_b_orig[i + 1] * shape_b[i + 1];
        }

        std::vector<int64_t> stride_out(ndim_out, 1);
        for (int64_t i = ndim_out - 2; i >= 0; --i) {
            stride_out[i] = stride_out[i + 1] * shape_out[i + 1];
        }

        std::vector<int64_t> stride_a(ndim_out, 0);
        for (int64_t i = 0; i < ndim_out; ++i) {
            int64_t idx_a = i - (ndim_out - ndim_a);
            if (idx_a >= 0 && idx_a < ndim_a) {
                if (shape_a[idx_a] == shape_out[i]) {
                    stride_a[i] = stride_a_orig[idx_a];
                } else if (shape_a[idx_a] == 1) {
                    stride_a[i] = 0;
                }
            }
        }
        
        std::vector<int64_t> stride_b(ndim_out, 0);
        for (int64_t i = 0; i < ndim_out; ++i) {
            int64_t idx_b = i - (ndim_out - ndim_b);
            if (idx_b >= 0 && idx_b < ndim_b) {
                if (shape_b[idx_b] == shape_out[i]) {
                    stride_b[i] = stride_b_orig[idx_b];
                } else if (shape_b[idx_b] == 1) {
                    stride_b[i] = 0;
                }
            }
        }

        int64_t size_out = 1;
        for (auto dim : shape_out) {
            size_out *= dim;
        }
        
        for (int64_t idx = 0; idx < size_out; ++idx) {
            int64_t idx_a = 0;
            int64_t idx_b = 0;
            int64_t remaining = idx;
            
            for (int64_t i = 0; i < ndim_out; ++i) {
                int64_t coord = remaining / stride_out[i];
                remaining = remaining % stride_out[i];
                
                idx_a += coord * stride_a[i];
                idx_b += coord * stride_b[i];
            }
            
            out[idx] = a[idx_a] + b[idx_b];
        }
    }

    void sub_cpu(const float *a, const float *b, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = a[i] - b[i];
        }
    }

    void mul_cpu(const float *a, const float *b, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = a[i] * b[i];
        }
    }

    void div_cpu(const float *a, const float *b, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = a[i] / b[i];
        }
    }

    // Matrix multiplication: C = A @ B
    // A: (m x k), B: (k x n), C: (m x n)
    void matmul_cpu(const float *a, const float *b, float *out,
                    int64_t m, int64_t n, int64_t k)
    {
        for (int64_t i = 0; i < m; ++i)
        {
            for (int64_t j = 0; j < n; ++j)
            {
                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p)
                {
                    sum += a[i * k + p] * b[p * n + j];
                }
                out[i * n + j] = sum;
            }
        }
    }

    // Activation functions
    void relu_cpu(const float *in, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = std::max(0.0f, in[i]);
        }
    }

    void sigmoid_cpu(const float *in, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = 1.0f / (1.0f + std::exp(-in[i]));
        }
    }

    void tanh_cpu(const float *in, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = std::tanh(in[i]);
        }
    }

    void sqrt_cpu(const float *in, float *out, int64_t size)
    {
        for (int64_t i = 0; i < size; ++i)
        {
            out[i] = std::sqrt(in[i]);
        }
    }

} // namespace tensora
