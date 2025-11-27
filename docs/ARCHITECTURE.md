# Tensora Architecture

This document provides an overview of Tensora's architecture and design decisions.

## Design Philosophy

Tensora is designed with the following principles:

1. **Performance**: Leverage CUDA for GPU acceleration while maintaining efficient CPU fallbacks
2. **Simplicity**: Clean, intuitive API similar to PyTorch
3. **Modularity**: Easy to extend with new operations and layers
4. **Compatibility**: Works with or without CUDA support

## System Architecture

### Layer 1: C++/CUDA Backend

The lowest layer consists of optimized C++ and CUDA implementations:

```
csrc/
├── tensor_ops.h         # Interface definitions
├── tensor_ops.cpp       # Python bindings with pybind11
├── cpu/                 # CPU implementations
│   └── tensor_cpu.cpp
└── cuda/               # CUDA implementations
    ├── cuda_utils.cuh
    ├── tensor_cuda.cu
    └── kernels/        # Optimized CUDA kernels
        ├── elementwise.cu
        ├── reduction.cu
        └── matmul.cu
```

**Key Design Decisions:**

- Use pybind11 for Python-C++ interop (simpler than raw CPython API)
- Separate CPU and CUDA implementations for conditional compilation
- Kernel-level optimization for performance-critical operations

### Layer 2: Python Tensor API

The middle layer provides the core Tensor abstraction:

```python
class Tensor:
    - Data storage (NumPy array or CUDA pointer)
    - Device management (CPU/CUDA)
    - Automatic differentiation support
    - Operator overloading for intuitive syntax
```

**Key Features:**

- Lazy gradient computation
- Automatic device transfer
- Dynamic computation graph for autograd

### Layer 3: Neural Network Modules

High-level API for building neural networks:

```
tensora/
├── nn/
│   ├── module.py        # Base Module class
│   └── layers.py        # Pre-built layers
├── functional.py        # Functional API
└── optim.py            # Optimizers
```

**Design Patterns:**

- Module pattern for composable layers
- Functional API for stateless operations
- Parameter management and device transfer

## Memory Management

### CPU Memory

- Uses NumPy arrays as backing storage
- Reference counting via Python's GC
- Copy-on-write where possible

### CUDA Memory

- Explicit allocation with `cudaMalloc`
- Manual memory management (freed in destructor)
- Device-to-host transfers on demand

### Future Improvements

- Memory pooling for reduced allocation overhead
- Pinned memory for faster transfers
- Multi-GPU support with peer-to-peer transfers

## Automatic Differentiation

Tensora implements reverse-mode automatic differentiation:

1. **Forward Pass**: Build computation graph

   ```python
   c = a + b  # Records: c depends on a, b via 'add'
   ```

2. **Backward Pass**: Traverse graph in reverse
   ```python
   c.backward()  # Computes gradients for a and b
   ```

**Graph Representation:**

```python
tensor._grad_fn = (op_name, *input_tensors)
```

**Current Limitations:**

- Static graph (rebuild each iteration)
- Limited operation support
- No higher-order gradients

**Planned Enhancements:**

- Dynamic graph with tape-based recording
- Full operator coverage
- Second-order derivatives

## CUDA Kernel Design

### Elementwise Operations

```cuda
// Coalesced memory access pattern
__global__ void add_kernel(float* a, float* b, float* out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}
```

**Characteristics:**

- Simple parallelization
- Good memory coalescing
- Minimal shared memory needed

### Matrix Multiplication

```cuda
// Tiled algorithm with shared memory
__global__ void matmul_kernel_tiled(...) {
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];
    // Load tiles, compute, write result
}
```

**Optimizations:**

- Shared memory for data reuse
- Tiled computation to fit cache
- Tuned block dimensions

### Reduction Operations

```cuda
// Two-phase reduction
__global__ void reduce_sum_kernel(...) {
    // Phase 1: Thread-level reduction
    // Phase 2: Block-level reduction in shared memory
}
```

**Complexity:**

- Requires synchronization
- Bank conflict avoidance
- Recursive kernel launches for large data

## Extension Points

### Adding New Operations

1. **Implement CUDA kernel** (`csrc/cuda/kernels/`)
2. **Add CPU fallback** (`csrc/cpu/`)
3. **Declare in header** (`csrc/tensor_ops.h`)
4. **Create Python binding** (`csrc/tensor_ops.cpp`)
5. **Expose in Tensor API** (`tensora/tensor.py`)

### Adding New Layers

1. **Subclass Module** (`tensora/nn/layers.py`)
2. **Implement forward method**
3. **Register parameters**
4. **Add to exports** (`tensora/nn/__init__.py`)

## Performance Considerations

### CPU Performance

- Use vectorized NumPy operations
- Minimize Python loops
- Consider Numba JIT for critical paths

### CUDA Performance

- Kernel fusion to reduce memory transfers
- Persistent kernels for repeated operations
- Stream-based parallelism for overlap
- Profile with Nsight Compute

### Python Overhead

- Batch operations to amortize call overhead
- Use C++ for performance-critical loops
- Cache frequently computed values

## Testing Strategy

1. **Unit Tests**: Test individual operations
2. **Integration Tests**: Test layer combinations
3. **Gradient Tests**: Numerical gradient checking
4. **Performance Tests**: Benchmarks vs. baseline
5. **CUDA Tests**: Device-specific validation

## Future Roadmap

### Near-term

- [ ] Complete autograd implementation
- [ ] Conv2D and pooling layers
- [ ] More activation functions
- [ ] Learning rate schedulers

### Mid-term

- [ ] Multi-GPU support
- [ ] Mixed precision training
- [ ] JIT compilation for custom ops
- [ ] Model serialization

### Long-term

- [ ] Distributed training
- [ ] Graph optimization
- [ ] Custom CUDA kernel DSL
- [ ] TPU support
