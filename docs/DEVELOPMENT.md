# Development Guide

This guide covers the development workflow for Tensora.

## Setting Up Development Environment

1. **Clone the repository:**

```bash
git clone https://github.com/NotShrirang/tensora.git
cd tensora
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies:**

```bash
pip install -e ".[dev]"
```

## Project Structure

```
tensora/
├── csrc/                  # C++ and CUDA source code
│   ├── cuda/             # CUDA-specific code
│   │   ├── kernels/     # CUDA kernel implementations
│   │   │   ├── elementwise.cu
│   │   │   ├── reduction.cu
│   │   │   └── matmul.cu
│   │   ├── cuda_utils.cuh
│   │   └── tensor_cuda.cu
│   ├── cpu/             # CPU-only implementations
│   │   └── tensor_cpu.cpp
│   ├── tensor_ops.h     # Header with declarations
│   └── tensor_ops.cpp   # Main C++ entry point with Python bindings
├── tensora/             # Python package
│   ├── __init__.py
│   ├── tensor.py        # Core Tensor class
│   ├── functional.py    # Functional API
│   ├── optim.py         # Optimizers
│   └── nn/             # Neural network modules
│       ├── __init__.py
│       ├── module.py    # Base Module class
│       └── layers.py    # Layer implementations
├── tests/              # Test suite
├── examples/           # Usage examples
├── docs/              # Documentation
├── setup.py           # Build configuration
├── pyproject.toml     # Project metadata
└── MANIFEST.in        # Package manifest
```

## Building the Extension

### CPU-only build:

```bash
python setup.py build_ext --inplace
```

### With CUDA support:

```bash
CUDA_HOME=/usr/local/cuda python setup.py build_ext --inplace
```

## Running Tests

Run all tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=tensora --cov-report=html
```

Run specific test file:

```bash
pytest tests/test_tensor.py -v
```

## Code Style

Format code with Black:

```bash
black tensora/ tests/
```

Check with flake8:

```bash
flake8 tensora/ tests/
```

Type checking with mypy:

```bash
mypy tensora/
```

## Adding New Features

### 1. Adding a new CUDA kernel

1. Create kernel in `csrc/cuda/kernels/your_kernel.cu`
2. Declare function in `csrc/tensor_ops.h`
3. Add Python binding in `csrc/tensor_ops.cpp`
4. Expose in Python API in `tensora/tensor.py` or `tensora/functional.py`
5. Add tests in `tests/`

### 2. Adding a new layer

1. Create layer class in `tensora/nn/layers.py`
2. Export in `tensora/nn/__init__.py`
3. Add tests in `tests/test_nn.py`
4. Add example in `examples/`

### 3. Adding a new optimizer

1. Create optimizer in `tensora/optim.py`
2. Add tests in `tests/test_optim.py`

## Performance Optimization Tips

### CUDA Kernels:

- Use shared memory for frequently accessed data
- Coalesce global memory accesses
- Minimize divergent branches
- Use appropriate block and grid dimensions
- Profile with `nvprof` or Nsight Compute

### Python Code:

- Minimize Python loops over tensors
- Use vectorized operations when possible
- Cache computed values
- Profile with `cProfile`

## Debugging

### Python debugging:

```python
import pdb; pdb.set_trace()
```

### CUDA debugging:

Use `cuda-gdb` or add debug prints:

```cuda
printf("Debug: value = %f\n", value);
```

### Memory debugging:

Use `cuda-memcheck`:

```bash
cuda-memcheck python your_script.py
```

## Documentation

Build documentation:

```bash
cd docs
make html
```

View docs:

```bash
python -m http.server -d docs/_build/html
```

## Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version bumped in `__init__.py`
- [ ] CHANGELOG updated
- [ ] Code formatted and linted
- [ ] Examples work correctly
- [ ] Build succeeds on all platforms
