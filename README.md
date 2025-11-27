# Tensora

**A high-performance tensor computation library with CUDA acceleration, designed for deep learning and numerical computing.**

Built from scratch for deep learning and numerical computing with blazing-fast GPU acceleration.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## ✨ Features

- 🚀 **Pure C++/CUDA Backend**: No PyTorch or NumPy dependencies - truly standalone
- ⚡ **Extreme Performance**: Up to 448x speedup on GPU operations (1024×1024 matmul)
- 🔄 **Complete Autograd**: Full automatic differentiation with computational graph
- 🧠 **PyTorch-like API**: Familiar interface for easy adoption
- 🎯 **Training Ready**: SGD and Adam optimizers with working backpropagation
- 🔧 **Flexible Deployment**: Works with or without CUDA - automatic fallback to CPU

## 🎯 Why Tensora?

Unlike other libraries that wrap PyTorch or depend on NumPy, Tensora is built **completely from scratch**:

- ✅ **Zero heavy dependencies** - Only requires `pybind11` for Python bindings
- ✅ **Production ready** - Complete training pipeline with optimizers and backprop
- ✅ **True CUDA acceleration** - Hand-written kernels, not wrappers
- ✅ **Educational** - Clean, readable codebase perfect for learning DL internals

## 📦 Installation

### Prerequisites

- Python 3.8+
- C++17 compatible compiler (g++, clang++)
- CUDA Toolkit 11.0+ (optional, for GPU support)
- pybind11 (automatically installed)

### Quick Install

```bash
git clone https://github.com/NotShrirang/tensora.git
cd tensora
bash build.sh       # Automatically detects CUDA
pip install -e .
```

### Manual Build

```bash
# CPU only
python setup.py build_ext --inplace

# With CUDA
CUDA_HOME=/usr/local/cuda python setup.py build_ext --inplace
```

## 🚀 Quick Start

### Run the Demo

```bash
python demo.py  # Comprehensive showcase of all features
```

### Basic Tensor Operations

```python
from tensora import Tensor

# Create tensors
a = Tensor([[1.0, 2.0], [3.0, 4.0]])
b = Tensor([[5.0, 6.0], [7.0, 8.0]])

# Arithmetic operations
c = a + b           # Addition
d = a - b           # Subtraction
e = a * b           # Element-wise multiplication
f = a / b           # Division
g = a @ b           # Matrix multiplication

# Tensor properties
print(a.shape)      # (2, 2)
print(a.T)          # Transpose
print(a.device)     # 'cpu' or 'cuda'

# Factory methods
zeros = Tensor.zeros((3, 3))
ones = Tensor.ones((2, 4))
rand = Tensor.randn((5, 5))

# GPU acceleration
if Tensor.cuda_is_available():
    a_gpu = a.cuda()
    b_gpu = b.cuda()
    c_gpu = a_gpu @ b_gpu  # 448x faster on 1024×1024!
    result = c_gpu.cpu()
```

### Automatic Differentiation

```python
from tensora import Tensor

# Create tensors with gradient tracking
x = Tensor([[2.0]], requires_grad=True)
w = Tensor([[3.0]], requires_grad=True)
b = Tensor([[1.0]], requires_grad=True)

# Forward pass
y = w * x + b  # y = 3*2 + 1 = 7

# Backward pass
y.backward()

# Gradients
print(x.grad)  # dy/dx = 3
print(w.grad)  # dy/dw = 2
print(b.grad)  # dy/db = 1
```

### Neural Networks & Training

```python
from tensora import nn, Tensor, optim, functional as F

# Define a model
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
    nn.Sigmoid()
)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    output = model(x_train)
    loss = F.mse_loss(output, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss.tolist()[0]:.4f}')
```

### Functional API

```python
from tensora import functional as F, Tensor

x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

# Activation functions
y1 = F.relu(x)      # [0.0, 0.0, 0.0, 1.0, 2.0]
y2 = F.sigmoid(x)   # [0.119, 0.269, 0.5, 0.731, 0.881]
y3 = F.tanh(x)      # [-0.964, -0.762, 0.0, 0.762, 0.964]
y4 = F.softmax(x, dim=-1)  # Normalized probabilities

# Loss functions
pred = Tensor([[2.0, 1.5, 3.0]])
target = Tensor([[2.5, 2.0, 2.5]])
loss = F.mse_loss(pred, target)  # Mean squared error
```

## Project Structure

```
tensora/
├── csrc/              # C++ and CUDA source code
│   ├── cuda/         # CUDA implementations
│   ├── cpu/          # CPU implementations
│   └── tensor_ops.*  # Core operations
├── tensora/          # Python package
│   ├── tensor.py     # Tensor class
│   ├── nn/          # Neural network modules
│   ├── functional.py # Functional API
│   └── optim.py     # Optimizers
├── tests/           # Test suite
├── examples/        # Usage examples
└── docs/           # Documentation
```

## ⚡ Performance

Tensora uses hand-optimized CUDA kernels for maximum performance:

| Operation       | Matrix Size | CPU Time   | CUDA Time | Speedup    |
| --------------- | ----------- | ---------- | --------- | ---------- |
| Matrix Multiply | 64×64       | 0.09 ms    | 0.02 ms   | **3.8x**   |
| Matrix Multiply | 128×128     | 0.83 ms    | 0.05 ms   | **17.1x**  |
| Matrix Multiply | 1024×1024   | 2382.89 ms | 5.31 ms   | **448.7x** |

### Optimization Techniques

- ✅ **Coalesced memory access** for elementwise operations
- ✅ **Tiled matrix multiplication** with shared memory
- ✅ **Efficient parallel reductions** for sum/max operations
- ✅ **Kernel fusion** to minimize memory transfers

## Documentation

- [Development Guide](docs/DEVELOPMENT.md) - How to contribute and develop
- [Architecture Overview](docs/ARCHITECTURE.md) - System design and internals
- [Examples](examples/) - Code examples and tutorials

## Development

### Setup development environment

```bash
# Clone repository
git clone https://github.com/NotShrirang/tensora.git
cd tensora

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### Run tests

```bash
pytest tests/
```

### Build extension

```bash
python setup.py build_ext --inplace
```

Or use the build script:

```bash
bash build.sh
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Implemented Features

### Core Operations

- [x] Element-wise operations (add, subtract, multiply, divide, sqrt)
- [x] Matrix operations (matmul, transpose)
- [x] Tensor creation (zeros, ones, full, randn)
- [x] Device management (CPU ↔ CUDA transfers)
- [x] Automatic differentiation (complete backpropagation)

### Neural Network Layers

- [x] Linear (fully connected)
- [x] Activation layers (ReLU, Sigmoid, Tanh)
- [x] Sequential container
- [x] Parameter management

### Optimizers

- [x] SGD (with momentum)
- [x] Adam (with bias correction)

### Loss Functions

- [x] Mean Squared Error (MSE)
- [x] Cross Entropy Loss

### Functional API

- [x] Activations (relu, sigmoid, tanh, softmax)
- [x] Loss functions (mse_loss, cross_entropy_loss)
- [x] Linear transformation

## 🗺️ Roadmap

- [ ] Convolution and pooling layers
- [ ] Batch normalization
- [ ] Dropout layer
- [ ] More activation functions (LeakyReLU, GELU, Swish)
- [ ] Additional optimizers (RMSprop, AdamW)
- [ ] Model serialization (save/load)
- [ ] Multi-GPU support
- [ ] Mixed precision training (FP16)
- [ ] Distributed training

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by PyTorch's design and API
- CUDA optimization techniques from various deep learning frameworks
- Community contributions and feedback

## 🎓 Learning Resource

Tensora is an excellent educational tool for understanding:

- Deep learning internals (how PyTorch/TensorFlow work under the hood)
- CUDA programming and GPU optimization
- Automatic differentiation implementation
- Building ML frameworks from scratch
- C++/Python interoperability with pybind11

Check out the [examples/](examples/) directory for tutorials!

## 📄 Citation

If you use Tensora in your research or project, please cite:

```bibtex
@software{tensora2025,
  title = {Tensora: Pure C++/CUDA Tensor Library},
  author = {NotShrirang},
  year = {2025},
  url = {https://github.com/NotShrirang/tensora}
}
```

## 📞 Contact & Support

- **GitHub**: [@NotShrirang](https://github.com/NotShrirang)
- **Issues**: [Report bugs or request features](https://github.com/NotShrirang/tensora/issues)
- **Discussions**: [Ask questions](https://github.com/NotShrirang/tensora/discussions)

## ⭐ Star History

If you find Tensora useful, please consider giving it a star! ⭐

---

**Built with ❤️ by [@NotShrirang](https://github.com/NotShrirang)**
