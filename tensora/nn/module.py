"""
Neural network module base classes.
"""

from typing import Iterator, Tuple
from ..tensor import Tensor


class Module:
    """Base class for all neural network modules."""
    
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def forward(self, *args, **kwargs):
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        """Make module callable."""
        return self.forward(*args, **kwargs)
    
    def parameters(self) -> Iterator[Tensor]:
        """Return iterator over module parameters."""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()
    
    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        """Return iterator over module parameters with names."""
        for name, param in self._parameters.items():
            yield name, param
        for module_name, module in self._modules.items():
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param
    
    def train(self, mode: bool = True):
        """Set module to training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self):
        """Set module to evaluation mode."""
        return self.train(False)
    
    def cuda(self):
        """Move all parameters to CUDA."""
        for name, param in self._parameters.items():
            self._parameters[name] = param.cuda()
        for module in self._modules.values():
            module.cuda()
        return self
    
    def cpu(self):
        """Move all parameters to CPU."""
        for name, param in self._parameters.items():
            self._parameters[name] = param.cpu()
        for module in self._modules.values():
            module.cpu()
        return self
    
    def to(self, device: str):
        """Move module to specified device."""
        if device == 'cuda':
            return self.cuda()
        elif device == 'cpu':
            return self.cpu()
        else:
            raise ValueError(f"Unknown device: {device}")
    
    def zero_grad(self):
        """Zero out gradients of all parameters."""
        for param in self.parameters():
            param.grad = None
