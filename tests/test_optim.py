import pytest
import numpy as np
from tensora import Tensor, optim, nn


class TestOptimizers:
    def test_sgd(self):
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        x = Tensor(np.random.randn(2, 10), dtype='float32')
        y = model(x)
        
        # Simple loss
        loss = Tensor(np.sum(y.numpy() ** 2))
        loss.requires_grad = True
        
        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None
    
    def test_adam(self):
        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        x = Tensor(np.random.randn(2, 10), dtype='float32')
        y = model(x)
        
        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None
    
    def test_sgd_with_momentum(self):
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        assert optimizer.momentum == 0.9
