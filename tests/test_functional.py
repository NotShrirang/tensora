import pytest
import numpy as np
from tensora import functional as F, Tensor


class TestFunctional:
    def test_relu(self):
        x = Tensor([[-1, 0, 1, 2]], dtype='float32')
        y = F.relu(x)
        expected = np.array([[0, 0, 1, 2]], dtype='float32')
        np.testing.assert_array_almost_equal(y.numpy(), expected)
    
    def test_sigmoid(self):
        x = Tensor([[0, 1, -1]], dtype='float32')
        y = F.sigmoid(x)
        expected = 1 / (1 + np.exp(-np.array([[0, 1, -1]], dtype='float32')))
        np.testing.assert_array_almost_equal(y.numpy(), expected, decimal=5)
    
    def test_tanh(self):
        x = Tensor([[0, 1, -1]], dtype='float32')
        y = F.tanh(x)
        expected = np.tanh(np.array([[0, 1, -1]], dtype='float32'))
        np.testing.assert_array_almost_equal(y.numpy(), expected, decimal=5)
    
    def test_softmax(self):
        x = Tensor([[1, 2, 3]], dtype='float32')
        y = F.softmax(x, dim=-1)
        # Softmax should sum to 1
        assert np.abs(np.sum(y.numpy()) - 1.0) < 1e-5
    
    def test_linear(self):
        x = Tensor(np.random.randn(2, 10), dtype='float32')
        weight = Tensor(np.random.randn(5, 10), dtype='float32')
        bias = Tensor(np.random.randn(5), dtype='float32')
        y = F.linear(x, weight, bias)
        assert y.shape == (2, 5)
    
    def test_mse_loss(self):
        pred = Tensor([[1, 2, 3]], dtype='float32')
        target = Tensor([[1, 2, 3]], dtype='float32')
        loss = F.mse_loss(pred, target)
        assert np.abs(loss.numpy()) < 1e-5
