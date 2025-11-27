import pytest
import numpy as np
from tensora import Tensor


class TestTensor:
    def test_tensor_creation(self):
        data = [[1, 2], [3, 4]]
        tensor = Tensor(data)
        assert tensor.shape == (2, 2)
        assert tensor.device == 'cpu'
    
    def test_tensor_addition(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        b = Tensor([[5, 6], [7, 8]], dtype='float32')
        c = a + b
        expected = np.array([[6, 8], [10, 12]], dtype='float32')
        np.testing.assert_array_almost_equal(c.numpy(), expected)
    
    def test_tensor_multiplication(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        b = Tensor([[2, 0], [0, 2]], dtype='float32')
        c = a * b
        expected = np.array([[2, 0], [0, 8]], dtype='float32')
        np.testing.assert_array_almost_equal(c.numpy(), expected)
    
    def test_tensor_matmul(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        b = Tensor([[5, 6], [7, 8]], dtype='float32')
        c = a @ b
        expected = np.array([[19, 22], [43, 50]], dtype='float32')
        np.testing.assert_array_almost_equal(c.numpy(), expected)
    
    def test_tensor_transpose(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        b = a.T
        assert b.shape == (3, 2)
        expected = np.array([[1, 4], [2, 5], [3, 6]], dtype='float32')
        np.testing.assert_array_almost_equal(b.numpy(), expected)
    
    def test_requires_grad(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32', requires_grad=True)
        assert a.requires_grad
        assert a.grad is None
    
    @pytest.mark.skipif(not Tensor([1]).cuda_is_available(), reason="CUDA not available")
    def test_cuda_transfer(self):
        a = Tensor([[1, 2], [3, 4]], dtype='float32')
        b = a.cuda()
        assert b.device == 'cuda'
        c = b.cpu()
        assert c.device == 'cpu'
        np.testing.assert_array_almost_equal(a.numpy(), c.numpy())


class TestGradient:
    def test_simple_addition_backward(self):
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        c = a + b
        c.backward()
        assert a.grad is not None
        assert b.grad is not None
        np.testing.assert_almost_equal(a.grad.numpy(), [1.0])
        np.testing.assert_almost_equal(b.grad.numpy(), [1.0])
    
    def test_multiplication_backward(self):
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        c = a * b
        c.backward()
        np.testing.assert_almost_equal(a.grad.numpy(), [3.0])
        np.testing.assert_almost_equal(b.grad.numpy(), [2.0])
