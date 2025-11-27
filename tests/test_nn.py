import pytest
import numpy as np
from tensora import nn, Tensor


class TestLayers:
    def test_linear_layer(self):
        layer = nn.Linear(10, 5)
        x = Tensor(np.random.randn(2, 10), dtype='float32')
        y = layer(x)
        assert y.shape == (2, 5)
    
    def test_relu_layer(self):
        layer = nn.ReLU()
        x = Tensor([[-1, 0, 1, 2]], dtype='float32')
        y = layer(x)
        expected = np.array([[0, 0, 1, 2]], dtype='float32')
        np.testing.assert_array_almost_equal(y.numpy(), expected)
    
    def test_sequential(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        x = Tensor(np.random.randn(2, 10), dtype='float32')
        y = model(x)
        assert y.shape == (2, 5)
    
    def test_dropout(self):
        layer = nn.Dropout(p=0.5)
        layer.train()
        x = Tensor(np.ones((100, 100)), dtype='float32')
        y = layer(x)
        # In training mode, some values should be zeroed
        assert not np.allclose(y.numpy(), x.numpy())
        
        layer.eval()
        y_eval = layer(x)
        # In eval mode, output should be same as input
        np.testing.assert_array_almost_equal(y_eval.numpy(), x.numpy())


class TestModule:
    def test_parameters(self):
        model = nn.Linear(10, 5)
        params = list(model.parameters())
        assert len(params) == 2  # weight and bias
    
    def test_named_parameters(self):
        model = nn.Linear(10, 5)
        named_params = dict(model.named_parameters())
        assert 'weight' in named_params
        assert 'bias' in named_params
    
    def test_train_eval_mode(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.Dropout(0.5))
        assert model.training
        model.eval()
        assert not model.training
        model.train()
        assert model.training
