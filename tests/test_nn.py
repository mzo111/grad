"""Module system, Linear, ReLU, Sequential."""

from __future__ import annotations

import numpy as np

from grad import Tensor, nn
from tests.gradcheck import TOL, check_grad


def test_linear_he_init():
    rng = np.random.default_rng(0)
    layer = nn.Linear(1000, 500, rng=rng)
    assert layer.weight.shape == (1000, 500)
    assert layer.bias.shape == (500,)
    assert layer.weight.requires_grad and layer.bias.requires_grad
    expected_std = np.sqrt(2.0 / 1000)
    assert abs(layer.weight.data.std() - expected_std) < 0.1 * expected_std
    assert abs(layer.weight.data.mean()) < 0.01 * expected_std * 10
    np.testing.assert_array_equal(layer.bias.data, 0.0)


def test_linear_is_deterministic_given_rng():
    a = nn.Linear(4, 3, rng=np.random.default_rng(7))
    b = nn.Linear(4, 3, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a.weight.data, b.weight.data)


def test_linear_forward():
    rng = np.random.default_rng(1)
    layer = nn.Linear(4, 3, rng=rng)
    x = rng.standard_normal((5, 4))
    out = layer(Tensor(x))
    np.testing.assert_allclose(out.data, x @ layer.weight.data + layer.bias.data)


def test_relu_module():
    x = Tensor([[-1.0, 0.0, 2.0]])
    np.testing.assert_array_equal(nn.ReLU()(x).data, [[0.0, 0.0, 2.0]])


def test_parameters_are_collected_once_in_order():
    rng = np.random.default_rng(2)
    inner = nn.Sequential(nn.Linear(3, 4, rng=rng), nn.ReLU())
    model = nn.Sequential(inner, nn.Linear(4, 2, rng=rng))
    params = model.parameters()
    expected = [
        inner.layers[0].weight,
        inner.layers[0].bias,
        model.layers[1].weight,
        model.layers[1].bias,
    ]
    assert [id(p) for p in params] == [id(p) for p in expected]
    # A parameter reachable twice (shared attribute) is still reported once.
    model.alias = inner.layers[0].weight
    assert len(model.parameters()) == 4


def test_zero_grad_clears_parameter_grads():
    model = nn.Sequential(nn.Linear(3, 2, rng=np.random.default_rng(0)))
    out = model(Tensor(np.ones((4, 3)))).sum()
    out.backward()
    assert all(p.grad is not None for p in model.parameters())
    model.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_mlp_gradients_match_finite_differences():
    rng = np.random.default_rng(3)
    w1, b1 = rng.standard_normal((4, 5)), rng.standard_normal(5)
    w2, b2 = rng.standard_normal((5, 2)), rng.standard_normal(2)
    x = rng.standard_normal((6, 4))

    def mlp(x, w1, b1, w2, b2):
        model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2))
        model.layers[0].weight, model.layers[0].bias = w1, b1
        model.layers[2].weight, model.layers[2].bias = w2, b2
        return model(x)

    pre = x @ w1 + b1
    assert np.abs(pre).min() > 1e-3, "pre-activations too close to the ReLU kink"
    assert check_grad(mlp, x, w1, b1, w2, b2) < TOL
