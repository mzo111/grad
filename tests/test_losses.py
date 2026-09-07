"""Softmax cross-entropy: value, gradient, and numerical stability."""

from __future__ import annotations

import numpy as np
import pytest

from grad import Tensor, losses
from tests.gradcheck import TOL, check_grad

rng = np.random.default_rng(0)


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_value_matches_naive_reference():
    z = rng.standard_normal((5, 4))
    y = rng.integers(0, 4, size=5)
    loss = losses.softmax_cross_entropy(Tensor(z), y)
    naive = -np.mean(np.log(_softmax(z)[np.arange(5), y]))
    np.testing.assert_allclose(loss.data, naive)
    assert loss.shape == ()


def test_gradient_matches_closed_form():
    z = rng.standard_normal((6, 3))
    y = rng.integers(0, 3, size=6)
    logits = Tensor(z, requires_grad=True)
    losses.softmax_cross_entropy(logits, y).backward()
    onehot = np.eye(3)[y]
    np.testing.assert_allclose(logits.grad, (_softmax(z) - onehot) / 6, atol=1e-12)


def test_gradient_matches_finite_differences():
    z = rng.standard_normal((5, 4))
    y = rng.integers(0, 4, size=5)
    assert check_grad(lambda t: losses.softmax_cross_entropy(t, y), z) < TOL


def test_integer_and_onehot_targets_agree():
    z = rng.standard_normal((4, 3))
    y = np.array([2, 0, 1, 1])
    a = losses.softmax_cross_entropy(Tensor(z), y)
    b = losses.softmax_cross_entropy(Tensor(z), np.eye(3)[y])
    c = losses.softmax_cross_entropy(Tensor(z), Tensor(np.eye(3)[y]))
    np.testing.assert_allclose(a.data, b.data)
    np.testing.assert_allclose(a.data, c.data)


@pytest.mark.parametrize(
    "z",
    [
        np.array([[1e5, 0.0, -1e5], [-1e5, 1e5, 0.0]]),
        np.full((3, 5), 1e5),
        np.array([[1e5, 1e5 - 1.0, 1e5 - 2.0]]),
        np.array([[3e8, -3e8]]),
    ],
)
def test_large_logits_are_finite(z):
    y = np.arange(z.shape[0]) % z.shape[1]
    logits = Tensor(z, requires_grad=True)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        loss = losses.softmax_cross_entropy(logits, y)
        loss.backward()
    assert np.isfinite(loss.data)
    assert np.all(np.isfinite(logits.grad))
    # Sanity: exp(1e5) really does overflow, which is what the shift protects against.
    with np.errstate(over="ignore"):
        assert np.isinf(np.exp(1e5))


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        losses.softmax_cross_entropy(Tensor(np.zeros(3)), np.zeros(3, dtype=int))
    with pytest.raises(ValueError):
        losses.softmax_cross_entropy(Tensor(np.zeros((2, 3))), np.zeros(5, dtype=int))
