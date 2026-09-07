"""Graph mechanics: requires_grad, accumulation, topological order, detaching."""

from __future__ import annotations

import numpy as np
import pytest

from grad import Tensor
from tests.gradcheck import TOL, check_grad


def test_constructor_coerces_to_float():
    t = Tensor([1, 2, 3])
    assert t.dtype == np.float64
    assert Tensor(np.zeros(2, dtype=np.float32)).dtype == np.float32
    assert Tensor(np.ones(2), dtype=np.float32).dtype == np.float32
    assert Tensor(Tensor([1.0])).data.shape == (1,)


def test_leaf_without_requires_grad_builds_no_graph():
    x = Tensor([1.0, 2.0])
    y = x * 2.0 + 1.0
    assert y._op is None
    assert y._parents == ()
    assert not y.requires_grad
    with pytest.raises(RuntimeError):
        y.backward()
    assert x.grad is None


def test_requires_grad_propagates_and_records_parents():
    x = Tensor([1.0, 2.0], requires_grad=True)
    c = Tensor([3.0, 4.0])
    y = x * c
    assert y.requires_grad
    assert y._op is not None
    assert y._parents == (x, c)


def test_repeated_use_accumulates():
    x = Tensor([1.0, -2.0, 3.0], requires_grad=True)
    (x * x + x).sum().backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 1)

    x.zero_grad()
    (x + x).sum().backward()
    np.testing.assert_allclose(x.grad, 2.0)


def test_diamond_dag_topological_order():
    # c = (2x)(3x) = 6x^2, so dc/dx = 12x. A node visited before all its
    # consumers have contributed would see a partial gradient and get this wrong.
    x = Tensor([0.5, 1.0, -1.5], requires_grad=True)
    a = x * 2.0
    b = x * 3.0
    c = (a * b).sum()
    c.backward()
    np.testing.assert_allclose(x.grad, 12 * x.data)
    np.testing.assert_allclose(a.grad, b.data)
    np.testing.assert_allclose(b.grad, a.data)


def test_deep_chain_does_not_recurse():
    x = Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.backward()
    assert x.grad == 1.0


def test_multiple_backward_calls_accumulate_until_zero_grad():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = (x * 3.0).sum()
    y.backward()
    y.backward()
    np.testing.assert_allclose(x.grad, 6.0)
    x.zero_grad()
    assert x.grad is None
    y.backward()
    np.testing.assert_allclose(x.grad, 3.0)


def test_detach_shares_data_and_cuts_graph():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2.0
    d = y.detach()
    assert d.data is y.data
    assert not d.requires_grad
    assert d._op is None and d._parents == ()

    # Only the un-detached path (x * 3) contributes.
    z = (d * x + x * 3.0).sum()
    z.backward()
    np.testing.assert_allclose(x.grad, d.data + 3.0)
    assert y.grad is None


def test_non_scalar_backward_requires_explicit_grad():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x * 2.0
    with pytest.raises(RuntimeError):
        y.backward()
    with pytest.raises(ValueError):
        y.backward(np.ones(3))
    seed = np.array([[1.0, 0.0], [0.0, -1.0]])
    y.backward(seed)
    np.testing.assert_allclose(x.grad, 2.0 * seed)


def test_single_element_backward_defaults_to_one():
    x = Tensor([[2.0]], requires_grad=True)
    (x * x).backward()
    np.testing.assert_allclose(x.grad, [[4.0]])


def test_constants_receive_no_grad():
    x = Tensor([1.0, 2.0], requires_grad=True)
    c = Tensor([5.0, 6.0])
    y = (x * c + np.array([1.0, 1.0]) + 2.0).sum()
    y.backward()
    np.testing.assert_allclose(x.grad, c.data)
    assert c.grad is None


def test_intermediate_grads_are_recorded():
    x = Tensor([1.0, 2.0], requires_grad=True)
    h = x.exp()
    y = (h * 2.0).sum()
    y.backward()
    np.testing.assert_allclose(h.grad, 2.0)
    np.testing.assert_allclose(y.grad, 1.0)


def test_gradient_seed_is_copied_not_aliased():
    x = Tensor([1.0, 2.0], requires_grad=True)
    seed = np.array([1.0, 1.0])
    (x * 1.0).backward(seed)
    x.grad[0] = 99.0
    assert seed[0] == 1.0


def test_operator_methods_match_functional_forms():
    x = np.random.default_rng(0).uniform(0.5, 2.0, size=(3, 4))

    def via_methods(t):
        return ((t.exp() + t.log()) / (t**2) - t.T.T).sum(axis=0).mean()

    assert check_grad(via_methods, x) < TOL


def test_properties_and_repr():
    t = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    assert t.shape == (2, 3) and t.ndim == 2 and t.size == 6 and len(t) == 2
    assert Tensor(3.5).item() == 3.5
    assert "requires_grad=True" in repr(t)
