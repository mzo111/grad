"""Forward-value and finite-difference gradient checks for every op."""

from __future__ import annotations

import numpy as np
import pytest

import grad
from grad import Tensor
from tests.gradcheck import TOL, check_grad

rng = np.random.default_rng(1234)


def rand(shape, low=-1.0, high=1.0):
    return rng.uniform(low, high, size=shape)


def positive(shape):
    return rand(shape, 0.5, 2.0)


# Broadcast pairs: same shape, scalar + matrix, row/column vector + matrix, two 3-D cases.
BROADCAST_SHAPES = [
    ((3, 4), (3, 4)),
    ((), (3, 4)),
    ((1, 4), (3, 4)),
    ((3, 1), (3, 4)),
    ((4,), (3, 4)),
    ((2, 1, 4), (1, 3, 4)),
    ((3, 4), (2, 3, 4)),
]
# Include both argument orders so broadcasting is exercised on either side.
BROADCAST_CASES = BROADCAST_SHAPES + [(b, a) for a, b in BROADCAST_SHAPES if a != b]

BINARY_OPS = {
    "add": (grad.add, np.add, rand, rand),
    "sub": (grad.sub, np.subtract, rand, rand),
    "mul": (grad.mul, np.multiply, rand, rand),
    "div": (grad.div, np.divide, rand, positive),
    "pow": (grad.pow, np.power, positive, rand),
}


@pytest.mark.parametrize("name", BINARY_OPS)
@pytest.mark.parametrize(("sa", "sb"), BROADCAST_CASES)
def test_binary_forward(name, sa, sb):
    op, ref, gen_a, gen_b = BINARY_OPS[name]
    a, b = gen_a(sa), gen_b(sb)
    out = op(Tensor(a), Tensor(b))
    np.testing.assert_allclose(out.data, ref(a, b))


@pytest.mark.parametrize("name", BINARY_OPS)
@pytest.mark.parametrize(("sa", "sb"), BROADCAST_CASES)
def test_binary_grad(name, sa, sb):
    op, _, gen_a, gen_b = BINARY_OPS[name]
    assert check_grad(op, gen_a(sa), gen_b(sb)) < TOL


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: x + 2.0,
        lambda x: 2.0 + x,
        lambda x: x - 2.0,
        lambda x: 2.0 - x,
        lambda x: x * 3.0,
        lambda x: 3.0 * x,
        lambda x: x / 2.0,
        lambda x: 2.0 / x,
        lambda x: x**3,
        lambda x: x**0.5,
        lambda x: 2.0**x,
        lambda x: x + np.ones(4),
    ],
)
def test_python_scalar_and_numpy_operands(fn):
    assert check_grad(fn, positive((3, 4))) < TOL


def test_unbroadcast_helper():
    g = np.ones((2, 3, 4))
    assert grad.unbroadcast(g, (2, 3, 4)).shape == (2, 3, 4)
    np.testing.assert_array_equal(grad.unbroadcast(g, (3, 4)), np.full((3, 4), 2.0))
    np.testing.assert_array_equal(grad.unbroadcast(g, (1, 4)), np.full((1, 4), 6.0))
    np.testing.assert_array_equal(grad.unbroadcast(g, (2, 1, 1)), np.full((2, 1, 1), 12.0))
    np.testing.assert_array_equal(grad.unbroadcast(g, ()), np.array(24.0))
    assert grad.unbroadcast(g, ()).shape == ()


# ------------------------------------------------------------------ unary


@pytest.mark.parametrize(
    ("op", "ref", "gen"),
    [
        (grad.neg, np.negative, rand),
        (grad.exp, np.exp, rand),
        (grad.log, np.log, positive),
    ],
)
def test_unary(op, ref, gen):
    x = gen((2, 3, 4))
    np.testing.assert_allclose(op(Tensor(x)).data, ref(x))
    assert check_grad(op, x) < TOL


# ------------------------------------------------------------------ matmul


MATMUL_SHAPES = [
    ((3, 4), (4, 5)),
    ((2, 3, 4), (4, 5)),
    ((3, 4), (2, 4, 5)),
    ((2, 3, 4), (2, 4, 5)),
    ((2, 3, 4), (1, 4, 5)),
    ((1, 3, 4), (2, 4, 5)),
    ((2, 1, 3, 4), (3, 4, 5)),
    ((4,), (4, 5)),
    ((3, 4), (4,)),
    ((4,), (4,)),
    ((4,), (2, 4, 5)),
    ((2, 3, 4), (4,)),
]


@pytest.mark.parametrize(("sa", "sb"), MATMUL_SHAPES)
def test_matmul(sa, sb):
    a, b = rand(sa), rand(sb)
    out = grad.matmul(Tensor(a), Tensor(b))
    np.testing.assert_allclose(out.data, a @ b)
    assert check_grad(grad.matmul, a, b) < TOL


def test_matmul_rejects_scalars():
    with pytest.raises(ValueError):
        grad.matmul(Tensor(1.0), Tensor(np.ones(3)))


# --------------------------------------------------------------- reductions


REDUCE_CASES = [
    (None, False),
    (None, True),
    (0, False),
    (1, True),
    (-1, False),
    ((0, 2), False),
    ((0, 2), True),
    ((1, -1), True),
]


@pytest.mark.parametrize(("axis", "keepdims"), REDUCE_CASES)
@pytest.mark.parametrize(
    ("op", "ref"),
    [(grad.sum, np.sum), (grad.mean, np.mean), (grad.max, np.max)],
)
def test_reductions(op, ref, axis, keepdims):
    x = rand((2, 3, 4))
    out = op(Tensor(x), axis=axis, keepdims=keepdims)
    expected = ref(x, axis=axis, keepdims=keepdims)
    assert out.shape == np.shape(expected)
    np.testing.assert_allclose(out.data, expected)
    assert check_grad(lambda t: op(t, axis=axis, keepdims=keepdims), x) < TOL


def test_max_splits_gradient_among_ties():
    x = Tensor([[1.0, 3.0, 3.0], [5.0, 2.0, 5.0]], requires_grad=True)
    x.max(axis=1).sum().backward()
    np.testing.assert_allclose(x.grad, [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])


def test_reduction_rejects_bad_axis():
    with pytest.raises(np.exceptions.AxisError):
        grad.sum(Tensor(np.ones((2, 3))), axis=2)
    with pytest.raises(ValueError):
        grad.sum(Tensor(np.ones((2, 3))), axis=(0, 0))


# ----------------------------------------------------------------- shape ops


@pytest.mark.parametrize(
    ("in_shape", "new_shape"),
    [((2, 6), (3, 4)), ((2, 3, 4), (24,)), ((2, 3, 4), (6, -1)), ((12,), (2, 2, 3))],
)
def test_reshape(in_shape, new_shape):
    x = rand(in_shape)
    out = grad.reshape(Tensor(x), new_shape)
    np.testing.assert_array_equal(out.data, x.reshape(new_shape))
    assert check_grad(lambda t: grad.reshape(t, new_shape), x) < TOL
    assert check_grad(lambda t: t.reshape(*new_shape), x) < TOL


@pytest.mark.parametrize("axes", [None, (2, 0, 1), (1, 0, 2), (0, 2, 1), (-1, 0, 1)])
def test_transpose(axes):
    x = rand((2, 3, 4))
    out = grad.transpose(Tensor(x), axes)
    np.testing.assert_array_equal(out.data, x.transpose(axes))
    assert check_grad(lambda t: grad.transpose(t, axes), x) < TOL


def test_transpose_T_property():
    x = rand((3, 4))
    np.testing.assert_array_equal(Tensor(x).T.data, x.T)
    assert check_grad(lambda t: t.T, x) < TOL


def test_transpose_rejects_bad_axes():
    with pytest.raises(ValueError):
        grad.transpose(Tensor(np.ones((2, 3, 4))), (0, 1))


# --------------------------------------------------------------------- relu


def test_relu():
    x = rand((2, 3, 4))
    # Finite differences straddling the kink at 0 are meaningless, so keep every
    # input at least 1e-3 away from it.
    x = np.where(np.abs(x) < 1e-3, 1e-3, x)
    np.testing.assert_array_equal(grad.relu(Tensor(x)).data, np.maximum(x, 0))
    assert check_grad(grad.relu, x) < TOL


def test_relu_gradient_at_zero_is_zero():
    x = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    grad.relu(x).sum().backward()
    np.testing.assert_array_equal(x.grad, [0.0, 0.0, 1.0])


# ------------------------------------------------- logsumexp / stack / slice


@pytest.mark.parametrize(("axis", "keepdims"), REDUCE_CASES)
def test_logsumexp(axis, keepdims):
    x = rand((2, 3, 4), -3.0, 3.0)
    out = grad.logsumexp(Tensor(x), axis=axis, keepdims=keepdims)
    m = x.max(axis=axis, keepdims=True)
    expected = np.log(np.exp(x - m).sum(axis=axis, keepdims=True)) + m
    if not keepdims:
        expected = np.squeeze(expected, axis=axis)
    np.testing.assert_allclose(out.data, expected)
    assert check_grad(lambda t: grad.logsumexp(t, axis=axis, keepdims=keepdims), x) < TOL


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_stack(axis):
    xs = [rand((3, 4)) for _ in range(3)]
    out = grad.stack([Tensor(x) for x in xs], axis=axis)
    np.testing.assert_array_equal(out.data, np.stack(xs, axis=axis))
    assert check_grad(lambda *ts: grad.stack(ts, axis=axis), *xs) < TOL


@pytest.mark.parametrize(
    "key",
    [
        1,
        -1,
        slice(1, 3),
        (slice(None), 2),
        (Ellipsis, slice(None, None, -1)),
        (None, 0, slice(None, None, 2)),
        (slice(None), np.array([0, 3, 0, 1])),  # repeated index accumulates
        (np.array([2, 2, 0]), np.array([1, 1, 3])),
        (np.array([[0, 1], [1, 2]]), slice(None)),
    ],
)
def test_slice(key):
    x = rand((3, 4))
    out = Tensor(x)[key]
    np.testing.assert_array_equal(out.data, x[key])
    assert check_grad(lambda t: t[key], x) < TOL


def test_slice_repeated_index_accumulates():
    x = Tensor(np.arange(4.0), requires_grad=True)
    x[np.array([1, 1, 1, 3])].sum().backward()
    np.testing.assert_array_equal(x.grad, [0.0, 3.0, 0.0, 1.0])


# ---------------------------------------------------------------- composite


def test_softmax_cross_entropy_end_to_end():
    """A small MLP-style loss built purely from the primitive ops."""
    x, w, b = rand((5, 4)), rand((4, 3)), rand((3,))
    targets = np.eye(3)[rng.integers(0, 3, size=5)]

    def loss(x, w, b):
        logits = x @ w + b
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - shifted.exp().sum(axis=1, keepdims=True).log()
        return -(log_probs * targets).sum(axis=1).mean()

    lx = loss(Tensor(x), Tensor(w), Tensor(b)).data
    z = x @ w + b
    ref = -np.mean(np.sum((z - np.log(np.exp(z).sum(1, keepdims=True))) * targets, axis=1))
    np.testing.assert_allclose(lx, ref)
    assert check_grad(loss, x, w, b) < TOL
