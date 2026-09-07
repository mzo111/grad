"""Differentiable operations.

Every op is a class with a ``forward`` rule (pure numpy) and a ``backward`` rule
that maps the gradient of the output to gradients for each input, already
shaped like that input. ``Op.apply`` wires the op into the graph.

Broadcasting is undone in exactly one place, :func:`unbroadcast`, which every
op that relies on numpy broadcasting calls on its input gradients.
"""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from typing import Any

import numpy as np

from grad.tensor import Tensor

_AxisT = int | tuple[int, ...] | None

# --------------------------------------------------------------------- helpers


def _as_tensor(x: Any) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape``, undoing numpy broadcasting.

    Numpy broadcasts by (1) prepending length-1 axes on the left until ranks
    match and (2) stretching any length-1 axis. The gradient of a broadcast
    input is therefore the output gradient summed over every prepended axis and
    every stretched axis, then reshaped to the original shape.
    """
    grad = np.asarray(grad)
    if grad.shape == shape:
        return grad
    n_extra = grad.ndim - len(shape)
    if n_extra > 0:
        grad = grad.sum(axis=tuple(range(n_extra)))
    stretched = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad.reshape(shape)


def _normalize_axis(axis: _AxisT, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    out = []
    for a in axis:
        if not -ndim <= a < ndim:
            raise np.exceptions.AxisError(a, ndim)
        out.append(a % ndim)
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate axes in {axis}")
    return tuple(out)


def _expand_reduced(
    grad: np.ndarray, in_shape: tuple[int, ...], axes: tuple[int, ...], keepdims: bool
) -> np.ndarray:
    """Broadcast a reduction's output gradient back over the reduced input shape."""
    if not keepdims:
        grad = np.expand_dims(grad, axes)
    return np.broadcast_to(grad, in_shape)


# ------------------------------------------------------------------- base class


class Op:
    """One application of a differentiable function.

    Subclasses implement ``forward`` and ``backward``. The instance is the
    context: anything ``backward`` needs is stashed on ``self`` by ``forward``.
    """

    def forward(self, *arrays: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        raise NotImplementedError

    def apply(self, *inputs: Any) -> Tensor:
        parents = tuple(_as_tensor(x) for x in inputs)
        out = Tensor(self.forward(*(p.data for p in parents)))
        if builtins.any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._op = self
            out._parents = parents
        return out


# ------------------------------------------------------------ elementwise binary


class Add(Op):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.a_shape, self.b_shape = a.shape, b.shape
        return a + b

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return unbroadcast(grad, self.a_shape), unbroadcast(grad, self.b_shape)


class Sub(Op):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.a_shape, self.b_shape = a.shape, b.shape
        return a - b

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return unbroadcast(grad, self.a_shape), unbroadcast(-grad, self.b_shape)


class Mul(Op):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.a, self.b = a, b
        return a * b

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return unbroadcast(grad * self.b, self.a.shape), unbroadcast(grad * self.a, self.b.shape)


class Div(Op):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.a, self.b = a, b
        return a / b

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ga = unbroadcast(grad / self.b, self.a.shape)
        gb = unbroadcast(-grad * self.a / (self.b * self.b), self.b.shape)
        return ga, gb


class Pow(Op):
    """``a ** b`` for a tensor base and a tensor (or constant) exponent."""

    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.a, self.b = a, b
        self.out = a**b
        return self.out

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.a, self.b
        ga = unbroadcast(grad * b * a ** (b - 1), a.shape)
        # d/db a**b = a**b * ln(a); only meaningful for a > 0, like the forward pass.
        gb = unbroadcast(grad * self.out * np.log(a), b.shape)
        return ga, gb


# ------------------------------------------------------------- elementwise unary


class Neg(Op):
    def forward(self, a: np.ndarray) -> np.ndarray:
        return -a

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray]:
        return (-grad,)


class Exp(Op):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.out = np.exp(a)
        return self.out

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray]:
        return (grad * self.out,)


class Log(Op):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.a = a
        return np.log(a)

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray]:
        return (grad / self.a,)


class Relu(Op):
    """Elementwise ``max(a, 0)``.

    Convention at exactly zero: the gradient is 0. ReLU is not differentiable
    there and any value in [0, 1] is a valid subgradient; using the strict mask
    ``a > 0`` means a unit sitting exactly on the threshold is treated as
    inactive, which is also what PyTorch does.
    """

    def forward(self, a: np.ndarray) -> np.ndarray:
        self.a = a
        return np.maximum(a, 0)

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray]:
        return (grad * (self.a > 0),)


# ------------------------------------------------------------------- matmul


class MatMul(Op):
    """``a @ b`` with numpy semantics: 1-D vectors, matrices, and broadcast batches.

    Backward works in the 2-D-or-batched regime, so 1-D operands are promoted
    (``(n,) -> (1, n)`` on the left, ``(n,) -> (n, 1)`` on the right) and the
    resulting gradients are reshaped back to 1-D at the end.
    """

    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.ndim == 0 or b.ndim == 0:
            raise ValueError("matmul: inputs must be at least 1-D")
        self.a_shape, self.b_shape = a.shape, b.shape
        self.a2 = a[None, :] if a.ndim == 1 else a
        self.b2 = b[:, None] if b.ndim == 1 else b
        return a @ b

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a2, b2 = self.a2, self.b2
        # Re-insert the axes that numpy squeezed out of the output for 1-D operands
        # so that ``grad`` matches ``a2 @ b2``.
        # (Right side first: for two 1-D operands the output is 0-d and must
        # grow to (1, 1).)
        g = grad
        if len(self.b_shape) == 1:
            g = np.expand_dims(g, -1)
        if len(self.a_shape) == 1:
            g = np.expand_dims(g, -2)
        ga = g @ np.swapaxes(b2, -1, -2)
        gb = np.swapaxes(a2, -1, -2) @ g
        ga = unbroadcast(ga, a2.shape).reshape(self.a_shape)
        gb = unbroadcast(gb, b2.shape).reshape(self.b_shape)
        return ga, gb


# ---------------------------------------------------------------- reductions


class _Reduction(Op):
    def __init__(self, axis: _AxisT = None, keepdims: bool = False) -> None:
        self.axis = axis
        self.keepdims = bool(keepdims)

    def _setup(self, a: np.ndarray) -> None:
        self.in_shape = a.shape
        self.axes = _normalize_axis(self.axis, a.ndim)
        self.n_reduced = int(np.prod([a.shape[i] for i in self.axes], dtype=np.int64))

    def _expand(self, grad: np.ndarray) -> np.ndarray:
        return _expand_reduced(grad, self.in_shape, self.axes, self.keepdims)


class Sum(_Reduction):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self._setup(a)
        return a.sum(axis=self.axes, keepdims=self.keepdims)

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray]:
        return (self._expand(grad),)


class Mean(_Reduction):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self._setup(a)
        return a.mean(axis=self.axes, keepdims=self.keepdims)

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray]:
        return (self._expand(grad) / self.n_reduced,)


class Max(_Reduction):
    """Gradient flows to the arg-max; exact ties share it equally."""

    def forward(self, a: np.ndarray) -> np.ndarray:
        self._setup(a)
        self.a = a
        self.out = a.max(axis=self.axes, keepdims=self.keepdims)
        return self.out

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray]:
        mask = (self.a == self._expand(self.out)).astype(self.a.dtype)
        count = mask.sum(axis=self.axes, keepdims=True)
        return (self._expand(grad) * mask / count,)


class LogSumExp(_Reduction):
    """``log(sum(exp(a)))`` over ``axis``, stable and safe at -inf.

    Forward shifts by the max of each reduced group (treated as 0 when the max
    is not finite), so an all--inf group or a zero-length axis gives -inf
    without overflow or NaN. Backward is ``g * exp(a - out)``, i.e. ``g`` times
    the softmax of the group, masked to exactly 0 wherever ``out`` is -inf so
    that empty groups contribute no gradient instead of NaN.
    """

    def forward(self, a: np.ndarray) -> np.ndarray:
        self._setup(a)
        self.a = a
        if self.n_reduced == 0:
            kept = a.sum(axis=self.axes, keepdims=True).shape
            self.out_k = np.full(kept, -np.inf, dtype=a.dtype)
        else:
            m = a.max(axis=self.axes, keepdims=True)
            m = np.where(np.isfinite(m), m, 0.0)
            with np.errstate(divide="ignore"):
                self.out_k = np.log(np.exp(a - m).sum(axis=self.axes, keepdims=True)) + m
        return self.out_k if self.keepdims else np.squeeze(self.out_k, axis=self.axes)

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray]:
        with np.errstate(invalid="ignore"):  # -inf - (-inf) inside an empty group
            weights = np.exp(self.a - self.out_k)
        weights = np.where(np.isfinite(self.out_k), weights, 0.0)
        return (self._expand(grad) * weights,)


# -------------------------------------------------------------- shape ops


class Reshape(Op):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = tuple(shape)

    def forward(self, a: np.ndarray) -> np.ndarray:
        self.in_shape = a.shape
        return a.reshape(self.shape)

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray]:
        return (grad.reshape(self.in_shape),)


class Transpose(Op):
    def __init__(self, axes: tuple[int, ...] | None = None) -> None:
        self.axes = None if axes is None else tuple(axes)

    def forward(self, a: np.ndarray) -> np.ndarray:
        self.perm = tuple(range(a.ndim))[::-1] if self.axes is None else self.axes
        self.perm = _normalize_axis(self.perm, a.ndim)
        if len(self.perm) != a.ndim:
            raise ValueError(f"axes {self.axes} do not match tensor of rank {a.ndim}")
        return a.transpose(self.perm)

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray]:
        return (grad.transpose(np.argsort(self.perm)),)


class Stack(Op):
    """Join same-shaped inputs along a new axis; backward splits the gradient back."""

    def __init__(self, axis: int = 0) -> None:
        self.axis = axis

    def forward(self, *arrays: np.ndarray) -> np.ndarray:
        self.count = len(arrays)
        return np.stack(arrays, axis=self.axis)

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        parts = np.moveaxis(grad, self.axis, 0)
        return tuple(parts[i] for i in range(self.count))


class Slice(Op):
    """``a[key]`` for basic indexing (ints, slices, None, Ellipsis) and integer arrays.

    Backward scatters the gradient into zeros of the input shape. Integer-array
    keys use ``np.add.at`` so repeated indices accumulate instead of overwrite.
    """

    def __init__(self, key: Any) -> None:
        if not isinstance(key, tuple):
            key = (key,)
        key = tuple(np.asarray(k) if isinstance(k, list | np.ndarray) else k for k in key)
        self.key = key
        self.advanced = builtins.any(isinstance(k, np.ndarray) for k in key)
        for k in key:
            if isinstance(k, np.ndarray) and not np.issubdtype(k.dtype, np.integer):
                raise TypeError("only integer arrays are supported as index arrays")

    def forward(self, a: np.ndarray) -> np.ndarray:
        self.in_shape = a.shape
        return a[self.key]

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray]:
        out = np.zeros(self.in_shape, dtype=grad.dtype)
        if self.advanced:
            np.add.at(out, self.key, grad)
        else:
            out[self.key] = grad
        return (out,)


# ------------------------------------------------------ functional interface


def add(a: Any, b: Any) -> Tensor:
    return Add().apply(a, b)


def sub(a: Any, b: Any) -> Tensor:
    return Sub().apply(a, b)


def mul(a: Any, b: Any) -> Tensor:
    return Mul().apply(a, b)


def div(a: Any, b: Any) -> Tensor:
    return Div().apply(a, b)


def pow(a: Any, b: Any) -> Tensor:  # noqa: A001 - mirrors the operator it implements
    return Pow().apply(a, b)


def neg(a: Any) -> Tensor:
    return Neg().apply(a)


def exp(a: Any) -> Tensor:
    return Exp().apply(a)


def log(a: Any) -> Tensor:
    return Log().apply(a)


def relu(a: Any) -> Tensor:
    return Relu().apply(a)


def matmul(a: Any, b: Any) -> Tensor:
    return MatMul().apply(a, b)


def sum(a: Any, axis: _AxisT = None, keepdims: bool = False) -> Tensor:  # noqa: A001
    return Sum(axis, keepdims).apply(a)


def mean(a: Any, axis: _AxisT = None, keepdims: bool = False) -> Tensor:
    return Mean(axis, keepdims).apply(a)


def max(a: Any, axis: _AxisT = None, keepdims: bool = False) -> Tensor:  # noqa: A001
    return Max(axis, keepdims).apply(a)


def logsumexp(a: Any, axis: _AxisT = None, keepdims: bool = False) -> Tensor:
    return LogSumExp(axis, keepdims).apply(a)


def stack(tensors: Sequence[Any], axis: int = 0) -> Tensor:
    return Stack(axis).apply(*tensors)


def index(a: Any, key: Any) -> Tensor:
    return Slice(key).apply(a)


def reshape(a: Any, shape: tuple[int, ...]) -> Tensor:
    return Reshape(shape).apply(a)


def transpose(a: Any, axes: tuple[int, ...] | None = None) -> Tensor:
    return Transpose(axes).apply(a)
