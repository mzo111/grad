"""Tensor: a numpy array plus the bookkeeping needed for reverse-mode autodiff.

The forward pass builds a DAG: every op that runs on a tensor requiring a
gradient produces a new tensor that records the op instance (which holds the
saved context for its backward rule) and the tensors that fed it. ``backward``
walks that DAG in reverse topological order and accumulates gradients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from grad.ops import Op

_AxisT = int | tuple[int, ...] | None


class Tensor:
    """A numpy-backed array node in a reverse-mode autodiff graph.

    Attributes:
        data: The forward value. Always a floating-point ndarray.
        grad: Accumulated gradient of the same shape as ``data``, or ``None``
            until a backward pass writes to it.
        requires_grad: Whether gradients should flow into this tensor.
    """

    __slots__ = ("data", "grad", "requires_grad", "_op", "_parents")

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        dtype: np.dtype | type | None = None,
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data
        arr = np.asarray(data, dtype=dtype)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64)
        self.data: np.ndarray = arr
        self.grad: np.ndarray | None = None
        self.requires_grad: bool = bool(requires_grad)
        # The op that produced this tensor and the tensors it consumed. Both are
        # empty for leaves and for any tensor computed without gradient tracking.
        self._op: Op | None = None
        self._parents: tuple[Tensor, ...] = ()

    # ------------------------------------------------------------------ basics

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self) -> Tensor:  # noqa: N802 - mirrors numpy's ``.T``
        return self.transpose()

    def item(self) -> float:
        return self.data.item()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        body = np.array2string(self.data, separator=", ", prefix="Tensor(")
        extra = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({body}{extra})"

    # ------------------------------------------------------------ graph control

    def detach(self) -> Tensor:
        """A new tensor sharing this data buffer but cut off from the graph."""
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self) -> None:
        self.grad = None

    def _topological_order(self) -> list[Tensor]:
        """Post-order DFS over parents, iterative so deep graphs do not recurse."""
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if id(parent) not in visited:
                    stack.append((parent, False))
        return order

    def backward(self, grad: Any = None) -> None:
        """Accumulate d(self)/d(node) into ``node.grad`` for every node in the graph.

        ``grad`` seeds the pass and must match this tensor's shape. It may be
        omitted only for single-element tensors, where it defaults to one.
        Gradients are added to whatever is already in ``grad`` buffers, so
        repeated calls accumulate; use ``zero_grad`` between steps.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "grad must be specified for a non-scalar tensor "
                    f"(shape {self.shape}); backward() only defaults it for scalars"
                )
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(grad, dtype=self.data.dtype)
            if seed.shape != self.shape:
                raise ValueError(
                    f"grad shape {seed.shape} does not match tensor shape {self.shape}"
                )

        # Gradients are propagated through a per-pass table so that a second
        # backward() call does not re-feed the grads left over from the first.
        # ``node.grad`` is only the accumulating, user-visible record.
        table: dict[int, np.ndarray] = {id(self): seed}
        self._accumulate(seed)
        for node in reversed(self._topological_order()):
            if node._op is None:
                continue
            grads = node._op.backward(table[id(node)])
            for parent, g in zip(node._parents, grads, strict=True):
                if g is None or not parent.requires_grad:
                    continue
                g = np.asarray(g)
                if g.shape != parent.shape:
                    raise RuntimeError(
                        f"{type(node._op).__name__}.backward produced gradient of shape "
                        f"{g.shape} for a parent of shape {parent.shape}"
                    )
                key = id(parent)
                table[key] = g if key not in table else table[key] + g
                parent._accumulate(g)

    def _accumulate(self, g: np.ndarray) -> None:
        # Out-of-place on first write so a gradient array returned by an op is never
        # aliased by a later in-place update.
        self.grad = g.copy() if self.grad is None else self.grad + g

    # ------------------------------------------------------------- operators
    # Imported lazily: ``grad.ops`` imports ``Tensor`` from this module.

    def __add__(self, other: Any) -> Tensor:
        from grad import ops

        return ops.add(self, other)

    def __radd__(self, other: Any) -> Tensor:
        from grad import ops

        return ops.add(other, self)

    def __sub__(self, other: Any) -> Tensor:
        from grad import ops

        return ops.sub(self, other)

    def __rsub__(self, other: Any) -> Tensor:
        from grad import ops

        return ops.sub(other, self)

    def __mul__(self, other: Any) -> Tensor:
        from grad import ops

        return ops.mul(self, other)

    def __rmul__(self, other: Any) -> Tensor:
        from grad import ops

        return ops.mul(other, self)

    def __truediv__(self, other: Any) -> Tensor:
        from grad import ops

        return ops.div(self, other)

    def __rtruediv__(self, other: Any) -> Tensor:
        from grad import ops

        return ops.div(other, self)

    def __matmul__(self, other: Any) -> Tensor:
        from grad import ops

        return ops.matmul(self, other)

    def __rmatmul__(self, other: Any) -> Tensor:
        from grad import ops

        return ops.matmul(other, self)

    def __pow__(self, other: Any) -> Tensor:
        from grad import ops

        return ops.pow(self, other)

    def __rpow__(self, other: Any) -> Tensor:
        from grad import ops

        return ops.pow(other, self)

    def __neg__(self) -> Tensor:
        from grad import ops

        return ops.neg(self)

    def exp(self) -> Tensor:
        from grad import ops

        return ops.exp(self)

    def log(self) -> Tensor:
        from grad import ops

        return ops.log(self)

    def __getitem__(self, key: Any) -> Tensor:
        from grad import ops

        return ops.index(self, key)

    def logsumexp(self, axis: _AxisT = None, keepdims: bool = False) -> Tensor:
        from grad import ops

        return ops.logsumexp(self, axis=axis, keepdims=keepdims)

    def sum(self, axis: _AxisT = None, keepdims: bool = False) -> Tensor:
        from grad import ops

        return ops.sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: _AxisT = None, keepdims: bool = False) -> Tensor:
        from grad import ops

        return ops.mean(self, axis=axis, keepdims=keepdims)

    def max(self, axis: _AxisT = None, keepdims: bool = False) -> Tensor:
        from grad import ops

        return ops.max(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape: int | tuple[int, ...]) -> Tensor:
        from grad import ops

        if len(shape) == 1 and isinstance(shape[0], tuple | list):
            shape = tuple(shape[0])
        return ops.reshape(self, shape)

    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        from grad import ops

        return ops.transpose(self, axes)
