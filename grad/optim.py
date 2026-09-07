"""Optimizers. Updates are plain numpy arithmetic written in place into ``p.data``."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from grad.tensor import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor], lr: float) -> None:
        self.params = list(params)
        self.lr = float(lr)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with classical momentum.

    v <- momentum * v + g
    p <- p - lr * v
    """

    def __init__(self, params: Iterable[Tensor], lr: float, momentum: float = 0.0) -> None:
        super().__init__(params, lr)
        self.momentum = float(momentum)
        self._velocity: dict[int, np.ndarray] = {}

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.momentum == 0.0:
                p.data -= self.lr * g
                continue
            v = self._velocity.get(id(p))
            v = g.copy() if v is None else self.momentum * v + g
            self._velocity[id(p)] = v
            p.data -= self.lr * v


class Adam(Optimizer):
    """Adam, Algorithm 1 of Kingma & Ba (2015), with bias correction.

    t <- t + 1
    m <- b1 * m + (1 - b1) * g
    v <- b2 * v + (1 - b2) * g^2
    m_hat <- m / (1 - b1^t)
    v_hat <- v / (1 - b2^t)
    p <- p - lr * m_hat / (sqrt(v_hat) + eps)

    ``eps`` is added outside the square root, as in the paper.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.t = 0
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bias1 = 1.0 - b1**self.t
        bias2 = 1.0 - b2**self.t
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            key = id(p)
            m = self._m.get(key)
            v = self._v.get(key)
            if m is None:
                m = np.zeros_like(p.data)
                v = np.zeros_like(p.data)
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * (g * g)
            self._m[key], self._v[key] = m, v
            m_hat = m / bias1
            v_hat = v / bias2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
