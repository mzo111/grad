"""Layers: a minimal Module system plus Linear, ReLU, and Sequential."""

from __future__ import annotations

from typing import Any

import numpy as np

from grad import ops
from grad.tensor import Tensor


class Module:
    """Base class. Subclasses implement ``forward`` and hold parameters as attributes.

    ``parameters()`` discovers them by walking the instance's attributes: any
    Tensor with ``requires_grad`` is a parameter, and Modules (or lists/tuples
    of Modules) are searched recursively.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Tensor]:
        found: list[Tensor] = []
        seen: set[int] = set()

        def visit(obj: Any) -> None:
            if isinstance(obj, Tensor):
                if obj.requires_grad and id(obj) not in seen:
                    seen.add(id(obj))
                    found.append(obj)
            elif isinstance(obj, Module):
                for value in vars(obj).values():
                    visit(value)
            elif isinstance(obj, list | tuple):
                for value in obj:
                    visit(value)

        visit(self)
        return found

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None


class Linear(Module):
    """``y = x @ W + b`` with He (Kaiming) normal initialization.

    ``W`` has shape ``(in_features, out_features)`` and entries drawn from
    ``N(0, sqrt(2 / in_features))``, the variance that keeps activations from
    shrinking or blowing up through ReLU layers. ``b`` starts at zero.
    """

    def __init__(
        self, in_features: int, out_features: int, rng: np.random.Generator | None = None
    ) -> None:
        rng = np.random.default_rng() if rng is None else rng
        std = np.sqrt(2.0 / in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(rng.normal(0.0, std, size=(in_features, out_features)), True)
        self.bias = Tensor(np.zeros(out_features), True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
