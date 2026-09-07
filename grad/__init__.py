"""grad: a small reverse-mode automatic differentiation engine on numpy."""

# The submodules only import grad.tensor / grad.ops, so any import order works.
from grad import losses, nn, optim
from grad.ops import (
    add,
    div,
    exp,
    index,
    log,
    logsumexp,
    matmul,
    max,
    mean,
    mul,
    neg,
    pow,
    relu,
    reshape,
    stack,
    sub,
    sum,
    transpose,
    unbroadcast,
)
from grad.tensor import Tensor

__all__ = [
    "Tensor",
    "losses",
    "nn",
    "optim",
    "add",
    "div",
    "exp",
    "index",
    "log",
    "logsumexp",
    "matmul",
    "max",
    "mean",
    "mul",
    "neg",
    "pow",
    "relu",
    "reshape",
    "stack",
    "sub",
    "sum",
    "transpose",
    "unbroadcast",
]
