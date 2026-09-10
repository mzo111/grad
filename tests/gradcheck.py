"""Central finite-difference gradient checker shared by the test modules."""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np

from grad import Tensor

TOL = 1e-6

# Every call appends ``(test id, error)`` here, so a whole suite run can be summarized
# afterwards by experiments/gradcheck_report.py. Recording is unconditional and costs one
# tuple per call: making it opt-in would mean the reporter had to patch this module before
# the test modules imported ``check_grad`` from it, and would report nothing if that
# ordering ever changed. Nothing reads this during an ordinary test run.
OBSERVED: list[tuple[str, float]] = []


def _current_test() -> str:
    """The running test's id, which pytest publishes in the environment."""
    return os.environ.get("PYTEST_CURRENT_TEST", "<not under pytest>").split(" (")[0]


def check_grad(
    fn: Callable[..., Tensor],
    *inputs: np.ndarray,
    eps: float = 1e-6,
    seed: int = 0,
) -> float:
    """Return the max absolute error between analytic and numeric gradients.

    ``fn`` maps tensors to one tensor of any shape. Its output is contracted
    with a fixed random cotangent ``v`` so the check covers the whole Jacobian,
    not only its row sums. Numeric gradients are central differences applied to
    the numpy-level function, element by element, for every input.
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(x, dtype=np.float64) for x in inputs]

    def scalar_fn(*arrs: np.ndarray) -> float:
        return float((fn(*(Tensor(a) for a in arrs)).data * v).sum())

    v = rng.standard_normal(fn(*(Tensor(a) for a in arrays)).shape)

    tensors = [Tensor(a, requires_grad=True) for a in arrays]
    out = fn(*tensors)
    (out * v).sum().backward()

    worst = 0.0
    for i, (t, a) in enumerate(zip(tensors, arrays, strict=True)):
        assert t.grad is not None, f"input {i} received no gradient"
        assert t.grad.shape == a.shape, f"input {i}: grad shape {t.grad.shape} != {a.shape}"
        numeric = np.zeros_like(a)
        for idx in np.ndindex(a.shape):
            plus = [x.copy() for x in arrays]
            minus = [x.copy() for x in arrays]
            plus[i][idx] += eps
            minus[i][idx] -= eps
            numeric[idx] = (scalar_fn(*plus) - scalar_fn(*minus)) / (2 * eps)
        worst = max(worst, float(np.abs(t.grad - numeric).max()))
    OBSERVED.append((_current_test(), worst))
    return worst
