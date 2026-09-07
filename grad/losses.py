"""Loss functions built from the engine's ops."""

from __future__ import annotations

from typing import Any

import numpy as np

from grad.tensor import Tensor


def softmax_cross_entropy(logits: Tensor, targets: Any) -> Tensor:
    """Mean cross-entropy between ``softmax(logits)`` and ``targets``.

    ``logits`` has shape ``(B, C)``. ``targets`` is either an integer array of
    class indices with shape ``(B,)`` or a one-hot ``(B, C)`` array/Tensor.

    Written as ``logsumexp(logits) - logits[target]`` rather than
    ``-log(softmax)``, and the logsumexp subtracts each row's max first:

        Without the shift, exp(z) overflows float64 to inf for any z > ~709
        (float32: > ~88). The row sum is then inf, log(inf) = inf, so the loss
        is inf; and the softmax that appears in the gradient, exp(z) / sum,
        evaluates to inf / inf = NaN, which then poisons every parameter on the
        next update. With the shift the largest term is exactly exp(0) = 1, the
        sum lies in [1, C], and the only rounding effect is that terms far below
        the max underflow to 0, which is harmless because they contribute
        nothing to the log anyway.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (batch, classes), got shape {logits.shape}")
    batch, n_classes = logits.shape

    t = targets.data if isinstance(targets, Tensor) else np.asarray(targets)
    if t.ndim == 1:
        if t.shape[0] != batch:
            raise ValueError(f"targets has {t.shape[0]} entries for a batch of {batch}")
        onehot = np.zeros((batch, n_classes), dtype=logits.dtype)
        onehot[np.arange(batch), t.astype(np.int64)] = 1.0
    elif t.shape == (batch, n_classes):
        onehot = np.asarray(t, dtype=logits.dtype)
    else:
        raise ValueError(f"targets shape {t.shape} does not match logits shape {logits.shape}")

    # The shift is a per-row constant: logsumexp(z) == logsumexp(z - m) + m for
    # any m, so its value has no effect on the result and d(loss)/dm is exactly
    # zero. Detaching it keeps the gradient path off the Max op's tie-handling.
    m = logits.max(axis=1, keepdims=True).detach()
    logsumexp = (logits - m).exp().sum(axis=1, keepdims=True).log() + m
    picked = (logits * onehot).sum(axis=1, keepdims=True)
    return (logsumexp - picked).mean()
