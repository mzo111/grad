"""The inside algorithm on the grad engine, vectorized over a batch and over spans.

Chart layout: ``chart[L]`` is a Tensor of shape ``(batch, n - L + 1, N)`` whose
entry ``[b, i, A]`` is log beta(A, i, i + L) for sentence ``b``: the log total
probability that A derives words ``i .. i+L-1``.

Loops. Two Python loops remain and both are stated here:

* Over span length L (2 .. n): unavoidable, every length depends on all shorter
  lengths, so the levels are sequentially dependent.
* Over split points l (1 .. L-1) within a length: each iteration is only a pair
  of view slices (left children of length l, right children of length L - l).
  The arithmetic for the whole level is one broadcast add and one logsumexp over
  ``batch x spans x splits x N^3``. Folding the split loop away would need a
  strided/diagonal gather (``as_strided`` over a dense (i, j) chart), which the
  engine does not have. It costs O(n^2) slice ops per sentence batch in total.

There is no loop over the batch or over individual cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grad import Tensor, stack
from grammar.pcfg import PCFG


@dataclass
class InsideResult:
    log_z: Tensor  # (batch,)
    chart: list[Tensor | None]  # chart[L] has shape (batch, n - L + 1, N); chart[0] is None
    log_binary: Tensor  # normalized (N, N, N), a graph tensor
    log_terminal: Tensor  # normalized (N, V), a graph tensor
    sentences: np.ndarray  # (batch, n)

    @property
    def n(self) -> int:
        return len(self.chart) - 1

    def _dense(self, b: int, take) -> np.ndarray:
        """Assemble a ``(n+1, n+1, N)`` array indexed ``[i, j, A]`` for sentence ``b``."""
        n = self.n
        n_nt = self.log_binary.shape[0]
        out = np.full((n + 1, n + 1, n_nt), -np.inf)
        for length in range(1, n + 1):
            values = take(self.chart[length])
            for i in range(n - length + 1):
                out[i, i + length] = values[b, i]
        return out

    def chart_array(self, b: int = 0) -> np.ndarray:
        """Dense log beta for sentence ``b``, ``-inf`` outside valid spans."""
        return self._dense(b, lambda t: t.data)

    def chart_grad_array(self, b: int = 0) -> np.ndarray:
        """Dense d(loss)/d(log beta) for sentence ``b`` after ``backward``; 0 outside spans."""
        out = self._dense(b, lambda t: t.grad)
        out[np.isneginf(out)] = 0.0
        return out


def inside(
    grammar: PCFG | tuple[Tensor, Tensor],
    sentences: np.ndarray,
    start: int = 0,
) -> InsideResult:
    """Log partition function of each sentence in a batch of equal-length sentences."""
    if isinstance(grammar, PCFG):
        log_binary, log_terminal = grammar.log_probs()
    else:
        log_binary, log_terminal = grammar
        if not isinstance(log_binary, Tensor):
            log_binary = Tensor(log_binary)
        if not isinstance(log_terminal, Tensor):
            log_terminal = Tensor(log_terminal)

    sentences = np.asarray(sentences, dtype=np.int64)
    if sentences.ndim != 2 or sentences.shape[1] == 0:
        raise ValueError(f"sentences must be (batch, n) with n >= 1, got {sentences.shape}")
    batch, n = sentences.shape
    n_nt = log_binary.shape[0]

    chart: list[Tensor | None] = [None] * (n + 1)
    # Length 1: gather A -> w_i for every position. Integer-array indexing gives
    # (N, batch, n); transpose to (batch, n, N).
    chart[1] = log_terminal[:, sentences].transpose((1, 2, 0))

    rules = log_binary.reshape(1, 1, 1, n_nt, n_nt, n_nt)
    for length in range(2, n + 1):
        spans = n - length + 1
        pairs = []
        for left_len in range(1, length):
            left = chart[left_len][:, :spans].reshape(batch, spans, n_nt, 1)
            right = chart[length - left_len][:, left_len : left_len + spans]
            right = right.reshape(batch, spans, 1, n_nt)
            pairs.append(left + right)  # (batch, spans, B, C): log beta_B + log beta_C
        children = stack(pairs, axis=2).reshape(batch, spans, length - 1, 1, n_nt, n_nt)
        # (batch, spans, splits, A, B, C) -> logsumexp over splits, B, C.
        chart[length] = (children + rules).logsumexp(axis=(2, 4, 5))

    log_z = chart[n][:, 0, start]
    return InsideResult(log_z, chart, log_binary, log_terminal, sentences)
