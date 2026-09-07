"""Naive inside algorithm: explicit loops over chart cells, plain numpy, no engine.

Reference implementation for the tests and the baseline for ``experiments/bench.py``.
"""

from __future__ import annotations

import numpy as np


def inside_naive(log_binary: np.ndarray, log_terminal: np.ndarray, sentence: np.ndarray):
    """Return the dense log-inside chart ``beta[i, j, A]`` of shape ``(n+1, n+1, N)``."""
    sentence = np.asarray(sentence)
    n = len(sentence)
    n_nt = log_binary.shape[0]
    beta = np.full((n + 1, n + 1, n_nt), -np.inf)
    for i in range(n):
        beta[i, i + 1] = log_terminal[:, sentence[i]]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length
            for k in range(i + 1, j):
                for a in range(n_nt):
                    # log sum over B, C of p(A -> B C) beta(B, i, k) beta(C, k, j)
                    terms = log_binary[a] + beta[i, k][:, None] + beta[k, j][None, :]
                    beta[i, j, a] = np.logaddexp(beta[i, j, a], np.logaddexp.reduce(terms.ravel()))
    return beta


def log_z_naive(
    log_binary: np.ndarray, log_terminal: np.ndarray, sentence: np.ndarray, start: int = 0
) -> float:
    return float(inside_naive(log_binary, log_terminal, sentence)[0, len(sentence), start])
