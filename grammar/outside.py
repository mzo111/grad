"""The outside algorithm and expected rule counts, written by hand in numpy.

This is the reference the autodiff result is checked against; it is not part
of any graph. All quantities are in log space.

Definitions (n words, A a nonterminal, 0 <= i < j <= n):
  beta(A, i, j)  = P(A =>* w_i .. w_{j-1})                       (inside)
  alpha(A, i, j) = P(S =>* w_0 .. w_{i-1}  A  w_j .. w_{n-1})     (outside)
  Z              = beta(S, 0, n)
  mu(A, i, j)    = alpha(A, i, j) beta(A, i, j) / Z             (span posterior)
"""

from __future__ import annotations

import numpy as np


def outside(
    log_binary: np.ndarray,
    log_terminal: np.ndarray,
    sentence: np.ndarray,
    beta: np.ndarray,
    start: int = 0,
) -> np.ndarray:
    """Dense log-outside chart ``alpha[i, j, A]`` given the inside chart ``beta``.

    Recurrence, spans processed from longest to shortest so every alpha(A, i, j)
    is complete before it is used:

      alpha(B, i, k) += sum_{A, C, j > k} alpha(A, i, j) p(A -> B C) beta(C, k, j)
      alpha(C, k, j) += sum_{A, B, i < k} alpha(A, i, j) p(A -> B C) beta(B, i, k)
    """
    n = len(sentence)
    n_nt = log_binary.shape[0]
    alpha = np.full((n + 1, n + 1, n_nt), -np.inf)
    alpha[0, n, start] = 0.0
    for length in range(n, 1, -1):
        for i in range(n - length + 1):
            j = i + length
            parent = alpha[i, j][:, None, None]  # (A, 1, 1)
            for k in range(i + 1, j):
                # (A, B, C) tables of alpha(A,i,j) + log p(A->BC) + log beta(sibling)
                to_left = parent + log_binary + beta[k, j][None, None, :]
                to_right = parent + log_binary + beta[i, k][None, :, None]
                alpha[i, k] = np.logaddexp(alpha[i, k], np.logaddexp.reduce(to_left, axis=(0, 2)))
                alpha[k, j] = np.logaddexp(alpha[k, j], np.logaddexp.reduce(to_right, axis=(0, 1)))
    return alpha


def span_posteriors(alpha: np.ndarray, beta: np.ndarray, log_z: float) -> np.ndarray:
    """``mu[i, j, A] = alpha beta / Z``; exactly 0 for unreachable cells."""
    with np.errstate(invalid="ignore"):  # -inf + -inf is fine, exp gives 0
        return np.exp(alpha + beta - log_z)


def expected_counts(
    log_binary: np.ndarray,
    log_terminal: np.ndarray,
    sentence: np.ndarray,
    beta: np.ndarray,
    alpha: np.ndarray,
    log_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected number of uses of each rule in a parse drawn from the posterior.

    c(A -> B C) = (1/Z) sum_{i<k<j} alpha(A,i,j) p(A->BC) beta(B,i,k) beta(C,k,j)
    c(A -> w)   = (1/Z) sum_{i : w_i = w} alpha(A,i,i+1) p(A->w)
    """
    n = len(sentence)
    n_nt, n_t = log_terminal.shape
    count_binary = np.zeros((n_nt, n_nt, n_nt))
    count_terminal = np.zeros((n_nt, n_t))
    for i in range(n):
        for j in range(i + 2, n + 1):
            parent = alpha[i, j][:, None, None]
            for k in range(i + 1, j):
                term = parent + log_binary + beta[i, k][None, :, None] + beta[k, j][None, None, :]
                count_binary += np.exp(term - log_z)
    for i in range(n):
        w = sentence[i]
        count_terminal[:, w] += np.exp(alpha[i, i + 1] + log_terminal[:, w] - log_z)
    return count_binary, count_terminal
