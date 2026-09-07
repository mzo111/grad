"""Brute-force partition function by enumerating every parse tree explicitly.

Only feasible for tiny inputs: catalan(n - 1) bracketings times N^(2n - 2)
labelings of the non-root nodes. It shares no code with the inside algorithm.
"""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np

Tree = int | tuple["Tree", "Tree"]


@cache
def bracketings(i: int, j: int) -> tuple[Tree, ...]:
    """All binary bracketings of positions i..j-1; a leaf is its position."""
    if j - i == 1:
        return (i,)
    out: list[Tree] = []
    for k in range(i + 1, j):
        for left in bracketings(i, k):
            for right in bracketings(k, j):
                out.append((left, right))
    return tuple(out)


def _internal_nodes(tree: Tree) -> int:
    return 0 if isinstance(tree, int) else 1 + _internal_nodes(tree[0]) + _internal_nodes(tree[1])


def _tree_prob(tree: Tree, label: int, labels, sentence, p_binary, p_terminal) -> float:
    """Probability of one fully labeled tree; ``labels`` is an iterator of child labels."""
    if isinstance(tree, int):
        return p_terminal[label, sentence[tree]]
    left_label = next(labels)
    right_label = next(labels)
    p = p_binary[label, left_label, right_label]
    p *= _tree_prob(tree[0], left_label, labels, sentence, p_binary, p_terminal)
    p *= _tree_prob(tree[1], right_label, labels, sentence, p_binary, p_terminal)
    return p


def brute_force_log_z(
    log_binary: np.ndarray, log_terminal: np.ndarray, sentence: np.ndarray, start: int = 0
) -> tuple[float, int]:
    """Return ``(log Z, number of labeled trees enumerated)``."""
    sentence = np.asarray(sentence)
    n = len(sentence)
    n_nt = log_binary.shape[0]
    p_binary, p_terminal = np.exp(log_binary), np.exp(log_terminal)
    total = 0.0
    count = 0
    for tree in bracketings(0, n):
        n_labels = 2 * _internal_nodes(tree)  # every internal node labels two children
        for labels in itertools.product(range(n_nt), repeat=n_labels):
            total += _tree_prob(tree, start, iter(labels), sentence, p_binary, p_terminal)
            count += 1
    with np.errstate(divide="ignore"):
        return float(np.log(total)), count
