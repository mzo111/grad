"""Reverse-mode autodiff through the inside algorithm recovers the outside quantities.

Why the gradients are the outside quantities
--------------------------------------------
Z = sum over parse trees of the product of their rule probabilities, and the
inside chart computes it by the recursion

    beta(A, i, j) = sum_{k, B, C} p(A -> B C) beta(B, i, k) beta(C, k, j).

Fix one cell (A, i, j) and regard its inside value beta(A, i, j) as a free
variable that feeds the cells above it. Every tree that contains A over span
(i, j) factorizes as (a subtree below the cell) x (the rest of the tree
outside the cell), and no tree uses the cell twice, so Z is *linear* in
beta(A, i, j):

    Z = beta(A, i, j) * [total weight of everything outside span (i, j) rooted
                          at A] + [weight of trees that avoid the cell].

The bracketed coefficient is exactly the outside probability alpha(A, i, j),
so dZ / d beta(A, i, j) = alpha(A, i, j).

The chart stores log beta and the loss is log Z, so by the chain rule

    d log Z / d log beta(A, i, j) = (1/Z) * dZ/dbeta * beta
                                  = alpha(A, i, j) beta(A, i, j) / Z
                                  = mu(A, i, j),

the posterior probability that a parse contains A over (i, j). That is what
``chart[L].grad`` holds after ``backward``. alpha itself is then recovered on
every reachable cell as grad * Z / beta.

The same argument applied to a rule probability p(r): Z is a polynomial in
p(r) whose degree-m terms are the trees using r exactly m times, so
p(r) dZ/dp(r) / Z = sum_m m * P(trees using r m times | sentence) = E[count of r],
i.e. d log Z / d log p(r) is the expected rule count of the E-step of
inside-outside. ``log_binary.grad`` and ``log_terminal.grad`` hold these.

The script computes every one of these by hand in numpy (grammar/outside.py)
and reports the maximum absolute differences.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammar import PCFG, inside  # noqa: E402
from grammar.outside import expected_counts, outside, span_posteriors  # noqa: E402


def sparse_grammar() -> PCFG:
    return PCFG.from_rules(
        ["S", "NP", "VP", "V", "D", "N"],
        ["the", "dog", "cat", "chases", "sees", "sleeps"],
        binary={("S", "NP", "VP"): 1.0, ("NP", "D", "N"): 0.7, ("VP", "V", "NP"): 0.6},
        terminal={
            ("NP", "dog"): 0.2,
            ("NP", "cat"): 0.1,
            ("VP", "sleeps"): 0.4,
            ("V", "chases"): 0.5,
            ("V", "sees"): 0.5,
            ("D", "the"): 1.0,
            ("N", "dog"): 0.5,
            ("N", "cat"): 0.5,
        },
    )


def compare(name: str, g: PCFG, sentences: np.ndarray) -> None:
    lb, lt = g.log_probs_numpy()
    result = inside(g, sentences)
    result.log_z.sum().backward()

    worst_mu = worst_log_alpha = 0.0
    counts_b, counts_t = np.zeros_like(lb), np.zeros_like(lt)
    n_cells = n_unreachable = 0
    for b, sentence in enumerate(sentences):
        beta = result.chart_array(b)
        alpha = outside(lb, lt, sentence, beta)
        log_z = beta[0, len(sentence), 0]
        mu = span_posteriors(alpha, beta, log_z)
        grad = result.chart_grad_array(b)

        # (1) chart gradient == span posterior mu = alpha * beta / Z, all cells
        worst_mu = max(worst_mu, np.abs(grad - mu).max())

        # (2) alpha recovered from the gradient on reachable cells: grad * Z / beta
        reachable = np.isfinite(beta) & (grad > 0)
        with np.errstate(divide="ignore"):
            log_alpha_rec = np.log(grad[reachable]) + log_z - beta[reachable]
        worst_log_alpha = max(worst_log_alpha, np.abs(log_alpha_rec - alpha[reachable]).max())

        valid = np.triu(np.ones((len(sentence) + 1,) * 2, dtype=bool), k=1)[:, :, None]
        valid = np.broadcast_to(valid, beta.shape)
        n_cells += int(valid.sum())
        n_unreachable += int((valid & np.isneginf(beta)).sum())

        cb, ct = expected_counts(lb, lt, sentence, beta, alpha, log_z)
        counts_b += cb
        counts_t += ct

    # (3) rule-table gradients == expected rule counts, summed over the batch
    worst_binary = np.abs(result.log_binary.grad - counts_b).max()
    worst_terminal = np.abs(result.log_terminal.grad - counts_t).max()

    print(
        f"{name}: batch {sentences.shape[0]}, length {sentences.shape[1]}, "
        f"N={g.n_nonterminals}, V={g.n_terminals}; "
        f"{n_cells} chart cells, {n_unreachable} unreachable"
    )
    print(f"  log Z                                   {result.log_z.data}")
    print(f"  max |d logZ/d log beta  -  alpha*beta/Z|  {worst_mu:.3e}")
    print(f"  max |log(grad*Z/beta)   -  log alpha|     {worst_log_alpha:.3e}  (reachable cells)")
    print(f"  max |d logZ/d log p(A->BC) - E[count]|    {worst_binary:.3e}")
    print(f"  max |d logZ/d log p(A->w)  - E[count]|    {worst_terminal:.3e}")


def main() -> None:
    g = sparse_grammar()
    w = g.terminals.index
    sentences = np.array(
        [
            [w(t) for t in "the dog chases the cat".split()],
            [w(t) for t in "the cat sees the dog".split()],
        ]
    )
    compare("sparse toy grammar", g, sentences)

    rng = np.random.default_rng(0)
    g = PCFG.random(4, 6, rng)
    compare("dense random grammar", g, rng.integers(0, g.n_terminals, size=(3, 8)))


if __name__ == "__main__":
    main()
