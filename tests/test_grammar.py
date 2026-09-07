"""PCFG: normalization, inside vs brute force and vs naive loops, gradient vs outside."""

from __future__ import annotations

import numpy as np
import pytest

from grammar import PCFG, inside
from grammar.enumerate import brute_force_log_z
from grammar.naive import inside_naive
from grammar.outside import expected_counts, outside, span_posteriors

# Tolerance for every float64 comparison in this file. Chart values are sums of
# at most a few thousand terms, so accumulated rounding is ~1e-13; 1e-10 leaves
# a wide margin while still catching any genuine algorithmic discrepancy.
ATOL = 1e-10


def sparse_grammar() -> PCFG:
    """Toy English grammar with absent rules, so charts contain unreachable cells."""
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


def words(g: PCFG, text: str) -> np.ndarray:
    return np.array([g.terminals.index(w) for w in text.split()])


def dense_grammar(seed: int = 0, n_nt: int = 3, n_t: int = 4) -> PCFG:
    return PCFG.random(n_nt, n_t, np.random.default_rng(seed))


# ------------------------------------------------------------------ basics


@pytest.mark.parametrize("g", [sparse_grammar(), dense_grammar()])
def test_rules_normalize_per_nonterminal(g):
    np.testing.assert_allclose(g.rule_distributions().sum(axis=1), 1.0, atol=1e-12)


def test_from_rules_keeps_given_probabilities():
    g = sparse_grammar()
    lb, lt = g.log_probs_numpy()
    s, np_, vp = 0, 1, 2
    np.testing.assert_allclose(np.exp(lb[s, np_, vp]), 1.0)
    np.testing.assert_allclose(np.exp(lt[np_, g.terminals.index("dog")]), 0.2)
    assert np.isneginf(lb[np_, np_, np_])


def test_rejects_nonterminal_without_rules():
    b = np.full((2, 2, 2), -np.inf)
    t = np.array([[0.0, 0.0], [-np.inf, -np.inf]])
    with pytest.raises(ValueError):
        PCFG(["S", "X"], ["a", "b"], b, t)


# --------------------------------------------------------- inside correctness


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_inside_matches_brute_force_enumeration_dense(n):
    g = dense_grammar()
    lb, lt = g.log_probs_numpy()
    rng = np.random.default_rng(n)
    sentence = rng.integers(0, g.n_terminals, size=n)
    expected, n_trees = brute_force_log_z(lb, lt, sentence)
    assert n_trees == (5 if n == 4 else 2 if n == 3 else 1) * 3 ** (2 * n - 2)
    got = inside(g, sentence[None, :]).log_z.data[0]
    np.testing.assert_allclose(got, expected, atol=ATOL)


@pytest.mark.parametrize("text", ["dog sleeps", "the cat sleeps", "cat sees dog"])
def test_inside_matches_brute_force_enumeration_sparse(text):
    g = sparse_grammar()
    lb, lt = g.log_probs_numpy()
    sentence = words(g, text)
    expected, _ = brute_force_log_z(lb, lt, sentence)
    got = inside(g, sentence[None, :]).log_z.data[0]
    np.testing.assert_allclose(got, expected, atol=ATOL)


def test_inside_hand_computed_probability():
    # "the dog chases the cat": the only parse is
    # S -> NP VP, NP -> D N (the dog), VP -> V NP, V -> chases, NP -> D N (the cat)
    g = sparse_grammar()
    p = 1.0 * 0.7 * 1.0 * 0.5 * 0.6 * 0.5 * 0.7 * 1.0 * 0.5
    got = inside(g, words(g, "the dog chases the cat")[None, :]).log_z.data[0]
    np.testing.assert_allclose(got, np.log(p), atol=ATOL)


def test_vectorized_matches_naive_dense_batch():
    g = dense_grammar(seed=3, n_nt=4, n_t=5)
    lb, lt = g.log_probs_numpy()
    rng = np.random.default_rng(1)
    sentences = rng.integers(0, g.n_terminals, size=(4, 6))
    result = inside(g, sentences)
    for b in range(4):
        beta = inside_naive(lb, lt, sentences[b])
        np.testing.assert_allclose(result.chart_array(b), beta, atol=ATOL)


def test_vectorized_matches_naive_sparse_with_unreachable_cells():
    g = sparse_grammar()
    lb, lt = g.log_probs_numpy()
    sentence = words(g, "the dog chases the cat")
    result = inside(g, sentence[None, :])
    beta = inside_naive(lb, lt, sentence)
    assert np.isneginf(beta).any(), "expected unreachable cells in this chart"
    np.testing.assert_allclose(result.chart_array(0), beta, atol=ATOL)


def test_batching_is_consistent_with_single_sentences():
    g = dense_grammar(seed=5)
    rng = np.random.default_rng(2)
    sentences = rng.integers(0, g.n_terminals, size=(3, 5))
    batched = inside(g, sentences).log_z.data
    singles = [inside(g, s[None, :]).log_z.data[0] for s in sentences]
    np.testing.assert_allclose(batched, singles, atol=ATOL)


def test_unparseable_sentence_has_neg_inf_log_z_and_nan_free_gradient():
    g = sparse_grammar()
    result = inside(g, words(g, "the the")[None, :])
    assert np.isneginf(result.log_z.data[0])
    with np.errstate(invalid="raise", over="raise", divide="raise"):
        result.log_z.sum().backward()
    for p in g.parameters():
        assert np.all(np.isfinite(p.grad))
        np.testing.assert_array_equal(p.grad, 0.0)


# ------------------------------------------------------------------ sampler


def test_sampler_is_deterministic_and_valid():
    g = sparse_grammar()
    a, rate_a = g.sample(np.random.default_rng(0), 20, max_len=8)
    b, rate_b = g.sample(np.random.default_rng(0), 20, max_len=8)
    assert [x.tolist() for x in a] == [x.tolist() for x in b]
    assert rate_a == rate_b
    for s in a:
        assert 1 <= len(s) <= 8
        assert np.isfinite(inside(g, s[None, :]).log_z.data[0])


# --------------------------------------------------------- gradient vs outside


def _check_gradient_against_outside(g: PCFG, sentences: np.ndarray) -> None:
    lb, lt = g.log_probs_numpy()
    result = inside(g, sentences)
    result.log_z.sum().backward()

    total_counts_b = np.zeros_like(lb)
    total_counts_t = np.zeros_like(lt)
    for b in range(len(sentences)):
        beta = result.chart_array(b)
        alpha = outside(lb, lt, sentences[b], beta)
        log_z = beta[0, len(sentences[b]), 0]
        mu = span_posteriors(alpha, beta, log_z)
        # d log Z / d log beta(A, i, j) == mu(A, i, j) (see experiments/inside_outside.py)
        np.testing.assert_allclose(result.chart_grad_array(b), mu, atol=ATOL)
        assert 0.0 < mu.max() <= 1.0 + ATOL
        cb, ct = expected_counts(lb, lt, sentences[b], beta, alpha, log_z)
        total_counts_b += cb
        total_counts_t += ct

    # d log Z / d log p(rule) == expected count of the rule, summed over the batch.
    np.testing.assert_allclose(result.log_binary.grad, total_counts_b, atol=ATOL)
    np.testing.assert_allclose(result.log_terminal.grad, total_counts_t, atol=ATOL)
    # Expected counts of a sentence's parse sum to 2n - 1 rule applications.
    n = sentences.shape[1]
    np.testing.assert_allclose(
        total_counts_b.sum() + total_counts_t.sum(), len(sentences) * (2 * n - 1), atol=ATOL
    )


def test_gradient_equals_outside_sparse():
    g = sparse_grammar()
    sentences = np.stack([words(g, "the dog chases the cat"), words(g, "the cat sees the dog")])
    _check_gradient_against_outside(g, sentences)


def test_gradient_equals_outside_dense():
    g = dense_grammar(seed=11, n_nt=4, n_t=5)
    rng = np.random.default_rng(7)
    _check_gradient_against_outside(g, rng.integers(0, g.n_terminals, size=(3, 7)))
