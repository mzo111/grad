"""Log-space arithmetic edge cases for the logsumexp op."""

from __future__ import annotations

import numpy as np

from grad import Tensor, logsumexp


def test_empty_axis_is_neg_inf():
    x = Tensor(np.zeros((3, 0)))
    out = logsumexp(x, axis=1)
    assert out.shape == (3,)
    assert np.all(np.isneginf(out.data))
    assert np.isneginf(logsumexp(Tensor(np.zeros(0))).data)


def test_all_neg_inf_is_neg_inf():
    x = Tensor(np.full((2, 3), -np.inf))
    with np.errstate(all="raise"):
        out = logsumexp(x, axis=1)
    assert np.all(np.isneginf(out.data))


def test_widely_separated_magnitudes():
    # A naive exp(1000) would overflow here. Underflow of the small term to 0 is
    # the intended behaviour, so it is not trapped.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert logsumexp(Tensor([1000.0, -1000.0])).data == 1000.0
        assert logsumexp(Tensor([-1000.0, 1000.0])).data == 1000.0
        np.testing.assert_allclose(
            logsumexp(Tensor([-1000.0, -1000.0])).data, -1000.0 + np.log(2.0)
        )
        assert logsumexp(Tensor([0.0, -800.0])).data == 0.0
        np.testing.assert_allclose(logsumexp(Tensor([1e6, 1e6 + 1])).data, 1e6 + np.log1p(np.e))


def test_partial_neg_inf_row_matches_finite_subset():
    x = np.array([1.0, -np.inf, 2.0, -np.inf])
    np.testing.assert_allclose(
        logsumexp(Tensor(x)).data, np.log(np.exp(1.0) + np.exp(2.0)), rtol=1e-15
    )


def test_gradient_is_softmax_and_zero_at_neg_inf():
    x = Tensor([[1.0, -np.inf, 2.0], [-np.inf, -np.inf, -np.inf]], requires_grad=True)
    with np.errstate(all="raise"):
        out = logsumexp(x, axis=1)
        out.backward(np.array([1.0, 1.0]))
    assert np.all(np.isfinite(x.grad))
    expected_row0 = np.exp([1.0, -np.inf, 2.0]) / (np.exp(1.0) + np.exp(2.0))
    np.testing.assert_allclose(x.grad[0], expected_row0, rtol=1e-15)
    np.testing.assert_array_equal(x.grad[1], 0.0)


def test_gradient_through_empty_axis_is_nan_free():
    x = Tensor(np.zeros((2, 0)), requires_grad=True)
    logsumexp(x, axis=1).sum().backward()
    assert x.grad.shape == (2, 0)
