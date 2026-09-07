"""SGD and Adam against hand-computed updates."""

from __future__ import annotations

import numpy as np

from grad import Tensor, optim


def _param(values, grad):
    p = Tensor(np.array(values, dtype=np.float64), requires_grad=True)
    p.grad = np.array(grad, dtype=np.float64)
    return p


def test_sgd_plain():
    p = _param([1.0, -2.0], [0.5, 0.25])
    optim.SGD([p], lr=0.1).step()
    np.testing.assert_allclose(p.data, [1.0 - 0.05, -2.0 - 0.025])


def test_sgd_momentum_two_steps():
    p = _param([1.0], [2.0])
    opt = optim.SGD([p], lr=0.1, momentum=0.9)
    opt.step()  # v = 2.0            -> p = 1.0 - 0.1 * 2.0 = 0.8
    np.testing.assert_allclose(p.data, [0.8])
    p.grad = np.array([1.0])
    opt.step()  # v = 0.9 * 2 + 1 = 2.8 -> p = 0.8 - 0.28 = 0.52
    np.testing.assert_allclose(p.data, [0.52])


def test_adam_first_step_matches_hand_computation():
    # After bias correction at t = 1: m_hat = g, v_hat = g^2, so the update is
    # lr * g / (|g| + eps), i.e. a step of about lr in the direction of -sign(g).
    lr, eps = 1e-3, 1e-8
    p = _param([1.0, -1.0, 0.5], [0.3, -2.0, 1e-4])
    g = p.grad.copy()
    optim.Adam([p], lr=lr, eps=eps).step()
    expected = np.array([1.0, -1.0, 0.5]) - lr * g / (np.abs(g) + eps)
    np.testing.assert_allclose(p.data, expected, rtol=0, atol=1e-15)
    # Without bias correction the step would be lr * 0.1 * g / (sqrt(0.001) * |g|)
    # ~ 0.0316 * lr, so this also fails if the correction is missing.
    assert np.allclose(np.abs(p.data - [1.0, -1.0, 0.5]), lr, atol=1e-6)


def test_adam_second_step_matches_hand_computation():
    lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
    theta0 = np.array([0.7, -0.3])
    g1 = np.array([0.5, -1.5])
    g2 = np.array([-0.25, 2.0])

    p = _param(theta0, g1)
    opt = optim.Adam([p], lr=lr, betas=(b1, b2), eps=eps)
    opt.step()
    p.grad = g2.copy()
    opt.step()

    # Kingma & Ba, Algorithm 1, expanded by hand for t = 1, 2.
    m1 = (1 - b1) * g1
    v1 = (1 - b2) * g1**2
    theta1 = theta0 - lr * (m1 / (1 - b1**1)) / (np.sqrt(v1 / (1 - b2**1)) + eps)
    m2 = b1 * m1 + (1 - b1) * g2
    v2 = b2 * v1 + (1 - b2) * g2**2
    theta2 = theta1 - lr * (m2 / (1 - b1**2)) / (np.sqrt(v2 / (1 - b2**2)) + eps)
    np.testing.assert_allclose(p.data, theta2, rtol=0, atol=1e-15)
    assert opt.t == 2


def test_optimizer_skips_params_without_grad_and_zero_grad_clears():
    p = _param([1.0], [1.0])
    q = Tensor([2.0], requires_grad=True)  # never received a gradient
    opt = optim.Adam([p, q], lr=0.1)
    opt.step()
    np.testing.assert_array_equal(q.data, [2.0])
    assert p.data[0] < 1.0
    opt.zero_grad()
    assert p.grad is None and q.grad is None
