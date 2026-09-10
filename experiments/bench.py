"""Vectorized inside on the engine vs naive loops vs PyTorch (if installed)."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import env  # noqa: E402
from grammar import PCFG, inside  # noqa: E402
from grammar.naive import log_z_naive  # noqa: E402

SHAPES = [  # (batch, n, N, V)
    (1, 8, 4, 6),
    (8, 12, 6, 10),
    (8, 16, 8, 12),
    (4, 20, 8, 12),
    (2, 30, 6, 10),
]
REPEATS = 3


def best_of(fn, repeats: int = REPEATS):
    times = []
    value = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - t0)
    return min(times), value


def inside_torch(log_binary, log_terminal, sentences):
    """The identical algorithm written with torch ops; returns (log_z, leaves)."""
    import torch

    lb = torch.tensor(log_binary, requires_grad=True)
    lt = torch.tensor(log_terminal, requires_grad=True)
    sent = torch.tensor(sentences)
    batch, n = sent.shape
    n_nt = lb.shape[0]
    chart = [None] * (n + 1)
    chart[1] = lt[:, sent].permute(1, 2, 0)
    rules = lb.reshape(1, 1, 1, n_nt, n_nt, n_nt)
    for length in range(2, n + 1):
        spans = n - length + 1
        pairs = [
            chart[left][:, :spans].reshape(batch, spans, n_nt, 1)
            + chart[length - left][:, left : left + spans].reshape(batch, spans, 1, n_nt)
            for left in range(1, length)
        ]
        children = torch.stack(pairs, dim=2).reshape(batch, spans, length - 1, 1, n_nt, n_nt)
        chart[length] = torch.logsumexp(children + rules, dim=(2, 4, 5))
    return chart[n][:, 0, 0], (lb, lt)


def run_shape(
    shape: tuple[int, int, int, int], rng: np.random.Generator, has_torch: bool
) -> tuple[str, dict[str, float]]:
    batch, n, n_nt, n_t = shape
    g = PCFG.random(n_nt, n_t, rng)
    lb, lt = g.log_probs_numpy()
    sentences = rng.integers(0, n_t, size=(batch, n))

    def fwd():
        return inside((lb, lt), sentences).log_z.data.copy()

    def fwd_bwd():
        res = inside(g, sentences)
        res.log_z.sum().backward()
        return res.log_z.data.copy()

    def naive():
        return np.array([log_z_naive(lb, lt, s) for s in sentences])

    t_fwd, log_z = best_of(fwd)
    t_fwd_bwd, log_z_2 = best_of(fwd_bwd)
    t_naive, z_naive = best_of(naive, repeats=1)
    assert np.allclose(log_z, z_naive, atol=1e-10) and np.allclose(log_z, log_z_2, atol=1e-10)
    timings = {"fwd": t_fwd, "fwd_bwd": t_fwd_bwd, "naive": t_naive}
    row = f"{str(shape):>18} {t_fwd:>10.3f}s {t_fwd_bwd:>14.3f}s {t_naive:>10.3f}s"
    if has_torch:

        def torch_fwd_bwd():
            z, _ = inside_torch(lb, lt, sentences)
            z.sum().backward()
            return z.detach().numpy().copy()

        t_torch, z_torch = best_of(torch_fwd_bwd)
        assert np.allclose(log_z, z_torch, atol=1e-10)
        row += f" {t_torch:>13.3f}s"
        timings["torch_fwd_bwd"] = t_torch
    return row, timings


def format_ratios(rows: list[tuple[tuple[int, int, int, int], dict[str, float]]]) -> str:
    """The comparisons the numbers are actually for, computed rather than left to the reader.

    Sub-millisecond shapes are excluded: at 1 ms the timer's own resolution is a large part
    of the measurement, and a ratio of two such numbers says more about the clock than about
    the code.
    """
    usable = [(s, x) for s, x in rows if x["fwd"] >= 0.002]
    if not usable:
        return "no shape ran long enough to take a meaningful ratio"
    lines = [
        f"{'(batch, n, N, V)':>18} {'naive/engine fwd':>17} {'fwd+bwd / fwd':>15}"
        f"{'  engine/torch fwd+bwd':>22}"
    ]
    for shape, x in usable:
        torch_ratio = (
            f"{x['fwd_bwd'] / x['torch_fwd_bwd']:>21.1f}x" if "torch_fwd_bwd" in x else f"{'-':>22}"
        )
        lines.append(
            f"{str(shape):>18} {x['naive'] / x['fwd']:>16.1f}x {x['fwd_bwd'] / x['fwd']:>14.2f}x"
            f"{torch_ratio}"
        )
    span = lambda f: (min(f(x) for _, x in usable), max(f(x) for _, x in usable))  # noqa: E731
    lo, hi = span(lambda x: x["naive"] / x["fwd"])
    lines.append(f"\nvectorizing the chart is worth {lo:.1f}x to {hi:.1f}x")
    lo, hi = span(lambda x: x["fwd_bwd"] / x["fwd"])
    lines.append(f"forward+backward costs {lo:.2f}x to {hi:.2f}x forward alone")
    if all("torch_fwd_bwd" in x for _, x in usable):
        lo, hi = span(lambda x: x["fwd_bwd"] / x["torch_fwd_bwd"])
        lines.append(f"PyTorch is {lo:.1f}x to {hi:.1f}x faster on forward+backward")
    return "\n".join(lines)


def main() -> None:
    has_torch = importlib.util.find_spec("torch") is not None
    print(env.render(env.collect()))
    print()
    if not has_torch:
        print("torch not available: the PyTorch column is skipped")
    header = f"{'(batch, n, N, V)':>18} {'engine fwd':>11} {'engine fwd+bwd':>15} {'naive fwd':>11}"
    if has_torch:
        header += f" {'torch fwd+bwd':>14}"
    print(header)
    rng = np.random.default_rng(0)
    rows = []
    for shape in SHAPES:
        row, timings = run_shape(shape, rng, has_torch)
        print(row)
        rows.append((shape, timings))
    print()
    print(format_ratios(rows))


if __name__ == "__main__":
    main()
