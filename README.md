# grad — reverse-mode autodiff from scratch, with a PCFG on top

A small autodiff engine written against NumPy, and enough of a training stack to use it: an
MLP that reaches 97.84% on MNIST, and a probabilistic context-free grammar whose
inside-outside algorithm falls out of the engine's backward pass rather than being written by
hand.

**The engine is NumPy/CPU-only.** There is no GPU support, no CUDA path and no device
concept at all; every tensor is a `float64` NumPy array in host memory. It is not a
replacement for PyTorch and is slower than PyTorch on the same problem (numbers below). The
point is that the derivatives are correct and the inside-outside identities hold to machine
precision, not that it is fast.

```
grad/tensor.py       Tensor: ndarray + autodiff DAG, reverse topological backward
grad/ops.py          the ops and their backward rules (add..logsumexp, matmul, slice, stack)
grad/nn.py           Module, Linear, ReLU, Sequential
grad/optim.py        SGD (momentum, weight decay), Adam
grad/losses.py       softmax cross-entropy, computed in log space
grammar/pcfg.py      PCFG in Chomsky normal form: rules, sampling, log-likelihood
grammar/inside.py    vectorized inside algorithm, built from engine ops
grammar/outside.py   outside quantities computed by hand in numpy (the reference)
grammar/naive.py     unvectorized inside, loop per cell (the reference)
grammar/enumerate.py brute-force Z by enumerating every parse tree (the reference)
tests/gradcheck.py   central finite differences against a random cotangent
examples/mnist.py    MLP on MNIST
experiments/inside_outside.py  the autodiff-equals-outside identities
experiments/recover.py         recover a known grammar by gradient descent
experiments/bench.py           vectorized vs naive vs PyTorch
```

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest -q                    # 282 tests, incl. 156 gradchecks
.venv/bin/python -m experiments.inside_outside   # the three identities, max error ~7e-15
.venv/bin/python -m experiments.recover          # KL 3.2381 -> 1.6541, ~5 s
.venv/bin/python examples/mnist.py               # 97.84% test accuracy, ~17 s
.venv/bin/python -m experiments.bench            # the benchmark table
```

MNIST downloads itself into `data/mnist` (~12 MB, gitignored) on first run. The benchmark's
PyTorch column is skipped unless torch is installed; it is not a dependency of the engine:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Every number in this README came from those commands on the machine described under
[Benchmark](#benchmark), and each is reproducible from a clean checkout with a fixed seed.

## The engine

`Tensor` is a NumPy array plus the bookkeeping for reverse-mode autodiff. The forward pass
builds a DAG: each op that touches a tensor requiring a gradient returns a new tensor holding
the op instance (which saved whatever its backward rule needs) and its parents. `backward`
walks that DAG in reverse topological order and accumulates into `.grad`.

Implemented ops: `add`, `sub`, `mul`, `div`, `pow`, `neg`, `exp`, `log`, `relu`, `matmul`,
`sum`, `mean`, `max`, `logsumexp`, `reshape`, `transpose`, `stack`, and slicing. Broadcasting
is undone in the backward pass by summing over the broadcast axes. `logsumexp` and
softmax cross-entropy are computed in log space with the max subtracted, so the PCFG work
below stays numerically sound at sentence lengths where the probabilities underflow.

On top: `Linear` / `ReLU` / `Sequential`, `SGD` with momentum and weight decay, and `Adam`.

## Correctness: finite differences at 1e-6

Every op's backward rule is checked against central finite differences with `eps = 1e-6` and
a tolerance of `1e-6` (`tests/gradcheck.py`). The output is contracted with a **fixed random
cotangent** before differencing, so the check covers the whole Jacobian rather than only its
row sums — a rule that is wrong in a way that cancels under a uniform cotangent still fails.

The suite runs **156 gradchecks** across ops, layers, losses and the grammar. The worst error
observed over all of them is **1.67e-09** — about 600× inside the tolerance:

```
.venv/bin/python -m pytest -q      # 282 passed
```

The grammar has two independent references beyond finite differences: `grammar/naive.py`
recomputes the inside chart with an explicit loop per cell, and `grammar/enumerate.py`
computes Z by enumerating every parse tree, sharing no code with the inside algorithm. Both
are asserted against the vectorized implementation in `tests/test_grammar.py`.

## MNIST

A 784 → 256 → 10 MLP, ReLU, Adam at `lr=1e-3`, batch 128, 5 epochs, seed 0, on the standard
normalization (`mean 0.1307`, `std 0.3081`).

| epoch | train loss | test accuracy | time |
|---:|---:|---:|---:|
| 1 | 0.2447 | 95.96% | 3.7 s |
| 2 | 0.0985 | 96.69% | 3.0 s |
| 3 | 0.0655 | 97.60% | 3.6 s |
| 4 | 0.0459 | 97.87% | 3.2 s |
| 5 | 0.0349 | **97.84%** | 3.1 s |

**97.84% final test accuracy in 16.6 s of training.** Accuracy peaked at epoch 4 (97.87%) and
the reported figure is the final epoch, not the best one. This is an ordinary number for this
architecture — a plain MLP on MNIST is a solved problem, and the result is here to show the
engine trains a real model end to end, not because the accuracy is interesting.

## PCFG: inside-outside falls out of the backward pass

For a PCFG in Chomsky normal form, the inside algorithm computes

```
beta(A, i, j) = sum_{k, B, C} p(A -> B C) beta(B, i, k) beta(C, k, j)
```

and Z is the root cell. Fix one cell and treat its inside value as a free variable: every
tree containing `A` over span `(i, j)` splits into the subtree below the cell and everything
outside it, and no tree uses the cell twice, so **Z is linear in `beta(A, i, j)`** and the
coefficient is exactly the outside value `alpha(A, i, j)`. The chart stores log-values and
the loss is `log Z`, so the chain rule gives the posterior `mu(A, i, j) = alpha*beta/Z`. The
same argument applied to a rule probability gives the expected rule count — the E-step of
inside-outside.

So the backward pass *is* the outside pass. `experiments/inside_outside.py` computes the
outside quantities by hand in NumPy (`grammar/outside.py`, which shares no code with the
engine) and reports the maximum absolute disagreement on three identities:

| identity | what it says | sparse toy | dense random |
|---|---|---:|---:|
| `d logZ / d log beta = alpha*beta/Z` | the gradient on a chart cell is that span's posterior | 0.00e+00 | 5.33e-15 |
| `log(grad * Z / beta) = log alpha` | outside values are recoverable from the gradient | 2.22e-16 | 7.11e-15 |
| `d logZ / d log p(rule) = E[count]` | the gradient on a rule is its expected count | 0.00e+00 | 5.33e-15 |

Sparse toy: batch 2, length 5, N=6, V=6 — 180 chart cells, 156 of them unreachable. Dense
random: batch 3, length 8, N=4, V=6 — 432 cells, none unreachable. The rule identity is
checked separately for binary and terminal rules; the worse of the two is shown. The
unreachable cells are why the second identity is stated over reachable cells only: `beta` is
`-inf` there and the ratio is undefined, which the code handles rather than papers over.

**Everything is at machine precision (≤ 7.11e-15).** These are not approximations that
happen to agree — they are the same quantity computed two ways.

## Recovering a known grammar

`experiments/recover.py` samples a corpus from a known grammar and fits a fresh one to it by
gradient descent on corpus log-likelihood, then measures `KL(true || learned)` per
nonterminal. Non-start nonterminals are identifiable only up to relabeling, so the KL is
minimized over permutations of them.

Setup: 3 nonterminals, 6 terminals, Dirichlet-peaked true rules with ≥55% terminal mass
(subcritical, so sampling terminates), 2000 sentences of length ≤ 10, 300 full-batch Adam
steps at `lr=0.05`, seed 0.

| | N0 | N1 | N2 | total |
|---|---:|---:|---:|---:|
| KL before | 1.0854 | 1.2099 | 0.9428 | **3.2381** |
| KL after | 0.2985 | 0.7692 | 0.5864 | **1.6541** |

**KL falls from 3.24 to 1.65 in 4.5 s.** Mean log-likelihood per sentence goes from -5.0178
at step 1 to -3.5511, against the true grammar's -3.5625 on the same corpus.

Two things that are worth reading honestly rather than as a win:

- **The learned grammar beats the true grammar's likelihood** (-3.5511 vs -3.5625). That is
  what fitting a finite sample looks like; it is evidence the optimizer worked, not evidence
  the generator was recovered.
- **The KL does not go to zero, and is not expected to.** Sampled sentences average 1.68
  tokens, so most of the corpus is one or two words and carries little information about the
  binary rules. The residual KL is concentrated in `N1` and `N2`, the symbols the short
  sentences barely exercise.

## Benchmark

Vectorized inside on the engine, against the unvectorized loop and against the identical
algorithm written in PyTorch ops. Best of 3 runs. Shapes are `(batch, sentence length,
nonterminals, terminals)`.

Measured on a 13th Gen Intel Core i7-13700KF (24 threads; torch used 12), NumPy 2.5.3,
torch 2.14.0+cpu. **Both columns are CPU** — the CPU-only torch wheel is the honest reference
for a NumPy engine, and the comparison would be meaningless against a GPU build.

| (batch, n, N, V) | engine fwd | engine fwd+bwd | naive fwd | torch fwd+bwd |
|---|---:|---:|---:|---:|
| (1, 8, 4, 6) | 0.001 s | 0.001 s | 0.001 s | 0.001 s |
| (8, 12, 6, 10) | 0.004 s | 0.009 s | 0.045 s | 0.003 s |
| (8, 16, 8, 12) | 0.016 s | 0.040 s | 0.174 s | 0.009 s |
| (4, 20, 8, 12) | 0.017 s | 0.039 s | 0.167 s | 0.013 s |
| (2, 30, 6, 10) | 0.017 s | 0.041 s | 0.180 s | 0.018 s |

Three things to read off it:

- **Vectorizing the chart is worth about 10×** (9.8× to 11.2×) — 0.016 s against 0.174 s at
  `(8, 16, 8, 12)`, forward only. That is the gap between filling the chart cell by cell in Python and filling
  a whole diagonal with array ops.
- **Forward+backward costs about 2.4× forward alone** (2.25× to 2.50× across the four larger
  shapes), so the backward pass is a little more expensive than the forward — the expected
  shape for reverse mode.
- **PyTorch is 2.3× to 4.4× faster on forward+backward, and the gap closes as the problem
  grows**: 4.4× at `(8, 16, 8, 12)` down to 2.3× at `(2, 30, 6, 10)`. The engine's overhead is per-op Python
  dispatch, so it amortizes as the arrays get bigger. It is behind, and it is not embarrassed.

The benchmark also serves as a correctness check: it asserts the engine, the naive loop and
the PyTorch implementation agree on `log Z` to `1e-10` at every shape, so a regression in any
of the three fails the run rather than quietly producing a faster wrong number.

## What this is not

- **No GPU, no float32, no device concept.** NumPy `float64` on the CPU, everywhere.
- **No convolutions, no recurrence, no attention.** The layer set is `Linear` and `ReLU`.
  MNIST is an MLP because that is what exists.
- **No graph optimization**: no fusion, no in-place reuse, no checkpointing. Every op
  allocates its output, and the whole DAG is retained until `backward` runs.
- **Small scale by construction.** The PCFG work runs at sentence lengths ≤ 30 and a handful
  of nonterminals; the inside algorithm is `O(n^3 N^3)` and nothing here changes that.

CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check` and `pytest -q` on
push. The test suite needs nothing but NumPy and pytest.
