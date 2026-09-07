"""Recover a known grammar's rule distributions from a sampled corpus by gradient descent.

The true grammar is dense but peaked (Dirichlet-sampled rule probabilities). A
fresh grammar with random weights is trained with Adam to maximize corpus
log-likelihood, and KL(p_true || p_learned) per nonterminal is reported before
and after. Nonterminals other than the start symbol are only identifiable up to
relabeling, so the KL is minimized over permutations of the non-start symbols.
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grad import optim  # noqa: E402
from grammar import PCFG  # noqa: E402

SEED = 0
N_NT, N_T = 3, 6
CORPUS_SIZE = 2000
MAX_LEN = 10
STEPS = 300
LR = 0.05


def true_grammar(rng: np.random.Generator) -> PCFG:
    """Peaked random rules with >= 55% terminal mass per nonterminal (subcritical)."""
    b = np.zeros((N_NT, N_NT, N_NT))
    t = np.zeros((N_NT, N_T))
    for a in range(N_NT):
        term_mass = rng.uniform(0.55, 0.8)
        b[a] = (1 - term_mass) * rng.dirichlet(0.3 * np.ones(N_NT * N_NT)).reshape(N_NT, N_NT)
        t[a] = term_mass * rng.dirichlet(0.3 * np.ones(N_T))
    return PCFG([f"N{i}" for i in range(N_NT)], [f"w{i}" for i in range(N_T)], np.log(b), np.log(t))


def kl_per_nonterminal(p: PCFG, q: PCFG) -> np.ndarray:
    """KL(p || q) for each nonterminal's rule distribution, min over relabelings."""
    pb, pt = (np.exp(x) for x in p.log_probs_numpy())
    qb, qt = (np.exp(x) for x in q.log_probs_numpy())
    best = None
    for perm in itertools.permutations(range(1, N_NT)):
        idx = np.array((0, *perm))
        qb_p = qb[np.ix_(idx, idx, idx)]
        qt_p = qt[idx]
        p_all = np.concatenate([pb.reshape(N_NT, -1), pt], axis=1)
        q_all = np.concatenate([qb_p.reshape(N_NT, -1), qt_p], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p_all > 0, p_all * (np.log(p_all) - np.log(q_all)), 0.0)
        kl = terms.sum(axis=1)
        if best is None or kl.sum() < best.sum():
            best = kl
    return best


def main() -> None:
    rng = np.random.default_rng(SEED)
    truth = true_grammar(rng)
    corpus, rejection = truth.sample(rng, CORPUS_SIZE, MAX_LEN)
    lengths = np.array([len(s) for s in corpus])
    print(
        f"corpus: {CORPUS_SIZE} sentences, mean length {lengths.mean():.2f}, "
        f"max {MAX_LEN}, rejection rate {rejection:.1%}"
    )
    true_ll = truth.log_likelihood(corpus).item() / CORPUS_SIZE
    print(f"true grammar mean log-likelihood per sentence: {true_ll:.4f}")

    learned = PCFG.random(N_NT, N_T, rng)
    opt = optim.Adam(learned.parameters(), lr=LR)
    kl_before = kl_per_nonterminal(truth, learned)
    print(
        f"KL(true || learned) per nonterminal before: {np.round(kl_before, 4)}  "
        f"total {kl_before.sum():.4f}"
    )

    start = time.perf_counter()
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        loss = -learned.log_likelihood(corpus) / CORPUS_SIZE
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            kl = kl_per_nonterminal(truth, learned)
            print(f"  step {step:4d}  mean log-lik {-loss.item():.4f}  KL total {kl.sum():.4f}")
    elapsed = time.perf_counter() - start

    kl_after = kl_per_nonterminal(truth, learned)
    learned_ll = learned.log_likelihood(corpus).item() / CORPUS_SIZE
    print(
        f"KL(true || learned) per nonterminal after:  {np.round(kl_after, 4)}  "
        f"total {kl_after.sum():.4f}  (min over relabelings of non-start symbols)"
    )
    print(f"mean log-likelihood per sentence: learned {learned_ll:.4f} vs true {true_ll:.4f}")
    print(f"training time {elapsed:.1f}s for {STEPS} full-batch Adam steps")


if __name__ == "__main__":
    main()
