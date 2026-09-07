"""A PCFG in Chomsky normal form with log-space rule weights as engine leaves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from grad import Tensor, logsumexp, stack


class PCFG:
    """Nonterminals ``0..N-1`` (0 is the start symbol), terminals ``0..V-1``.

    Rules are stored as two unnormalized log-weight tables, both leaf Tensors:

    * ``binary_scores[A, B, C]`` for A -> B C, shape ``(N, N, N)``
    * ``terminal_scores[A, w]`` for A -> w, shape ``(N, V)``

    ``-inf`` marks an absent rule. ``log_probs`` normalizes each nonterminal's
    rules (binary and terminal together) into one distribution inside the
    graph, so the leaves can be trained without leaving the simplex.
    """

    def __init__(
        self,
        nonterminals: Sequence[str],
        terminals: Sequence[str],
        binary_scores: np.ndarray | Tensor,
        terminal_scores: np.ndarray | Tensor,
    ) -> None:
        self.nonterminals = list(nonterminals)
        self.terminals = list(terminals)
        n, v = len(self.nonterminals), len(self.terminals)
        b = binary_scores.data if isinstance(binary_scores, Tensor) else np.asarray(binary_scores)
        t = (
            terminal_scores.data
            if isinstance(terminal_scores, Tensor)
            else np.asarray(terminal_scores)
        )
        if b.shape != (n, n, n) or t.shape != (n, v):
            raise ValueError(f"expected shapes {(n, n, n)} and {(n, v)}, got {b.shape}, {t.shape}")
        has_rule = np.isfinite(b).any(axis=(1, 2)) | np.isfinite(t).any(axis=1)
        if not has_rule.all():
            missing = [self.nonterminals[i] for i in np.flatnonzero(~has_rule)]
            raise ValueError(f"nonterminals without any rule: {missing}")
        self.binary_scores = Tensor(b.astype(np.float64), requires_grad=True)
        self.terminal_scores = Tensor(t.astype(np.float64), requires_grad=True)

    # ----------------------------------------------------------- constructors

    @classmethod
    def from_rules(
        cls,
        nonterminals: Sequence[str],
        terminals: Sequence[str],
        binary: Mapping[tuple[str, str, str], float],
        terminal: Mapping[tuple[str, str], float],
    ) -> PCFG:
        """Sparse grammar from rule probabilities (or unnormalized weights)."""
        nt = {name: i for i, name in enumerate(nonterminals)}
        tm = {name: i for i, name in enumerate(terminals)}
        b = np.full((len(nt), len(nt), len(nt)), -np.inf)
        t = np.full((len(nt), len(tm)), -np.inf)
        for (a, left, right), p in binary.items():
            b[nt[a], nt[left], nt[right]] = np.log(p)
        for (a, w), p in terminal.items():
            t[nt[a], tm[w]] = np.log(p)
        return cls(nonterminals, terminals, b, t)

    @classmethod
    def random(
        cls,
        nonterminals: Sequence[str] | int,
        terminals: Sequence[str] | int,
        rng: np.random.Generator,
        scale: float = 1.0,
    ) -> PCFG:
        """Dense grammar with Gaussian log-weights (every rule has positive probability)."""
        if isinstance(nonterminals, int):
            nonterminals = [f"N{i}" for i in range(nonterminals)]
        if isinstance(terminals, int):
            terminals = [f"w{i}" for i in range(terminals)]
        n, v = len(nonterminals), len(terminals)
        b = rng.normal(0.0, scale, size=(n, n, n))
        t = rng.normal(0.0, scale, size=(n, v))
        return cls(nonterminals, terminals, b, t)

    # ------------------------------------------------------------- accessors

    @property
    def n_nonterminals(self) -> int:
        return len(self.nonterminals)

    @property
    def n_terminals(self) -> int:
        return len(self.terminals)

    def parameters(self) -> list[Tensor]:
        return [self.binary_scores, self.terminal_scores]

    def log_probs(self) -> tuple[Tensor, Tensor]:
        """Normalized ``(log_binary, log_terminal)`` as graph tensors.

        For each A: log Z_A = logsumexp over all of A's binary and terminal
        weights; both tables are shifted by it.
        """
        n = self.n_nonterminals
        log_z_binary = logsumexp(self.binary_scores, axis=(1, 2))  # (N,)
        log_z_terminal = logsumexp(self.terminal_scores, axis=1)  # (N,)
        log_z = logsumexp(stack([log_z_binary, log_z_terminal], axis=1), axis=1)  # (N,)
        log_binary = self.binary_scores - log_z.reshape(n, 1, 1)
        log_terminal = self.terminal_scores - log_z.reshape(n, 1)
        return log_binary, log_terminal

    def log_probs_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        lb, lt = self.log_probs()
        return lb.data.copy(), lt.data.copy()

    def rule_distributions(self) -> np.ndarray:
        """``(N, N*N + V)`` probabilities: row A is A's distribution over all its rules."""
        lb, lt = self.log_probs_numpy()
        n = self.n_nonterminals
        return np.concatenate([np.exp(lb).reshape(n, -1), np.exp(lt)], axis=1)

    # -------------------------------------------------------------- sampling

    def sample(
        self, rng: np.random.Generator, n_sentences: int, max_len: int
    ) -> tuple[list[np.ndarray], float]:
        """Sample sentences; those longer than ``max_len`` are rejected and redrawn.

        Returns the sentences (int arrays of terminal ids) and the rejection
        rate, so callers can report how much the truncation biases the corpus.
        """
        lb, lt = self.log_probs_numpy()
        n, v = self.n_nonterminals, self.n_terminals
        # Per nonterminal: one categorical over its N*N binary rules then V terminals.
        probs = np.concatenate([np.exp(lb).reshape(n, -1), np.exp(lt)], axis=1)
        probs /= probs.sum(axis=1, keepdims=True)
        n_binary = n * n

        sentences: list[np.ndarray] = []
        rejected = 0
        while len(sentences) < n_sentences:
            words: list[int] = []
            pending = [0]  # depth-first; the leftmost pending symbol is expanded first
            ok = True
            while pending:
                if len(words) + len(pending) > max_len:
                    ok = False
                    break
                a = pending.pop()
                choice = rng.choice(n_binary + v, p=probs[a])
                if choice < n_binary:
                    b, c = divmod(choice, n)
                    pending.append(c)  # push right child first so left is expanded next
                    pending.append(b)
                else:
                    words.append(choice - n_binary)
            if ok:
                sentences.append(np.array(words, dtype=np.int64))
            else:
                rejected += 1
        return sentences, rejected / (rejected + n_sentences)

    # ------------------------------------------------------------ likelihood

    def log_likelihood(self, corpus: Sequence[np.ndarray]) -> Tensor:
        """Summed log Z over the corpus as one graph tensor (sentences batched by length)."""
        from grammar.inside import inside

        log_binary, log_terminal = self.log_probs()
        by_length: dict[int, list[np.ndarray]] = {}
        for s in corpus:
            by_length.setdefault(len(s), []).append(np.asarray(s))
        total: Tensor | None = None
        for length in sorted(by_length):
            batch = np.stack(by_length[length])
            part = inside((log_binary, log_terminal), batch).log_z.sum()
            total = part if total is None else total + part
        assert total is not None, "empty corpus"
        return total
