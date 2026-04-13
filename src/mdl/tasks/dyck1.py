"""Dyck-1 task: well-balanced single-type brackets.

PCFG: S → [S]S (prob 1/3) | ε (prob 2/3).
Alphabet: {#, (, )} encoded as 0, 1, 2.
Max training string length: 200.

Test set: all valid Dyck-1 strings of length ≤ 10 (exhaustive).

Reference: Abudy et al. (2025, "Learning Formal Languages with Small
RNNs Using MDL Regularization", Appendix A.1, Table 6).
"""

from __future__ import annotations

import numpy as np

from .base import TaskSpec

SYMBOL_HASH = 0
SYMBOL_OPEN = 1
SYMBOL_CLOSE = 2

# PCFG probabilities
P_RECURSE = 1 / 3  # S → (S)S
P_EMPTY = 2 / 3    # S → ε


class Dyck1Task(TaskSpec):
    """Dyck-1 (single-type balanced brackets) task."""

    def __init__(self, p: float = 1 / 3):
        # p here is P_RECURSE for the PCFG (not geometric like anbn)
        super().__init__(
            name="dyck1",
            alphabet=["#", "(", ")"],
            num_symbols=3,
            p=p,
        )

    def generate_strings(
        self, num_strings: int, seed: int = 0,
        max_length: int = 200,
    ) -> list[list[int]]:
        """Sample from the Dyck-1 PCFG.

        S → (S)S  with prob p
        S → ε     with prob 1-p

        Strings exceeding max_length are discarded and resampled.
        """
        rng = np.random.RandomState(seed)
        strings = []
        while len(strings) < num_strings:
            body = self._sample_S(rng)
            if len(body) <= max_length:
                strings.append([SYMBOL_HASH] + body + [SYMBOL_HASH])
        return strings

    def _sample_S(self, rng: np.random.RandomState) -> list[int]:
        """Recursively sample from S."""
        if rng.rand() < self.p:
            # S → (S1)S2
            s1 = self._sample_S(rng)
            s2 = self._sample_S(rng)
            return [SYMBOL_OPEN] + s1 + [SYMBOL_CLOSE] + s2
        else:
            return []

    def make_fixed_n(self, n: int) -> tuple[list[int], list[int]]:
        """Canonical test string: (^n )^n."""
        s = [SYMBOL_HASH] + [SYMBOL_OPEN] * n + [SYMBOL_CLOSE] * n + [SYMBOL_HASH]
        return s[:-1], s[1:]

    def make_test_set(
        self, max_n: int | None = None, max_length: int = 10,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Exhaustive test set: all valid Dyck-1 strings of length ≤ max_length.

        Overrides the base class method because Dyck-1 test sets enumerate
        by string length (all valid structures), not by a single parameter n.

        Following Abudy et al. (2025): test set is all valid Dyck-1 strings
        of length ≤ 10.
        """
        all_strings = self._enumerate_dyck1(max_length)
        inputs, targets = [], []
        for body in all_strings:
            s = [SYMBOL_HASH] + body + [SYMBOL_HASH]
            inputs.append(s[:-1])
            targets.append(s[1:])
        return inputs, targets

    def _enumerate_dyck1(self, max_length: int) -> list[list[int]]:
        """Enumerate all valid Dyck-1 strings of length ≤ max_length.

        Uses the PCFG structure: S → (S)S | ε.
        A Dyck-1 string of length 2k has Catalan(k) distinct parse trees,
        but many share the same string. We enumerate unique strings.
        """
        results = set()
        self._enumerate_S(max_length, [], results)
        # Sort by length then lexicographically for determinism
        sorted_results = sorted(results, key=lambda s: (len(s), s))
        return [list(s) for s in sorted_results]

    def _enumerate_S(
        self, remaining: int, prefix: list[int], results: set[tuple[int, ...]],
    ) -> None:
        """Recursively enumerate all Dyck-1 derivations within length budget."""
        # S → ε
        results.add(tuple(prefix))

        # S → (S1)S2: need at least 2 characters for ()
        if remaining >= 2:
            # For each possible S1 (inside the brackets)
            s1_options = set()
            self._enumerate_S(remaining - 2, [], s1_options)
            for s1 in s1_options:
                inner = list(s1)
                if len(inner) + 2 > remaining:
                    continue
                partial = prefix + [SYMBOL_OPEN] + inner + [SYMBOL_CLOSE]
                # For each possible S2 (after the brackets)
                self._enumerate_S(remaining - len(inner) - 2, partial, results)

    def compute_grammar_weights(
        self, max_n: int = 0, min_n: int = 0, max_length: int = 10,
    ) -> np.ndarray:
        """PCFG probability for each string in the exhaustive test set.

        Unlike aⁿbⁿ (parameterized by n), Dyck-1 test strings have
        variable structure. Each string's weight is its PCFG probability.
        """
        all_strings = self._enumerate_dyck1(max_length)
        weights = np.array([self._string_pcfg_prob(s) for s in all_strings])
        return weights

    def _string_pcfg_prob(self, body: list[int]) -> float:
        """Compute the total PCFG probability of a Dyck-1 string.

        A string may have multiple derivation trees; the probability is
        the sum over all derivations. For Dyck-1, each string of length 2k
        has a unique leftmost derivation, so we can parse unambiguously.
        """
        prob, _ = self._parse_S_prob(body, 0)
        return prob

    def _parse_S_prob(
        self, body: list[int], pos: int,
    ) -> tuple[float, int]:
        """Parse S from position pos, return (probability, end_position).

        S → (S1)S2 with prob p
        S → ε      with prob 1-p
        """
        if pos >= len(body) or body[pos] != SYMBOL_OPEN:
            # S → ε
            return (1 - self.p), pos

        # S → (S1)S2
        prob = self.p
        pos += 1  # consume (
        # Parse S1
        p1, pos = self._parse_S_prob(body, pos)
        prob *= p1
        assert pos < len(body) and body[pos] == SYMBOL_CLOSE
        pos += 1  # consume )
        # Parse S2
        p2, pos = self._parse_S_prob(body, pos)
        prob *= p2
        return prob, pos

    def compute_target_probs(self) -> dict:
        """Optimal output probabilities per context.

        The optimal LM for Dyck-1 with PCFG S → (S)S (p) | ε (1-p):
        - After #:           P(() = p, P(#) = 1-p
        - After ( at depth d>0: P(() = p, P()) = 1-p
          (Inside brackets, we're generating a new S which recurses with
          prob p or terminates with prob 1-p)
        - After ) at depth d>1: P(() = p, P()) = 1-p
          (After closing, we're generating S2 which has same distribution)
        - After ) at depth 1: P(() = p, P(#) = 1-p
          (Back at top level, generating trailing S)
        """
        p = self.p
        return {
            "top_level": np.array([1 - p, p, 0.0], dtype=np.float32),
            "inside": np.array([0.0, p, 1 - p], dtype=np.float32),
            "after_close_deep": np.array([0.0, p, 1 - p], dtype=np.float32),
            "after_close_top": np.array([1 - p, p, 0.0], dtype=np.float32),
        }

    def generate_negative_examples(
        self, num: int, max_n: int = 10, seed: int = 42,
    ) -> list[list[int]]:
        rng = np.random.RandomState(seed)
        strings = []
        per_type = num // 3
        remainder = num - 3 * per_type

        def _wrap(body):
            return [SYMBOL_HASH] + body + [SYMBOL_HASH]

        # Type 1: unbalanced (more opens than closes or vice versa)
        for _ in range(per_type + remainder):
            n = rng.randint(1, max(max_n, 2))
            if rng.random() < 0.5:
                body = [SYMBOL_OPEN] * n + [SYMBOL_CLOSE] * (n + rng.randint(1, 3))
            else:
                body = [SYMBOL_OPEN] * (n + rng.randint(1, 3)) + [SYMBOL_CLOSE] * n
            strings.append(_wrap(body))

        # Type 2: wrong nesting (closes before matching open)
        for _ in range(per_type):
            n = rng.randint(1, max(max_n, 2))
            body = [SYMBOL_CLOSE] + [SYMBOL_OPEN] * n + [SYMBOL_CLOSE] * (n - 1)
            strings.append(_wrap(body))

        # Type 3: random brackets (mostly invalid)
        for _ in range(per_type):
            length = rng.randint(1, max(max_n * 2, 3))
            body = [rng.choice([SYMBOL_OPEN, SYMBOL_CLOSE]) for _ in range(length)]
            # Check if accidentally valid
            depth = 0
            valid = True
            for c in body:
                if c == SYMBOL_OPEN:
                    depth += 1
                else:
                    depth -= 1
                if depth < 0:
                    valid = False
                    break
            if valid and depth == 0 and len(body) > 0:
                body[0] = SYMBOL_CLOSE  # break validity
            strings.append(_wrap(body))

        inputs = [s[:-1] for s in strings]
        targets = [s[1:] for s in strings]
        return inputs, targets

    def _string_n(self, s: list[int]) -> int:
        """For Dyck-1, n is the number of opening brackets."""
        return s.count(SYMBOL_OPEN)
