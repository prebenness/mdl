"""aⁿbⁿcⁿ task: the language {#aⁿbⁿcⁿ# | n ≥ 0}.

PCFG: sample n ~ Geometric(p), produce aⁿbⁿcⁿ.
Alphabet: {#, a, b, c} encoded as 0, 1, 2, 3.

Reference: Abudy et al. (2025, "Learning Formal Languages with Small
RNNs Using MDL Regularization", Appendix A.1). Same geometric
distribution as aⁿbⁿ. Test set: n=1 to n_max_train + 1000.
"""

from __future__ import annotations

import numpy as np

from .base import TaskSpec

SYMBOL_HASH = 0
SYMBOL_A = 1
SYMBOL_B = 2
SYMBOL_C = 3


class AnbncnTask(TaskSpec):
    """aⁿbⁿcⁿ formal language task."""

    def __init__(self, p: float = 0.3):
        super().__init__(
            name="anbncn",
            alphabet=["#", "a", "b", "c"],
            num_symbols=4,
            p=p,
        )

    def generate_strings(
        self, num_strings: int, seed: int = 0,
    ) -> list[list[int]]:
        rng = np.random.RandomState(seed)
        strings = []
        for _ in range(num_strings):
            n = 0
            while rng.rand() > self.p:
                n += 1
            s = ([SYMBOL_HASH]
                 + [SYMBOL_A] * n
                 + [SYMBOL_B] * n
                 + [SYMBOL_C] * n
                 + [SYMBOL_HASH])
            strings.append(s)
        return strings

    def make_fixed_n(self, n: int) -> tuple[list[int], list[int]]:
        s = ([SYMBOL_HASH]
             + [SYMBOL_A] * n
             + [SYMBOL_B] * n
             + [SYMBOL_C] * n
             + [SYMBOL_HASH])
        return s[:-1], s[1:]

    def compute_grammar_weights(
        self, max_n: int, min_n: int = 1,
    ) -> np.ndarray:
        # Same geometric PCFG as aⁿbⁿ
        ns = np.arange(min_n, max_n + 1)
        return self.p * (1 - self.p) ** ns

    def compute_target_probs(self) -> dict:
        """Optimal output probabilities per phase.

        - After #:   P(a) = 1-p, P(#) = p         (start or empty string)
        - During a:  P(a) = 1-p, P(b) = p          (continue a's or switch to b's)
        - During b (not last): P(b) = 1             (must continue b's)
        - Last b:    P(c) = 1                       (switch to c's)
        - During c (not last): P(c) = 1             (must continue c's)
        - Last c:    P(#) = 1                       (end of string)
        """
        p = self.p
        return {
            "start": np.array([p, 1 - p, 0.0, 0.0], dtype=np.float32),
            "a_phase": np.array([0.0, 1 - p, p, 0.0], dtype=np.float32),
            "b_phase": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "last_b": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "c_phase": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "last_c": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }

    def generate_negative_examples(
        self, num: int, max_n: int, seed: int = 42,
    ) -> list[list[int]]:
        rng = np.random.RandomState(seed)
        strings = []
        per_type = num // 4
        remainder = num - 4 * per_type

        def _sample_length():
            return int(rng.geometric(self.p)) if rng.random() > 0.5 else rng.randint(1, max(max_n, 2))

        def _wrap(body):
            return [SYMBOL_HASH] + body + [SYMBOL_HASH]

        # Type 1: wrong counts (a^l b^m c^n where not all equal)
        for _ in range(per_type + remainder):
            l = _sample_length()
            m = _sample_length()
            n = _sample_length()
            if l == m == n:
                n = l + 1  # break equality
            strings.append(_wrap([SYMBOL_A] * l + [SYMBOL_B] * m + [SYMBOL_C] * n))

        # Type 2: wrong order (e.g. a's and c's swapped, interleaved)
        for _ in range(per_type):
            n = _sample_length()
            # c's before b's
            strings.append(_wrap([SYMBOL_A] * n + [SYMBOL_C] * n + [SYMBOL_B] * n))

        # Type 3: missing section (only a's and b's, or only b's and c's)
        for _ in range(per_type):
            n = _sample_length()
            if rng.random() < 0.5:
                strings.append(_wrap([SYMBOL_A] * n + [SYMBOL_B] * n))
            else:
                strings.append(_wrap([SYMBOL_B] * n + [SYMBOL_C] * n))

        # Type 4: random symbols from {a, b, c}
        for _ in range(per_type):
            length = _sample_length() * 3
            s = [rng.choice([SYMBOL_A, SYMBOL_B, SYMBOL_C]) for _ in range(length)]
            # Break if accidentally valid
            na = sum(1 for c in s if c == SYMBOL_A)
            nb = sum(1 for c in s if c == SYMBOL_B)
            nc = sum(1 for c in s if c == SYMBOL_C)
            if na == nb == nc and na > 0:
                prefix_a = all(c == SYMBOL_A for c in s[:na])
                mid_b = all(c == SYMBOL_B for c in s[na:na+nb])
                suffix_c = all(c == SYMBOL_C for c in s[na+nb:])
                if prefix_a and mid_b and suffix_c:
                    s[0] = SYMBOL_C  # break validity
            strings.append(_wrap(s))

        inputs = [s[:-1] for s in strings]
        targets = [s[1:] for s in strings]
        return inputs, targets

    def _string_n(self, s: list[int]) -> int:
        return s.count(SYMBOL_A)
