"""aⁿbⁿ task: the language {#aⁿbⁿ# | n ≥ 0}.

PCFG: S → aSb (prob 1-p) | ε (prob p), with p = 0.3.
Alphabet: {#, a, b} encoded as 0, 1, 2.

Reference: Lan et al. (2024, "MDL Regularization of LSTM Language
Models for Formal Language Learning", ACL 2024).
"""

from __future__ import annotations

import numpy as np

from .base import TaskSpec

SYMBOL_HASH = 0
SYMBOL_A = 1
SYMBOL_B = 2


class AnbnTask(TaskSpec):
    """aⁿbⁿ formal language task."""

    def __init__(self, p: float = 0.3):
        super().__init__(
            name="anbn",
            alphabet=["#", "a", "b"],
            num_symbols=3,
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
            s = [SYMBOL_HASH] + [SYMBOL_A] * n + [SYMBOL_B] * n + [SYMBOL_HASH]
            strings.append(s)
        return strings

    def make_fixed_n(self, n: int) -> tuple[list[int], list[int]]:
        s = [SYMBOL_HASH] + [SYMBOL_A] * n + [SYMBOL_B] * n + [SYMBOL_HASH]
        return s[:-1], s[1:]

    def compute_grammar_weights(
        self, max_n: int, min_n: int = 1,
    ) -> np.ndarray:
        ns = np.arange(min_n, max_n + 1)
        return self.p * (1 - self.p) ** ns

    def compute_target_probs(self) -> dict:
        p = self.p
        return {
            "start": np.array([p, 1 - p, 0.0], dtype=np.float32),
            "a_phase": np.array([0.0, 1 - p, p], dtype=np.float32),
            "b_phase": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "last_b": np.array([1.0, 0.0, 0.0], dtype=np.float32),
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

        # Type 1: wrong counts a^m b^n, m != n
        for _ in range(per_type + remainder):
            m = _sample_length()
            n = _sample_length()
            if m == n:
                n = m + 1
            strings.append(_wrap([SYMBOL_A] * m + [SYMBOL_B] * n))

        # Type 2: interleaved
        for _ in range(per_type):
            length = _sample_length() * 2
            s = [SYMBOL_A if rng.random() < 0.5 else SYMBOL_B for _ in range(length)]
            if len(s) >= 2 and all(c == SYMBOL_A for c in s[:len(s)//2]) \
                    and all(c == SYMBOL_B for c in s[len(s)//2:]) \
                    and s.count(SYMBOL_A) == s.count(SYMBOL_B):
                s[0] = SYMBOL_B
            strings.append(_wrap(s))

        # Type 3: single symbol
        for _ in range(per_type):
            length = _sample_length()
            sym = SYMBOL_A if rng.random() < 0.5 else SYMBOL_B
            strings.append(_wrap([sym] * length))

        # Type 4: random
        for _ in range(per_type):
            length = _sample_length()
            s = [rng.choice([SYMBOL_A, SYMBOL_B]) for _ in range(length)]
            na = s.count(SYMBOL_A)
            nb = s.count(SYMBOL_B)
            if na == nb and na > 0:
                is_valid = all(c == SYMBOL_A for c in s[:na]) and \
                           all(c == SYMBOL_B for c in s[na:])
                if is_valid:
                    s[0] = SYMBOL_B
            strings.append(_wrap(s))

        # Return as (inputs, targets) pairs
        inputs = [s[:-1] for s in strings]
        targets = [s[1:] for s in strings]
        return inputs, targets

    def _string_n(self, s: list[int]) -> int:
        return s.count(SYMBOL_A)
