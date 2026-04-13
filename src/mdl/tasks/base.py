"""Abstract base for formal language tasks."""

from __future__ import annotations

import abc
import dataclasses

import numpy as np


@dataclasses.dataclass
class TaskSpec(abc.ABC):
    """Specification for a formal language task.

    Subclasses provide data generation, test sets, and grammar weights
    for a specific formal language (e.g. aⁿbⁿ, aⁿbⁿcⁿ, Dyck-1).
    """

    name: str
    alphabet: list[str]
    num_symbols: int
    p: float = 0.3

    # --- Data generation ---

    @abc.abstractmethod
    def generate_strings(
        self, num_strings: int, seed: int = 0,
    ) -> list[list[int]]:
        """Sample strings from the task's PCFG."""

    def make_dataset(
        self,
        num_strings: int = 1000,
        seed: int = 0,
        max_n_train: int | None = None,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Generate training (input, target) pairs for language modeling.

        Input is s[:-1], target is s[1:].
        """
        strings = self.generate_strings(num_strings, seed=seed)
        inputs, targets = [], []
        for s in strings:
            if max_n_train is not None and self._string_n(s) > max_n_train:
                continue
            inputs.append(s[:-1])
            targets.append(s[1:])
        return inputs, targets

    # --- Test / validation sets ---

    @abc.abstractmethod
    def make_fixed_n(self, n: int) -> tuple[list[int], list[int]]:
        """Canonical test string for parameter n → (input, target)."""

    def make_test_set(
        self, max_n: int = 1500,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """All canonical strings for n=1..max_n."""
        inputs, targets = [], []
        for n in range(1, max_n + 1):
            inp, tgt = self.make_fixed_n(n)
            inputs.append(inp)
            targets.append(tgt)
        return inputs, targets

    def make_validation_set(
        self,
        train_max_n: int,
        val_max_n: int = 71,
        val_min_n: int | None = None,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Validation set: canonical strings in a held-out n-range."""
        inputs, targets = [], []
        start_n = train_max_n + 1
        if val_min_n is not None:
            start_n = max(start_n, int(val_min_n))
        for n in range(start_n, val_max_n + 1):
            inp, tgt = self.make_fixed_n(n)
            inputs.append(inp)
            targets.append(tgt)
        return inputs, targets

    # --- Grammar weights ---

    @abc.abstractmethod
    def compute_grammar_weights(
        self, max_n: int, min_n: int = 1,
    ) -> np.ndarray:
        """PCFG weights for the canonical test set n=min_n..max_n."""

    # --- Target probabilities ---

    @abc.abstractmethod
    def compute_target_probs(self) -> dict:
        """Theoretically optimal output probabilities per phase."""

    # --- Negative examples (for recognition accuracy) ---

    @abc.abstractmethod
    def generate_negative_examples(
        self, num: int, max_n: int, seed: int = 42,
    ) -> list[list[int]]:
        """Generate strings NOT in the language for recognition tests."""

    # --- Helpers ---

    def _string_n(self, s: list[int]) -> int:
        """Extract the canonical 'n' parameter from a full string.

        Default: count occurrences of symbol index 1 (first non-delimiter).
        Override for tasks where this doesn't apply.
        """
        return s.count(1)
