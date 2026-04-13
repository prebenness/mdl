"""Tests for the task abstraction layer (src/mdl/tasks/)."""

import numpy as np
import pytest

from src.mdl.tasks import get_task, AnbnTask
from src.mdl.tasks.base import TaskSpec
from src.mdl import data as legacy_data
from src.mdl.evaluation import compute_anbn_grammar_weights


class TestTaskRegistry:
    def test_get_anbn(self):
        task = get_task("anbn")
        assert isinstance(task, AnbnTask)
        assert task.name == "anbn"

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task("nonexistent")

    def test_custom_p(self):
        task = get_task("anbn", p=0.5)
        assert task.p == 0.5


class TestAnbnTask:
    """Verify AnbnTask matches legacy data.py exactly."""

    @pytest.fixture
    def task(self):
        return AnbnTask(p=0.3)

    def test_alphabet(self, task):
        assert task.alphabet == ["#", "a", "b"]
        assert task.num_symbols == 3

    def test_generate_strings_matches_legacy(self, task):
        new = task.generate_strings(100, seed=42)
        old = legacy_data.generate_anbn_strings(100, p=0.3, seed=42)
        assert len(new) == len(old)
        for s_new, s_old in zip(new, old):
            assert s_new == s_old, f"Mismatch: {s_new} vs {s_old}"

    def test_make_dataset_matches_legacy(self, task):
        new_inp, new_tgt = task.make_dataset(100, seed=42)
        old_inp, old_tgt = legacy_data.make_anbn_dataset(100, p=0.3, seed=42)
        assert len(new_inp) == len(old_inp)
        for ni, oi in zip(new_inp, old_inp):
            assert ni == oi
        for nt, ot in zip(new_tgt, old_tgt):
            assert nt == ot

    def test_make_fixed_n(self, task):
        inp, tgt = task.make_fixed_n(5)
        old_inp, old_tgt = legacy_data.make_anbn_fixed_n(5)
        assert inp == old_inp
        assert tgt == old_tgt

    def test_make_test_set_matches_legacy(self, task):
        new_inp, new_tgt = task.make_test_set(max_n=10)
        old_inp, old_tgt = legacy_data.make_test_set(max_n=10)
        assert len(new_inp) == len(old_inp) == 10
        for ni, oi in zip(new_inp, old_inp):
            assert ni == oi

    def test_make_validation_set_matches_legacy(self, task):
        new_inp, new_tgt = task.make_validation_set(
            train_max_n=10, val_max_n=20, val_min_n=15,
        )
        old_inp, old_tgt = legacy_data.make_validation_set(
            train_max_n=10, val_max_n=20, val_min_n=15,
        )
        assert len(new_inp) == len(old_inp)
        for ni, oi in zip(new_inp, old_inp):
            assert ni == oi

    def test_grammar_weights_match_legacy(self, task):
        new_w = task.compute_grammar_weights(max_n=50, min_n=1)
        old_w = compute_anbn_grammar_weights(max_n=50, p=0.3, min_n=1)
        np.testing.assert_allclose(new_w, old_w)

    def test_target_probs_match_legacy(self, task):
        new_tp = task.compute_target_probs()
        old_tp = legacy_data.compute_target_probs(p=0.3)
        for key in old_tp:
            np.testing.assert_allclose(new_tp[key], old_tp[key])

    def test_string_n_counts_as(self, task):
        s = [0, 1, 1, 1, 2, 2, 2, 0]  # #aaabbb#
        assert task._string_n(s) == 3

    def test_make_dataset_max_n_filter(self, task):
        inp, tgt = task.make_dataset(500, seed=0, max_n_train=3)
        for i in inp:
            # input is s[:-1], count a's
            assert i.count(1) <= 3

    def test_negative_examples_all_invalid(self, task):
        neg_inp, neg_tgt = task.generate_negative_examples(100, max_n=20)
        assert len(neg_inp) == 100
        for inp, tgt in zip(neg_inp, neg_tgt):
            full = inp + [tgt[-1]]  # reconstruct full string
            # A valid anbn has form [#, a^n, b^n, #]
            body = full[1:-1]
            na = sum(1 for c in body if c == 1)
            nb = sum(1 for c in body if c == 2)
            if na == nb and na > 0:
                # Check if it's actually a^n b^n (all a's then all b's)
                is_valid = all(c == 1 for c in body[:na]) and \
                           all(c == 2 for c in body[na:])
                assert not is_valid, f"Negative example is valid: {full}"


class TestAnbncnTask:
    """Tests for aⁿbⁿcⁿ task."""

    @pytest.fixture
    def task(self):
        return get_task("anbncn")

    def test_alphabet(self, task):
        assert task.alphabet == ["#", "a", "b", "c"]
        assert task.num_symbols == 4

    def test_string_structure(self, task):
        strings = task.generate_strings(200, seed=0)
        for s in strings:
            assert s[0] == 0 and s[-1] == 0, "Must start/end with #"
            body = s[1:-1]
            if len(body) == 0:
                continue  # empty string (n=0) is valid
            # Count symbols
            na = sum(1 for c in body if c == 1)
            nb = sum(1 for c in body if c == 2)
            nc = sum(1 for c in body if c == 3)
            assert na == nb == nc, f"Counts unequal: a={na}, b={nb}, c={nc}"
            # Check order: all a's, then all b's, then all c's
            assert body == [1]*na + [2]*nb + [3]*nc

    def test_make_fixed_n(self, task):
        inp, tgt = task.make_fixed_n(4)
        full = inp + [tgt[-1]]
        assert full == [0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 0]

    def test_test_set_size(self, task):
        inp, tgt = task.make_test_set(max_n=10)
        assert len(inp) == 10

    def test_grammar_weights_geometric(self, task):
        w = task.compute_grammar_weights(max_n=5, min_n=1)
        assert len(w) == 5
        # Check geometric: w[i] / w[i+1] = (1-p) for all i
        for i in range(len(w) - 1):
            np.testing.assert_allclose(w[i+1] / w[i], 1 - task.p, rtol=1e-10)

    def test_target_probs_keys(self, task):
        tp = task.compute_target_probs()
        assert "start" in tp
        assert "a_phase" in tp
        assert "last_b" in tp
        assert "last_c" in tp
        for v in tp.values():
            assert len(v) == 4  # 4 symbols
            np.testing.assert_allclose(v.sum(), 1.0)

    def test_negative_examples_all_invalid(self, task):
        neg_inp, neg_tgt = task.generate_negative_examples(100, max_n=10)
        assert len(neg_inp) == 100
        for inp, tgt in zip(neg_inp, neg_tgt):
            full = inp + [tgt[-1]]
            body = full[1:-1]
            na = sum(1 for c in body if c == 1)
            nb = sum(1 for c in body if c == 2)
            nc = sum(1 for c in body if c == 3)
            if na == nb == nc and na > 0:
                is_valid = (body == [1]*na + [2]*nb + [3]*nc)
                assert not is_valid, f"Negative example is valid: {full}"

    def test_make_dataset_pairs(self, task):
        inp, tgt = task.make_dataset(50, seed=0)
        for i, t in zip(inp, tgt):
            assert len(i) == len(t)
            # target is input shifted by 1
            assert i[0] == 0  # starts with #


class TestDyck1Task:
    """Tests for Dyck-1 task."""

    @pytest.fixture
    def task(self):
        return get_task("dyck1")

    def test_alphabet(self, task):
        assert task.alphabet == ["#", "(", ")"]
        assert task.num_symbols == 3

    def test_generated_strings_balanced(self, task):
        strings = task.generate_strings(200, seed=0)
        for s in strings:
            assert s[0] == 0 and s[-1] == 0, "Must start/end with #"
            body = s[1:-1]
            # Check balance
            depth = 0
            for c in body:
                if c == 1:
                    depth += 1
                elif c == 2:
                    depth -= 1
                assert depth >= 0, f"Negative depth in {s}"
            assert depth == 0, f"Unbalanced: {s}"

    def test_generated_strings_length_limit(self, task):
        strings = task.generate_strings(100, seed=0, max_length=20)
        for s in strings:
            body = s[1:-1]
            assert len(body) <= 20

    def test_make_fixed_n(self, task):
        inp, tgt = task.make_fixed_n(3)
        full = inp + [tgt[-1]]
        assert full == [0, 1, 1, 1, 2, 2, 2, 0]  # #((()))#

    def test_exhaustive_test_set_count(self, task):
        """Dyck-1 strings of length ≤ 10.

        Length 0: ε (1 string = C(0))
        Length 2: () (1 string = C(1))
        Length 4: (()), ()() (2 strings = C(2))
        Length 6: ((())), (())(), ()(()), ()()(),(()()) (5 = C(3))
        Length 8: C(4) = 14
        Length 10: C(5) = 42
        Total: 1 + 1 + 2 + 5 + 14 + 42 = 65
        """
        inp, tgt = task.make_test_set(max_length=10)
        assert len(inp) == 65

    def test_exhaustive_test_set_all_valid(self, task):
        inp, tgt = task.make_test_set(max_length=10)
        for i, t in zip(inp, tgt):
            full = i + [t[-1]]
            assert full[0] == 0 and full[-1] == 0
            body = full[1:-1]
            depth = 0
            for c in body:
                if c == 1:
                    depth += 1
                elif c == 2:
                    depth -= 1
                assert depth >= 0
            assert depth == 0

    def test_grammar_weights_sum_reasonable(self, task):
        w = task.compute_grammar_weights(max_length=10)
        # Weights should sum to < 1 (the grammar can generate strings > length 10)
        assert 0 < w.sum() < 1.0
        # Should have 65 weights (one per test string)
        assert len(w) == 65

    def test_grammar_weights_empty_string(self, task):
        """The empty string has probability (1-p) = 2/3."""
        w = task.compute_grammar_weights(max_length=10)
        # First string in enumeration is the empty string (length 0)
        np.testing.assert_allclose(w[0], 2/3, rtol=1e-10)

    def test_grammar_weights_single_pair(self, task):
        """() has probability p * (1-p)^2 = (1/3) * (2/3)^2."""
        w = task.compute_grammar_weights(max_length=10)
        # Second string should be () (length 2)
        expected = (1/3) * (2/3) * (2/3)  # p * P(S1→ε) * P(S2→ε)
        np.testing.assert_allclose(w[1], expected, rtol=1e-10)

    def test_negative_examples_all_invalid(self, task):
        neg_inp, neg_tgt = task.generate_negative_examples(60, max_n=5)
        assert len(neg_inp) == 60
        for inp, tgt in zip(neg_inp, neg_tgt):
            full = inp + [tgt[-1]]
            body = full[1:-1]
            depth = 0
            valid = True
            for c in body:
                if c == 1:
                    depth += 1
                elif c == 2:
                    depth -= 1
                if depth < 0:
                    valid = False
                    break
            if valid:
                valid = (depth == 0)
            assert not valid, f"Negative example is valid Dyck-1: {full}"

    def test_target_probs(self, task):
        tp = task.compute_target_probs()
        assert "top_level" in tp
        assert "inside" in tp
        for v in tp.values():
            assert len(v) == 3
            np.testing.assert_allclose(v.sum(), 1.0)


class TestTaskAgnosticEval:
    """Test that the task-agnostic evaluation matches legacy."""

    def test_grammar_weighted_nll_matches(self):
        """compute_grammar_weighted_nll_bits_task with AnbnTask == legacy."""
        from src.mdl.evaluation import (
            compute_grammar_weighted_nll_bits,
            compute_grammar_weighted_nll_bits_task,
        )
        from src.mdl.golden import build_golden_network_params, golden_forward

        params = build_golden_network_params(p=0.3)
        def fwd(x):
            return golden_forward(params, x)

        task = get_task("anbn", p=0.3)

        old = compute_grammar_weighted_nll_bits(fwd, max_n=50, p=0.3)
        new = compute_grammar_weighted_nll_bits_task(fwd, task, max_n=50)

        np.testing.assert_allclose(
            old["data_dh_bits"], new["data_dh_bits"], rtol=1e-10,
        )


# We need compute_anbn_grammar_weights accessible from legacy path
class TestLegacyBackcompat:
    """Ensure data.py still works for all existing imports."""

    def test_symbols_still_exported(self):
        assert legacy_data.SYMBOL_HASH == 0
        assert legacy_data.SYMBOL_A == 1
        assert legacy_data.SYMBOL_B == 2
        assert legacy_data.NUM_SYMBOLS == 3

    def test_generate_still_works(self):
        strings = legacy_data.generate_anbn_strings(10, seed=0)
        assert len(strings) == 10

    def test_make_dataset_still_works(self):
        inp, tgt = legacy_data.make_anbn_dataset(10, seed=0)
        assert len(inp) == 10

    def test_make_fixed_n_still_works(self):
        inp, tgt = legacy_data.make_anbn_fixed_n(5)
        assert len(inp) == 11  # 2*5 + 1

    def test_make_test_set_still_works(self):
        inp, tgt = legacy_data.make_test_set(max_n=5)
        assert len(inp) == 5

    def test_sequences_to_padded_arrays_still_works(self):
        inp, tgt = legacy_data.make_anbn_dataset(5, seed=0)
        x, y, mask = legacy_data.sequences_to_padded_arrays(inp, tgt)
        assert x.shape[0] == 5
