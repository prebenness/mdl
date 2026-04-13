"""Tests for the golden Dyck-1 LSTM network."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.mdl.golden_dyck1 import (
    build_golden_dyck1_params,
    golden_dyck1_forward,
    golden_dyck1_mdl_score,
    HIDDEN_SIZE,
)
from src.mdl.tasks.dyck1 import SYMBOL_HASH, SYMBOL_OPEN, SYMBOL_CLOSE


class TestGoldenDyck1Forward:
    @pytest.fixture
    def params(self):
        return build_golden_dyck1_params(p=1/3)

    def _predict(self, params, input_seq):
        """Run forward pass and return softmax probabilities."""
        x = jnp.array([input_seq], dtype=jnp.int32)
        logits = golden_dyck1_forward(params, x)
        probs = jax.nn.softmax(logits[0], axis=-1)
        return np.array(probs)

    def test_after_hash_top_level(self, params):
        """After #, predict P(()=1/3, P(#)=2/3, P())=0."""
        probs = self._predict(params, [SYMBOL_HASH])
        # Position 0: after seeing #
        p = probs[0]
        assert p[SYMBOL_OPEN] == pytest.approx(1/3, abs=0.01)
        assert p[SYMBOL_HASH] == pytest.approx(2/3, abs=0.01)
        assert p[SYMBOL_CLOSE] < 0.01

    def test_inside_brackets_depth1(self, params):
        """After #(, depth=1, predict P(()=1/3, P())=2/3."""
        probs = self._predict(params, [SYMBOL_HASH, SYMBOL_OPEN])
        p = probs[1]  # prediction after (
        assert p[SYMBOL_OPEN] == pytest.approx(1/3, abs=0.01)
        assert p[SYMBOL_CLOSE] == pytest.approx(2/3, abs=0.01)
        assert p[SYMBOL_HASH] < 0.01

    def test_inside_brackets_depth2_argmax(self, params):
        """After #((, depth=2, argmax is correct even though probs drift.

        Due to tanh(depth) saturation, the softmax probabilities at
        depth>1 drift from {1/3, 2/3}. But the ranking is preserved:
        P()) > P(() > P(#) ≈ 0.
        """
        probs = self._predict(params, [SYMBOL_HASH, SYMBOL_OPEN, SYMBOL_OPEN])
        p = probs[2]
        assert p[SYMBOL_CLOSE] > p[SYMBOL_OPEN]
        assert p[SYMBOL_HASH] < 0.01

    def test_after_close_back_to_top(self, params):
        """After #(), depth=0, predict top-level P(()=1/3, P(#)=2/3."""
        probs = self._predict(params, [SYMBOL_HASH, SYMBOL_OPEN, SYMBOL_CLOSE])
        p = probs[2]  # prediction after )
        assert p[SYMBOL_OPEN] == pytest.approx(1/3, abs=0.01)
        assert p[SYMBOL_HASH] == pytest.approx(2/3, abs=0.01)
        assert p[SYMBOL_CLOSE] < 0.01

    def test_after_close_still_inside(self, params):
        """After #((), depth=1, predict inside P(()=1/3, P())=2/3."""
        probs = self._predict(params, [SYMBOL_HASH, SYMBOL_OPEN, SYMBOL_OPEN, SYMBOL_CLOSE])
        p = probs[3]  # prediction after inner )
        assert p[SYMBOL_OPEN] == pytest.approx(1/3, abs=0.02)
        assert p[SYMBOL_CLOSE] == pytest.approx(2/3, abs=0.02)
        assert p[SYMBOL_HASH] < 0.01

    def test_depth1_probabilities_exact(self, params):
        """For (^1)^1 = (), the probabilities are exact at depth 1.

        Note: argmax at stochastic positions (e.g. after #, P(#)=2/3)
        does NOT match the target — that's correct LM behavior.
        """
        # String: # ( ) #
        # Input:  [0, 1, 2]
        # Target: [1, 2, 0]
        probs = self._predict(params, [SYMBOL_HASH, SYMBOL_OPEN, SYMBOL_CLOSE])

        # t=0 (after #): top level — P(#)=2/3, P(()=1/3
        np.testing.assert_allclose(probs[0, SYMBOL_HASH], 2/3, atol=0.01)
        np.testing.assert_allclose(probs[0, SYMBOL_OPEN], 1/3, atol=0.01)

        # t=1 (after (, depth=1): inside — P(()=1/3, P())=2/3
        np.testing.assert_allclose(probs[1, SYMBOL_OPEN], 1/3, atol=0.01)
        np.testing.assert_allclose(probs[1, SYMBOL_CLOSE], 2/3, atol=0.01)

        # t=2 (after ), depth=0): top level again
        np.testing.assert_allclose(probs[2, SYMBOL_HASH], 2/3, atol=0.01)
        np.testing.assert_allclose(probs[2, SYMBOL_OPEN], 1/3, atol=0.01)

    def test_deterministic_positions_correct(self, params):
        """At deterministic positions, the golden network is perfect.

        For (^n)^n strings, the only deterministic position is the
        last ) which must be followed by #. All other positions are
        stochastic. This verifies the golden gives P(#)≈1 there.
        """
        from src.mdl.tasks.dyck1 import Dyck1Task
        task = Dyck1Task(p=1/3)

        for n in [1, 3, 5, 10]:
            inp, tgt = task.make_fixed_n(n)
            x = jnp.array([inp], dtype=jnp.int32)
            logits = golden_dyck1_forward(params, x)
            probs = jax.nn.softmax(logits[0], axis=-1)

            # Last position: after final ), predict #
            last_probs = np.array(probs[len(tgt) - 1])
            assert last_probs[SYMBOL_HASH] > 0.5, (
                f"n={n}: last position P(#)={last_probs[SYMBOL_HASH]:.3f}"
            )


class TestGoldenDyck1MDL:
    def test_mdl_score_positive(self):
        score = golden_dyck1_mdl_score(p=1/3)
        assert score["total_bits"] > 0
        assert score["arch_bits"] > 0
        assert score["weight_bits"] > 0

    def test_param_count(self):
        """3x3 LSTM has 108 parameters (same as aⁿbⁿ)."""
        score = golden_dyck1_mdl_score(p=1/3)
        I, H, O = 3, 3, 3
        expected = 4*I*H + 4*H*H + 8*H + H*O + O  # 108
        assert score["n_params"] == expected
