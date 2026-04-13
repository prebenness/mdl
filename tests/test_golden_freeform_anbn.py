"""Tests for the golden free-form aⁿbⁿ network."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fractions import Fraction

from src.mdl.golden_freeform_anbn import (
    TOPOLOGY,
    WEIGHTS_RATIONAL,
    BIASES_RATIONAL,
    build_golden_freeform_anbn_params,
    golden_freeform_anbn_forward,
    golden_freeform_anbn_mdl_score,
)
from src.mdl.freeform_rnn import freeform_forward, zero_neg_normalize


class TestGoldenFreeformAnbnForward:
    @pytest.fixture
    def params(self):
        return build_golden_freeform_anbn_params()

    def _probs(self, params, input_seq):
        """Run forward and return zero-neg-normalized probabilities."""
        x = jnp.array([input_seq], dtype=jnp.int32)
        log_probs = golden_freeform_anbn_forward(params, x)
        return np.exp(np.array(log_probs[0]))

    def test_a_phase_probs(self, params):
        """During a-phase: P(a)=0.7, P(b)=0.3, P(#)≈0."""
        probs = self._probs(params, [0, 1, 1])  # # a a
        # Position 1 (after first a): a-phase
        np.testing.assert_allclose(probs[1, 1], 0.7, atol=0.01)  # P(a)
        np.testing.assert_allclose(probs[1, 2], 0.3, atol=0.01)  # P(b)
        assert probs[1, 0] < 0.001  # P(#)

    def test_b_phase_probs(self, params):
        """During b-phase (counter>0): P(b)=1, P(a)=0, P(#)≈0."""
        probs = self._probs(params, [0, 1, 1, 2])  # # a a b
        # Position 3 (after first b, counter still > 0): b-phase
        np.testing.assert_allclose(probs[3, 2], 1.0, atol=0.01)  # P(b)
        assert probs[3, 1] < 0.001  # P(a)
        assert probs[3, 0] < 0.001  # P(#)

    def test_after_last_b_predicts_hash(self, params):
        """After final b (counter=0): P(#)≈1."""
        probs = self._probs(params, [0, 1, 2])  # # a b (a^1 b^1)
        # Position 2 (after last b): counter = 0
        assert probs[2, 0] > 0.99  # P(#)

    def test_initial_position(self, params):
        """After #: P(a)=1 (counter=0, step(0)=0, only P(a) is positive)."""
        probs = self._probs(params, [0])
        np.testing.assert_allclose(probs[0, 1], 1.0, atol=0.01)  # P(a)

    def test_counter_mechanism(self, params):
        """Counter correctly tracks a's and decrements for b's."""
        w = jnp.array(build_golden_freeform_anbn_params())
        # Feed a^3 b^3 and check raw unit activations via logits
        x = jnp.array([[0, 1, 1, 1, 2, 2, 2]], dtype=jnp.int32)
        logits = freeform_forward(TOPOLOGY, w, x)
        # P(b) output (unit 6, step function) should be:
        # t=0(#): step(0)=0, t=1(a): step(1)=1, t=2(a): step(2)=1
        # t=3(a): step(3)=1, t=4(b): step(2)=1, t=5(b): step(1)=1
        # t=6(b): step(0)=0
        step_vals = np.array(logits[0, :, 2])  # output idx 2 = P(b)
        np.testing.assert_allclose(step_vals, [0, 1, 1, 1, 1, 1, 0], atol=1e-5)

    def test_deterministic_positions_n1_to_50(self, params):
        """Argmax correct at all deterministic positions for n=1..50.

        Stochastic: a-phase transition (target=b, but P(a)=0.7 > P(b)=0.3).
        Deterministic: b-phase (P(b)=1), final position (P(#)≈1).
        """
        for n in range(1, 51):
            seq = [0] + [1]*n + [2]*n + [0]  # # a...a b...b #
            x = jnp.array([seq[:-1]], dtype=jnp.int32)
            log_probs = golden_freeform_anbn_forward(params, x)
            preds = np.argmax(np.array(log_probs[0]), axis=-1)
            targets = seq[1:]

            # Check deterministic positions only:
            # b-phase: positions n+1..2n-1 (input=b, target=b, counter>0)
            for t in range(n + 1, 2 * n):
                assert preds[t] == targets[t], (
                    f"n={n}, t={t}: pred={preds[t]}, target={targets[t]}"
                )
            # Final position: input=last_b, target=#
            t_final = 2 * n
            assert preds[t_final] == 0, (
                f"n={n}: final pred={preds[t_final]}, expected #=0"
            )


class TestGoldenFreeformAnbnMDL:
    def test_h_bits(self):
        """|H| = 137 bits (paper reports 139; 2-bit convention difference)."""
        score = golden_freeform_anbn_mdl_score()
        assert score["total_bits"] == 137

    def test_param_count(self):
        """5 non-input connections + 2 biases = 7 encoded parameters."""
        score = golden_freeform_anbn_mdl_score()
        assert score["n_params"] == TOPOLOGY.n_weights

    def test_test_dh_matches_paper(self):
        """Grammar-weighted test |D:H| ≈ 2.94 bits (normalized weights).

        Reference: Abudy et al. (2025, arXiv:2505.13398v2), Table 1.
        Our code uses unnormalized PCFG weights (sum≈0.7); dividing by
        that sum recovers the paper's normalized |D:H| = 2.94.
        """
        from src.mdl.tasks import get_task
        from src.mdl.evaluation import compute_grammar_weighted_nll_bits_task

        params = build_golden_freeform_anbn_params()
        task = get_task("anbn", p=0.3)

        def fwd(x):
            return golden_freeform_anbn_forward(params, x)

        result = compute_grammar_weighted_nll_bits_task(fwd, task, max_n=100)
        unnorm_dh = result["data_dh_bits"]

        # Normalize: sum of PCFG weights = p * sum(1-p)^n ≈ p*(1-p)/(1-(1-p)) = 1-p
        normalized_dh = unnorm_dh / 0.7
        assert normalized_dh == pytest.approx(2.94, abs=0.05)
