"""Tests for QAT (quantization-aware training) and integer-attraction
utilities added to prime_rationals.py.

Run with:
    JAX_PLATFORMS=cpu pytest tests/test_prime_rationals.py -v
"""
import math
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prime_rationals import (
    round_ste,
    integer_attraction_penalty,
    integer_distance_penalty,
    reconstruct_weight,
    get_log_primes,
    get_forward_mode,
    get_integer_mu,
    compute_qat_diagnostics,
    clamp_exponents_in_params,
    PrimeExpLinear,
    PrimeExpMLP,
    PrimeExpLSTM,
    PrimeRationalConfig,
)


# ===================================================================
# round_ste
# ===================================================================

class TestRoundSTE:
    """Tests for the straight-through estimator rounding function."""

    def test_forward_is_round(self):
        z = jnp.array([0.3, 1.7, -0.6, 2.5, -1.2])
        result = round_ste(z)
        expected = jnp.round(z)
        np.testing.assert_allclose(result, expected, atol=1e-7)

    def test_forward_integers_unchanged(self):
        z = jnp.array([0.0, 1.0, -2.0, 3.0])
        result = round_ste(z)
        np.testing.assert_allclose(result, z, atol=1e-7)

    def test_gradient_is_identity(self):
        """Gradient of round_ste should be 1.0 everywhere (STE)."""
        z = jnp.array([0.3, 1.7, -0.6, 2.5])
        grad_fn = jax.grad(lambda x: jnp.sum(round_ste(x)))
        grads = grad_fn(z)
        np.testing.assert_allclose(grads, jnp.ones_like(z), atol=1e-7)

    def test_gradient_at_integers(self):
        z = jnp.array([0.0, 1.0, -2.0])
        grad_fn = jax.grad(lambda x: jnp.sum(round_ste(x)))
        grads = grad_fn(z)
        np.testing.assert_allclose(grads, jnp.ones_like(z), atol=1e-7)

    def test_gradient_at_half_integers(self):
        z = jnp.array([0.5, 1.5, -0.5])
        grad_fn = jax.grad(lambda x: jnp.sum(round_ste(x)))
        grads = grad_fn(z)
        np.testing.assert_allclose(grads, jnp.ones_like(z), atol=1e-7)


# ===================================================================
# integer_attraction_penalty
# ===================================================================

class TestIntegerAttractionPenalty:
    """Tests for the sin^2(pi*z) integer-attraction penalty."""

    def test_zero_at_integers(self):
        z = jnp.array([0.0, 1.0, -2.0, 3.0, -5.0])
        penalty = integer_attraction_penalty(z)
        assert float(penalty) < 1e-12

    def test_max_at_half_integers(self):
        z = jnp.array([0.5, 1.5, -0.5, 2.5])
        penalty = integer_attraction_penalty(z)
        np.testing.assert_allclose(float(penalty), 1.0, atol=1e-6)

    def test_intermediate_values(self):
        z = jnp.array([0.25])
        penalty = integer_attraction_penalty(z)
        expected = math.sin(math.pi * 0.25) ** 2
        np.testing.assert_allclose(float(penalty), expected, atol=1e-6)

    def test_gradient_nonzero_away_from_integer(self):
        z = jnp.array(0.3)
        grad_fn = jax.grad(lambda x: integer_attraction_penalty(x[None]))
        g = grad_fn(z)
        assert abs(float(g)) > 0.1

    def test_gradient_zero_at_integer(self):
        z = jnp.array(1.0)
        grad_fn = jax.grad(lambda x: integer_attraction_penalty(x[None]))
        g = grad_fn(z)
        np.testing.assert_allclose(float(g), 0.0, atol=1e-5)


# ===================================================================
# integer_distance_penalty
# ===================================================================

class TestIntegerDistancePenalty:
    """Tests for the (z - round(z))^2 distance penalty."""

    def test_zero_at_integers(self):
        z = jnp.array([0.0, 1.0, -2.0, 3.0])
        penalty = integer_distance_penalty(z)
        assert float(penalty) < 1e-12

    def test_max_at_half_integers(self):
        z = jnp.array([0.5, 1.5, -0.5])
        penalty = integer_distance_penalty(z)
        np.testing.assert_allclose(float(penalty), 0.25, atol=1e-6)


# ===================================================================
# reconstruct_weight with modes
# ===================================================================

class TestReconstructWeightModes:
    """Tests for reconstruct_weight with continuous/rounded/frozen_rounded."""

    def setup_method(self):
        self.P = 3
        self.log_primes = get_log_primes(self.P)
        self.z = jnp.array([[0.3, -0.7, 1.2], [2.0, 0.0, -1.0]])
        self.u = jnp.array([0.5, -0.5])

    def test_continuous_default(self):
        w1 = reconstruct_weight(self.z, self.u, self.log_primes)
        w2 = reconstruct_weight(self.z, self.u, self.log_primes,
                                mode="continuous")
        np.testing.assert_allclose(w1, w2, atol=1e-7)

    def test_rounded_forward(self):
        w = reconstruct_weight(self.z, self.u, self.log_primes, mode="rounded")
        z_rounded = jnp.round(self.z)
        w_expected = reconstruct_weight(z_rounded, self.u, self.log_primes,
                                        mode="continuous")
        np.testing.assert_allclose(w, w_expected, atol=1e-6)

    def test_rounded_has_gradient(self):
        """STE mode should allow gradients through z."""
        def f(z):
            w = reconstruct_weight(z, self.u, self.log_primes, mode="rounded")
            return jnp.sum(w)
        g = jax.grad(f)(self.z)
        assert jnp.any(g != 0), "STE rounding should pass gradients through"

    def test_frozen_rounded_no_gradient(self):
        """Frozen rounded should block gradients through z."""
        def f(z):
            w = reconstruct_weight(z, self.u, self.log_primes,
                                   mode="frozen_rounded")
            return jnp.sum(w)
        g = jax.grad(f)(self.z)
        np.testing.assert_allclose(g, jnp.zeros_like(self.z), atol=1e-7)

    def test_frozen_rounded_forward_equals_rounded(self):
        w1 = reconstruct_weight(self.z, self.u, self.log_primes,
                                mode="rounded")
        w2 = reconstruct_weight(self.z, self.u, self.log_primes,
                                mode="frozen_rounded")
        np.testing.assert_allclose(w1, w2, atol=1e-6)


# ===================================================================
# Forward mode schedule
# ===================================================================

class TestGetForwardMode:
    """Tests for the QAT forward mode schedule."""

    def test_start_is_continuous(self):
        assert get_forward_mode(0.0) == "continuous"
        assert get_forward_mode(0.1) == "continuous"

    def test_mid_is_rounded(self):
        assert get_forward_mode(0.5) == "rounded"
        assert get_forward_mode(0.2) == "rounded"

    def test_end_is_frozen(self):
        assert get_forward_mode(0.95) == "frozen_rounded"
        assert get_forward_mode(1.0) == "frozen_rounded"

    def test_custom_thresholds(self):
        assert get_forward_mode(0.0, round_warmup_frac=0.0) == "rounded"
        assert get_forward_mode(0.5, round_warmup_frac=0.6) == "continuous"
        assert get_forward_mode(0.99, freeze_frac=0.99) == "frozen_rounded"

    def test_boundary_values(self):
        """At exact boundary, should transition to next mode."""
        assert get_forward_mode(0.2, round_warmup_frac=0.2) == "rounded"
        assert get_forward_mode(0.95, freeze_frac=0.95) == "frozen_rounded"

    def test_no_rounding_if_warmup_is_1(self):
        """If round_warmup_frac >= freeze_frac, stay continuous then frozen."""
        assert get_forward_mode(0.5, round_warmup_frac=1.0, freeze_frac=0.95) == "continuous"


# ===================================================================
# Integer mu schedule
# ===================================================================

class TestGetIntegerMu:
    """Tests for the integer-attraction penalty weight schedule."""

    def test_zero_before_start(self):
        assert get_integer_mu(0.0, mu_max=1.0) == 0.0
        assert get_integer_mu(0.05, mu_max=1.0) == 0.0

    def test_max_after_full(self):
        assert get_integer_mu(0.5, mu_max=1.0) == 1.0
        assert get_integer_mu(0.8, mu_max=1.0) == 1.0
        assert get_integer_mu(1.0, mu_max=1.0) == 1.0

    def test_linear_ramp(self):
        """mu should be linearly interpolated between start and full."""
        mu = get_integer_mu(0.3, mu_max=1.0, mu_start_frac=0.1, mu_full_frac=0.5)
        expected = 1.0 * (0.3 - 0.1) / (0.5 - 0.1)
        np.testing.assert_allclose(mu, expected, atol=1e-7)

    def test_midpoint(self):
        mu = get_integer_mu(0.3, mu_max=2.0, mu_start_frac=0.1, mu_full_frac=0.5)
        expected = 2.0 * (0.3 - 0.1) / (0.5 - 0.1)
        np.testing.assert_allclose(mu, expected, atol=1e-7)

    def test_zero_mu_max(self):
        assert get_integer_mu(0.5, mu_max=0.0) == 0.0


# ===================================================================
# Exponent clamping
# ===================================================================

class TestClampExponents:
    """Tests for clamp_exponents_in_params."""

    def test_basic_clamping(self):
        params = {
            "z_exponents": jnp.array([[10.0, -8.0, 3.0], [0.5, -0.5, 7.0]]),
            "u_signs": jnp.array([1.0, -1.0]),
        }
        clamped = clamp_exponents_in_params(params, E_max=6.0)
        z = clamped["z_exponents"]
        assert float(jnp.max(z)) <= 6.0
        assert float(jnp.min(z)) >= -6.0

    def test_u_unchanged(self):
        params = {
            "z_exponents": jnp.array([[10.0, -8.0]]),
            "u_signs": jnp.array([99.0]),
        }
        clamped = clamp_exponents_in_params(params, E_max=6.0)
        np.testing.assert_allclose(clamped["u_signs"], params["u_signs"])

    def test_within_range_unchanged(self):
        params = {
            "z_exponents": jnp.array([[1.0, -2.0, 3.0]]),
            "u_signs": jnp.array([0.5]),
        }
        clamped = clamp_exponents_in_params(params, E_max=6.0)
        np.testing.assert_allclose(clamped["z_exponents"],
                                   params["z_exponents"], atol=1e-7)

    def test_nested_params(self):
        params = {
            "layer0": {
                "z_weight": jnp.array([[[10.0, -10.0]]]),
                "u_weight": jnp.array([[1.0]]),
            },
        }
        clamped = clamp_exponents_in_params(params, E_max=5.0)
        z = clamped["layer0"]["z_weight"]
        assert float(jnp.max(z)) <= 5.0
        assert float(jnp.min(z)) >= -5.0


# ===================================================================
# QAT diagnostics
# ===================================================================

class TestQATDiagnostics:
    """Tests for compute_qat_diagnostics."""

    def test_exact_integers(self):
        params = {
            "z_exponents": jnp.array([[1.0, 0.0, -2.0], [3.0, 1.0, 0.0]]),
        }
        diag = compute_qat_diagnostics(params, P=3)
        np.testing.assert_allclose(diag["d_int"], 0.0, atol=1e-7)
        np.testing.assert_allclose(diag["d_logw"], 0.0, atol=1e-7)
        np.testing.assert_allclose(diag["f_eps_01"], 1.0, atol=1e-7)
        np.testing.assert_allclose(diag["f_eps_005"], 1.0, atol=1e-7)
        np.testing.assert_allclose(diag["int_penalty"], 0.0, atol=1e-7)

    def test_half_integers(self):
        params = {
            "z_exponents": jnp.array([[0.5, 0.5, 0.5]]),
        }
        diag = compute_qat_diagnostics(params, P=3)
        np.testing.assert_allclose(diag["d_int"], 0.5, atol=1e-6)
        np.testing.assert_allclose(diag["int_penalty"], 1.0, atol=1e-6)
        assert diag["f_eps_01"] == 0.0
        assert diag["f_eps_005"] == 0.0

    def test_no_z_params(self):
        params = {"weights": jnp.array([1.0, 2.0])}
        diag = compute_qat_diagnostics(params, P=3)
        assert diag["d_int"] == 0.0
        assert diag["f_eps_01"] == 1.0

    def test_nested_params(self):
        params = {
            "layer0": {
                "z_weight": jnp.array([[[0.0, 0.0, 0.0]]]),
            },
            "layer1": {
                "z_bias": jnp.array([[0.5, 0.5, 0.5]]),
            },
        }
        diag = compute_qat_diagnostics(params, P=3)
        # 3 zeros + 3 half-integers -> d_int = mean of [0,0,0,0.5,0.5,0.5] = 0.25
        np.testing.assert_allclose(diag["d_int"], 0.25, atol=1e-6)


# ===================================================================
# Module mode parameter
# ===================================================================

class TestModuleModePassing:
    """Test that Flax modules accept and pass mode parameter."""

    def test_prime_exp_linear_accepts_mode(self):
        model = PrimeExpLinear(features=4, P=3)
        x = jnp.ones((2, 3))
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, x)["params"]
        # Should not raise for any mode
        for mode in ["continuous", "rounded", "frozen_rounded"]:
            out = model.apply({"params": params}, x, mode=mode)
            assert out.shape == (2, 4)

    def test_prime_exp_mlp_accepts_mode(self):
        model = PrimeExpMLP(hidden_dim=4, output_dim=2, P=3)
        x = jnp.ones((2, 3))
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, x)["params"]
        for mode in ["continuous", "rounded", "frozen_rounded"]:
            out = model.apply({"params": params}, x, mode=mode)
            assert out.shape == (2, 2)

    def test_prime_exp_lstm_accepts_mode(self):
        model = PrimeExpLSTM(hidden_size=3, input_size=3, output_size=3, P=3)
        x = jnp.array([[1, 2, 0, 1]], dtype=jnp.int32)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, x)["params"]
        for mode in ["continuous", "rounded", "frozen_rounded"]:
            logits, aux = model.apply({"params": params}, x, mode=mode)
            assert logits.shape == (1, 4, 3)
            assert "z_exponents" in aux

    def test_lstm_rounded_equals_continuous_at_integers(self):
        """If exponents are already integers, rounded should match continuous."""
        model = PrimeExpLSTM(hidden_size=3, input_size=3, output_size=3, P=3)
        x = jnp.array([[1, 2, 0]], dtype=jnp.int32)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, x)["params"]
        # Round params to integers
        params = {
            k: jnp.round(v) if 'z_' in k else v
            for k, v in params.items()
        }
        logits_c, _ = model.apply({"params": params}, x, mode="continuous")
        logits_r, _ = model.apply({"params": params}, x, mode="rounded")
        np.testing.assert_allclose(logits_c, logits_r, atol=1e-5)


# ===================================================================
# Backward compatibility
# ===================================================================

class TestBackwardCompatibility:
    """Ensure existing code works without QAT flags."""

    def test_default_mode_is_continuous(self):
        """Calling without mode arg should use continuous (original behavior)."""
        model = PrimeExpLinear(features=4, P=3)
        x = jnp.ones((2, 3))
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, x)["params"]
        # Without mode arg
        out1 = model.apply({"params": params}, x)
        # With explicit continuous
        out2 = model.apply({"params": params}, x, mode="continuous")
        np.testing.assert_allclose(out1, out2, atol=1e-7)

    def test_config_defaults_qat_disabled(self):
        cfg = PrimeRationalConfig()
        assert cfg.qat_enabled is False
        assert cfg.int_attraction_mu_max == 0.0
        assert cfg.E_max == 6.0

    def test_reconstruct_weight_default_unchanged(self):
        """reconstruct_weight uses sign_ste(u) for sign, not tanh(u)."""
        P = 3
        log_primes = get_log_primes(P)
        z = jnp.array([[0.3, 0.7, -0.5]])
        u = jnp.array([0.8])
        w = reconstruct_weight(z, u, log_primes)
        # sign(0.8) = +1, so weight = +1 * exp(z^T log(p))
        logmag = jnp.sum(z * log_primes, axis=-1)
        logmag = jnp.clip(logmag, -10.0, 10.0)
        expected = 1.0 * jnp.exp(logmag)
        np.testing.assert_allclose(w, expected, atol=1e-6)
