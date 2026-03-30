"""Tests for integer-arithmetic training (Ghaffari et al. NeurIPS 2022 method).

Run with:
    JAX_PLATFORMS=cpu pytest tests/test_integer_training.py -v
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import random as jrandom
import numpy as np
import pytest

from prime_rationals_int import (
    stochastic_round,
    float_to_int,
    int_to_float,
    int_matmul,
    IntegerPrimeExpLSTM,
    IntegerTrainingConfig,
    create_int_anbn_train_state,
    make_int_anbn_train_step,
    cross_entropy_bits,
)
from prime_rationals import (
    get_log_primes,
    reconstruct_weight,
    compute_mdl_penalty,
)
from src.mdl.data import (
    make_anbn_dataset,
    sequences_to_padded_arrays,
    NUM_SYMBOLS,
)


class TestRepresentationMapping:
    """Test Module 1: float_to_int / int_to_float round-trip."""

    def test_round_trip_int8(self):
        """int_to_float(float_to_int(A)) ~= A within 1 ULP at int8."""
        rng = jrandom.PRNGKey(42)
        A = jnp.array([0.5, -1.0, 0.25, 3.14, -0.001, 0.0, 127.0])
        bits = 8

        A_int, e_max = float_to_int(A, bits, rng)
        A_reconstructed = int_to_float(A_int, e_max, bits)

        # ULP at this scale: 2^(e_max) / 2^(bits-1)
        ulp = float(jnp.exp2(e_max.astype(jnp.float32) - (bits - 1)))
        error = jnp.abs(A - A_reconstructed)

        # Each element should be within 1 ULP
        assert jnp.all(error <= ulp + 1e-10), (
            f"Round-trip error exceeds 1 ULP: max_error={float(jnp.max(error))}, "
            f"ulp={ulp}"
        )

    def test_round_trip_various_bitwidths(self):
        """Round-trip works for int6, int5, int4."""
        rng = jrandom.PRNGKey(0)
        A = jnp.array([1.0, -0.5, 0.125, 2.0])

        for bits in [6, 5, 4]:
            A_int, e_max = float_to_int(A, bits, rng)
            A_recon = int_to_float(A_int, e_max, bits)
            ulp = float(jnp.exp2(e_max.astype(jnp.float32) - (bits - 1)))
            error = jnp.abs(A - A_recon)
            assert jnp.all(error <= ulp + 1e-10), (
                f"bits={bits}: max_error={float(jnp.max(error))}, ulp={ulp}"
            )

    def test_all_zeros(self):
        """float_to_int handles all-zero tensors correctly."""
        rng = jrandom.PRNGKey(0)
        A = jnp.zeros(5)
        A_int, e_max = float_to_int(A, 8, rng)
        assert jnp.all(A_int == 0), "Zero input should give zero mantissas"
        A_recon = int_to_float(A_int, e_max, 8)
        assert jnp.all(A_recon == 0), "Round-trip of zeros should be zeros"


class TestStochasticRounding:
    """Test stochastic rounding unbiasedness."""

    def test_unbiasedness(self):
        """N=10000 roundings of 3.7, mean within 3 sigma of 3.7."""
        N = 10000
        x = jnp.full((N,), 3.7)
        rng = jrandom.PRNGKey(42)
        rounded = stochastic_round(x, rng)

        mean_val = float(jnp.mean(rounded))
        # For Bernoulli(0.7): var = 0.7 * 0.3 = 0.21
        # std of mean = sqrt(0.21 / N)
        std_of_mean = np.sqrt(0.21 / N)

        assert abs(mean_val - 3.7) < 3 * std_of_mean, (
            f"Mean {mean_val} not within 3 sigma ({3 * std_of_mean}) of 3.7"
        )

    def test_integer_values(self):
        """Stochastic rounding produces integer-valued floats."""
        rng = jrandom.PRNGKey(0)
        x = jnp.array([1.3, 2.7, -0.5, 4.1])
        rounded = stochastic_round(x, rng)

        # Stop-gradient means forward values are integers
        for val in rounded:
            val_f = float(val)
            assert val_f == int(val_f), f"Not integer: {val_f}"


class TestIntegerGEMM:
    """Test Module 2: integer matrix multiplication."""

    def test_3x3_matmul(self):
        """3x3 integer GEMM matches float within quantization tolerance."""
        rng = jrandom.PRNGKey(42)
        A = jnp.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]])
        B = jnp.array([[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9]])

        C_float = A @ B
        C_int = int_matmul(A, B, 8, rng)

        # Relative tolerance: int8 has ~1% precision for these magnitudes
        rel_error = jnp.abs(C_int - C_float) / (jnp.abs(C_float) + 1e-10)
        max_rel_error = float(jnp.max(rel_error))
        assert max_rel_error < 0.25, (
            f"Integer GEMM relative error too large: {max_rel_error}"
        )

    def test_identity_matmul(self):
        """Multiplying by near-identity should roughly preserve values."""
        rng = jrandom.PRNGKey(0)
        A = jnp.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]])
        B = jnp.array([[2.0], [3.0], [4.0]])
        C_int = int_matmul(A, B, 8, rng)
        C_float = A @ B
        assert jnp.allclose(C_int, C_float, atol=1.0), (
            f"Identity matmul error: {C_int} vs {C_float}"
        )


class TestIntegerLSTM:
    """Test Module 3: IntegerPrimeExpLSTM."""

    def _make_model_and_params(self, use_integer=True, bits=8):
        """Create model and initialize params."""
        model = IntegerPrimeExpLSTM(
            hidden_size=3,
            input_size=3,
            output_size=3,
            P=6,
            init_std=0.1,
            clamp_logmag=10.0,
            use_integer=use_integer,
            int_bits=bits,
        )
        rng = jrandom.PRNGKey(42)
        rng_init, rng_model = jax.random.split(rng)
        # Single timestep input
        dummy_x = jnp.zeros((1, 3), dtype=jnp.int32)
        params = model.init(rng_init, dummy_x, rng=rng_model)["params"]
        return model, params

    def test_forward_single_timestep(self):
        """Integer LSTM forward pass on single timestep produces finite output."""
        model_int, params = self._make_model_and_params(use_integer=True, bits=8)
        model_float, _ = self._make_model_and_params(use_integer=False)

        x = jnp.array([[0, 1, 2]], dtype=jnp.int32)  # 3 timesteps
        rng = jrandom.PRNGKey(99)

        # Integer forward
        logits_int, aux_int = model_int.apply({"params": params}, x, rng=rng)
        assert logits_int.shape == (1, 3, 3), f"Shape mismatch: {logits_int.shape}"
        assert jnp.all(jnp.isfinite(logits_int)), "Integer output has NaN/Inf"

        # Float forward (same params)
        logits_float, aux_float = model_float.apply({"params": params}, x, rng=None)
        assert jnp.all(jnp.isfinite(logits_float)), "Float output has NaN/Inf"

        # They should be reasonably close (but not identical due to quantization)
        diff = float(jnp.max(jnp.abs(logits_int - logits_float)))
        # With init_std=0.1, weights are small, outputs are small
        assert diff < 5.0, f"Int vs float logit difference too large: {diff}"

    def test_output_layer_uses_int_matmul(self):
        """Verify the model runs without error in integer mode."""
        model, params = self._make_model_and_params(use_integer=True, bits=8)
        x = jnp.array([[0, 1, 2, 1, 0]], dtype=jnp.int32)
        rng = jrandom.PRNGKey(0)
        logits, _ = model.apply({"params": params}, x, rng=rng)
        assert logits.shape == (1, 5, 3)
        assert jnp.all(jnp.isfinite(logits))


class TestGradientFlow:
    """Test that gradients flow through integer LSTM via straight-through."""

    def test_gradients_nonzero(self):
        """jax.grad through integer LSTM gives non-zero finite gradients."""
        model = IntegerPrimeExpLSTM(
            hidden_size=3, input_size=3, output_size=3,
            P=6, init_std=0.1, clamp_logmag=10.0,
            use_integer=True, int_bits=8,
        )

        rng = jrandom.PRNGKey(42)
        rng_init, rng_model, rng_grad = jax.random.split(rng, 3)

        x = jnp.array([[0, 1, 2, 1, 0]], dtype=jnp.int32)
        y = jnp.array([[1, 2, 1, 0, 0]], dtype=jnp.int32)
        mask = jnp.array([[1, 1, 1, 1, 0]], dtype=jnp.float32)

        params = model.init(rng_init, x, rng=rng_model)["params"]

        def loss_fn(params):
            logits, aux = model.apply({"params": params}, x, rng=rng_grad)
            return cross_entropy_bits(logits, y, mask)

        grads = jax.grad(loss_fn)(params)

        # Check z_exponents gradients
        z_grad = grads["z_exponents"]
        assert z_grad.shape[0] == 108  # 108 weights
        assert z_grad.shape[1] == 6    # P=6
        assert jnp.any(z_grad != 0), "z_exponents gradients are all zero"
        assert jnp.all(jnp.isfinite(z_grad)), "z_exponents gradients have NaN/Inf"

        # Check u_signs gradients
        u_grad = grads["u_signs"]
        assert u_grad.shape == (108,)
        assert jnp.any(u_grad != 0), "u_signs gradients are all zero"
        assert jnp.all(jnp.isfinite(u_grad)), "u_signs gradients have NaN/Inf"

    def test_gradients_float_mode(self):
        """Float mode also produces non-zero gradients (sanity check)."""
        model = IntegerPrimeExpLSTM(
            hidden_size=3, input_size=3, output_size=3,
            P=6, init_std=0.1, clamp_logmag=10.0,
            use_integer=False, int_bits=8,
        )

        rng = jrandom.PRNGKey(42)
        rng_init, rng_model = jax.random.split(rng)

        x = jnp.array([[0, 1, 2]], dtype=jnp.int32)
        y = jnp.array([[1, 2, 0]], dtype=jnp.int32)
        mask = jnp.ones((1, 3), dtype=jnp.float32)

        params = model.init(rng_init, x, rng=rng_model)["params"]

        def loss_fn(params):
            logits, _ = model.apply({"params": params}, x, rng=None)
            return cross_entropy_bits(logits, y, mask)

        grads = jax.grad(loss_fn)(params)
        assert jnp.any(grads["z_exponents"] != 0)
        assert jnp.any(grads["u_signs"] != 0)


class TestSmokeTraining:
    """Test Module 4: short training smoke test."""

    def test_50_epochs_loss_decreases(self):
        """50 epochs of int8 training on tiny ANBN, loss should decrease.

        Uses a moderate lambda_mdl to avoid the regularization term
        dominating the gradient on a tiny dataset.
        """
        # Tiny dataset
        train_inputs, train_targets = make_anbn_dataset(
            num_strings=50, p=0.3, seed=0,
        )
        x_train, y_train, mask_train = sequences_to_padded_arrays(
            train_inputs, train_targets,
        )
        n_train = len(train_inputs)

        model = IntegerPrimeExpLSTM(
            hidden_size=3, input_size=NUM_SYMBOLS, output_size=NUM_SYMBOLS,
            P=6, init_std=0.01, clamp_logmag=10.0,
            use_integer=True, int_bits=8,
        )

        rng = jrandom.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)

        state = create_int_anbn_train_state(
            model, init_rng, x_train.shape[1], x_train.shape[0],
            lr=0.01, momentum=0.9,
        )

        # Use lambda_mdl=1.0 (not 100) so the data-fitting NLL gradient
        # dominates and the loss decreases within 50 epochs.
        train_step = make_int_anbn_train_step(
            lambda_mdl=1.0, n_train=n_train, P=6,
            use_integer=True, int_bits=8,
        )

        losses = []
        for epoch in range(1, 51):
            epoch_rng = jax.random.fold_in(rng, epoch)
            state, loss, aux = train_step(
                state, x_train, y_train, mask_train, epoch_rng,
            )
            losses.append(float(loss))

        first_loss = np.mean(losses[:5])
        last_loss = np.mean(losses[-5:])

        assert last_loss < first_loss, (
            f"Loss did not decrease: first 5 mean={first_loss:.4f}, "
            f"last 5 mean={last_loss:.4f}"
        )
        assert np.all(np.isfinite(losses)), "Training produced NaN/Inf losses"

    def test_float_sgd_baseline_runs(self):
        """Float SGD baseline (use_integer=False) runs without error."""
        train_inputs, train_targets = make_anbn_dataset(
            num_strings=20, p=0.3, seed=0,
        )
        x_train, y_train, mask_train = sequences_to_padded_arrays(
            train_inputs, train_targets,
        )
        n_train = len(train_inputs)

        model = IntegerPrimeExpLSTM(
            hidden_size=3, input_size=NUM_SYMBOLS, output_size=NUM_SYMBOLS,
            P=6, init_std=0.01, clamp_logmag=10.0,
            use_integer=False, int_bits=8,
        )

        rng = jrandom.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)

        state = create_int_anbn_train_state(
            model, init_rng, x_train.shape[1], x_train.shape[0],
            lr=0.01, momentum=0.9,
        )

        train_step = make_int_anbn_train_step(
            lambda_mdl=100.0, n_train=n_train, P=6,
            use_integer=False, int_bits=8,
        )

        for epoch in range(1, 11):
            epoch_rng = jax.random.fold_in(rng, epoch)
            state, loss, aux = train_step(
                state, x_train, y_train, mask_train, epoch_rng,
            )
            assert np.isfinite(float(loss)), f"NaN/Inf at epoch {epoch}"
