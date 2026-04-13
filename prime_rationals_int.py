"""Integer-arithmetic training for prime-exponent MDL, following
Ghaffari et al. (NeurIPS 2022), "Is Integer Arithmetic Enough for Deep
Learning Training?", arXiv:2207.08822.

Implements Ghaffari's fully-integer training pipeline in JAX: forward pass,
backward pass (via jax.custom_vjp), and SGD all use integer-grid arithmetic.
Float32 dtype carries integer-grid values for JAX compatibility; stochastic
rounding provides unbiased estimation at each quantization point.

The heavy linear algebra (LSTM gate matmuls, output projection) is
performed on the integer grid; pointwise nonlinearities (sigmoid, tanh,
log-softmax) remain in float, consistent with Ghaffari Section 5.

Usage:
    python prime_rationals_int.py config/prime_rationals_int/anbn_int8.yaml
"""

import math
import os
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random as jrandom
import flax.linen as nn
import numpy as np
import yaml

# Reuse existing codebase utilities
from prime_rationals import (
    first_primes,
    get_log_primes,
    reconstruct_weight,
    compute_mdl_penalty,
    discretize_params,
    collect_mdl_penalty_from_params,
    cross_entropy_bits,
    evaluate_anbn_accuracy,
    make_forward_fn,
    compute_h_bits,
    _normal_init,
    PrimeExpLSTM,
)
from src.mdl.data import (
    make_anbn_dataset,
    make_test_set,
    sequences_to_padded_arrays,
    NUM_SYMBOLS,
)
from src.mdl.evaluation import (
    compute_grammar_weighted_nll_bits,
    compute_train_dh,
)
from src.utils.checkpointing import (
    TeeLogger,
    make_experiment_dir,
    checkpoint_path,
    save_checkpoint,
    save_results,
    save_config,
)


# ===========================================================================
# Module 1: Representation Mapping Functions
# ===========================================================================

def stochastic_round(x, rng):
    """Unbiased stochastic rounding with straight-through gradient.

    Forward: returns floor(x) + Bernoulli(frac(x)), so E[output] = x.
    Backward: gradient is 1 (straight-through estimator).

    Reference: Ghaffari et al. (NeurIPS 2022), Section 3.1.
    """
    floor_x = jnp.floor(x)
    frac = x - floor_x
    u = jax.random.uniform(rng, shape=x.shape)
    rounded = floor_x + (u < frac).astype(x.dtype)
    # Straight-through: gradient flows through as if rounding didn't happen
    return x + jax.lax.stop_gradient(rounded - x)


def float_to_int(A, bits, rng):
    """Map a float tensor to dynamic fixed-point (integer-grid values).

    Uses jnp.frexp to extract exponents, aligns to a shared exponent
    e_max, then stochastically rounds scaled values to B-bit integers.

    Returns float32 tensors (not int32) so JAX autodiff works.

    Args:
        A: float32 tensor of any shape.
        bits: integer bit-width B (e.g. 8 for int8 range [-128, 127]).
        rng: JAX PRNG key.

    Returns:
        (A_int, e_max) where A_int has integer-grid float values in
        [-2^(B-1), 2^(B-1)-1] and e_max is a scalar.
    """
    half_range = 2 ** (bits - 1)  # e.g. 128 for int8

    # Extract mantissa and exponent: A = m * 2^e, 0.5 <= |m| < 1
    _, e = jnp.frexp(A)
    # frexp returns e such that A = m * 2^e.  For zero elements, e = 0.

    # Handle zeros: set their exponent to a large negative so they don't
    # affect e_max.
    is_zero = (A == 0)
    e_safe = jnp.where(is_zero, jnp.full_like(e, -1000), e)
    e_max = jnp.max(e_safe)

    # If entire tensor is zero, set e_max = 0
    all_zero = jnp.all(is_zero)
    e_max = jnp.where(all_zero, 0, e_max)

    # Scale factor: maps the largest-magnitude value to ~half_range
    # A_scaled = A * 2^(bits-1) / 2^e_max = A * 2^(bits-1 - e_max)
    scale = jnp.exp2((bits - 1) - e_max.astype(jnp.float32))
    A_scaled = A * scale

    # Stochastic round to integer grid
    A_int = stochastic_round(A_scaled, rng)

    # Clamp to B-bit signed range
    A_int = jnp.clip(A_int, -half_range, half_range - 1)

    return A_int, e_max


def int_to_float(A_int, e_max, bits):
    """Inverse mapping: integer-grid values back to float.

    float_value = A_int * 2^(e_max - (bits-1))

    Reference: Ghaffari et al. (NeurIPS 2022), Section 3.2.
    """
    return A_int * jnp.exp2(e_max.astype(jnp.float32) - (bits - 1))


# ===========================================================================
# Module 2: Integer GEMM
# ===========================================================================

def _int_matmul_impl(A_float, B_float, bits, rng):
    """Core integer matmul: quantize inputs, integer GEMM, dequantize.

    Returns (C_float, A_int, e_A, B_int, e_B) — the extra outputs are
    needed by the custom_vjp backward to reuse forward quantizations.
    """
    rng1, rng2 = jax.random.split(rng)
    A_int, e_A = float_to_int(A_float, bits, rng1)
    B_int, e_B = float_to_int(B_float, bits, rng2)
    C_int = A_int @ B_int
    e_C = e_A + e_B
    C_float = int_to_float(C_int, e_C, 2 * bits - 1)
    return C_float, A_int, e_A, B_int, e_B


def _rng_to_float(rng):
    """Bitcast uint32 RNG key to float32, preserving all bits exactly."""
    return jax.lax.bitcast_convert_type(rng, jnp.float32)


def _float_to_rng(rng_f):
    """Bitcast float32 back to uint32 RNG key."""
    return jax.lax.bitcast_convert_type(rng_f, jnp.uint32)


def make_int_matmul(bits):
    """Factory for integer matmul with custom_vjp for integer backward.

    The backward GEMMs (dA = dC @ B^T, dB = A^T @ dC) are also performed
    in integer arithmetic, matching Ghaffari et al. Section 3.3 and
    Appendix B: both forward and backward use the integer GEMM pipeline.

    RNG keys are bitcast to float32 so they can be passed as regular
    differentiable arguments to custom_vjp (avoids nondiff_argnums issues
    with JAX tracers inside jax.lax.scan). Zero tangent returned for rng.

    Reference: Ghaffari et al. (NeurIPS 2022), Section 3.3, Appendix B.
    """

    @jax.custom_vjp
    def _core(A, B, rng_f):
        rng = _float_to_rng(rng_f)
        C_float, _, _, _, _ = _int_matmul_impl(A, B, bits, rng)
        return C_float

    def _core_fwd(A, B, rng_f):
        rng = _float_to_rng(rng_f)
        rng_fwd, rng_bwd = jax.random.split(rng)
        C_float, A_int, e_A, B_int, e_B = _int_matmul_impl(
            A, B, bits, rng_fwd,
        )
        return C_float, (A_int, e_A, B_int, e_B, _rng_to_float(rng_bwd))

    def _core_bwd(res, g):
        A_int, e_A, B_int, e_B, rng_bwd_f = res
        rng_bwd = _float_to_rng(rng_bwd_f)
        rng1, rng2 = jax.random.split(rng_bwd)

        g_int, e_g = float_to_int(g, bits, rng1)

        B_int_T = jnp.swapaxes(B_int, -2, -1)
        dA_int = g_int @ B_int_T
        e_dA = e_g + e_B
        dA = int_to_float(dA_int, e_dA, 2 * bits - 1)

        g_int2, e_g2 = float_to_int(g, bits, rng2)
        A_int_T = jnp.swapaxes(A_int, -2, -1)
        dB_int = A_int_T @ g_int2
        e_dB = e_A + e_g2
        dB_full = int_to_float(dB_int, e_dB, 2 * bits - 1)

        # Sum over batch dims to match B's shape (A may be batched, B not)
        n_extra = dB_full.ndim - B_int.ndim
        if n_extra > 0:
            dB = dB_full.sum(axis=tuple(range(n_extra)))
        else:
            dB = dB_full

        return (dA, dB, jnp.zeros_like(rng_bwd_f))

    _core.defvjp(_core_fwd, _core_bwd)

    def int_matmul(A, B, rng):
        return _core(A, B, _rng_to_float(rng))

    return int_matmul


def make_int_elemwise_mul(bits):
    """Factory for integer element-wise multiply with integer backward.

    Forward: quantize a, b → a_int * b_int (element-wise) → dequantize.
    Backward: da = g * b, db = g * a, both in integer arithmetic.

    The product of two B-bit integers fits in (2B-1) bits, so dequantize
    uses the wider bit-width for the combined exponent.

    Reference: Ghaffari et al. (NeurIPS 2022), Section 3.4.
    """

    @jax.custom_vjp
    def _core(a, b, rng_f):
        rng = _float_to_rng(rng_f)
        rng1, rng2 = jax.random.split(rng)
        a_int, e_a = float_to_int(a, bits, rng1)
        b_int, e_b = float_to_int(b, bits, rng2)
        c_int = a_int * b_int
        e_c = e_a + e_b
        return int_to_float(c_int, e_c, 2 * bits - 1)

    def _core_fwd(a, b, rng_f):
        rng = _float_to_rng(rng_f)
        rng_fwd, rng_bwd = jax.random.split(rng)
        rng1, rng2 = jax.random.split(rng_fwd)
        a_int, e_a = float_to_int(a, bits, rng1)
        b_int, e_b = float_to_int(b, bits, rng2)
        c_int = a_int * b_int
        e_c = e_a + e_b
        c_float = int_to_float(c_int, e_c, 2 * bits - 1)
        return c_float, (a_int, e_a, b_int, e_b, _rng_to_float(rng_bwd))

    def _core_bwd(res, g):
        a_int, e_a, b_int, e_b, rng_bwd_f = res
        rng_bwd = _float_to_rng(rng_bwd_f)
        rng1, rng2 = jax.random.split(rng_bwd)

        g_int1, e_g1 = float_to_int(g, bits, rng1)
        da_int = g_int1 * b_int
        e_da = e_g1 + e_b
        da = int_to_float(da_int, e_da, 2 * bits - 1)

        g_int2, e_g2 = float_to_int(g, bits, rng2)
        db_int = g_int2 * a_int
        e_db = e_g2 + e_a
        db = int_to_float(db_int, e_db, 2 * bits - 1)

        return (da, db, jnp.zeros_like(rng_bwd_f))

    _core.defvjp(_core_fwd, _core_bwd)

    def int_mul(a, b, rng):
        return _core(a, b, _rng_to_float(rng))

    return int_mul


def make_int_elemwise_add(bits):
    """Factory for integer element-wise add with integer backward.

    Forward: quantize a and b to a shared exponent (via concatenation so
    float_to_int produces one e_max), then integer add, dequantize.
    Backward: gradient passes through to both inputs, re-quantized to
    the integer grid (paper: all gradients are in fixed-point).

    Reference: Ghaffari et al. (NeurIPS 2022), Section 3.4.
    """

    @jax.custom_vjp
    def _core(a, b, rng_f):
        rng = _float_to_rng(rng_f)
        ab = jnp.concatenate([a, b], axis=-1)
        ab_int, e_ab = float_to_int(ab, bits, rng)
        n = a.shape[-1]
        c_int = ab_int[..., :n] + ab_int[..., n:]
        return int_to_float(c_int, e_ab, bits)

    def _core_fwd(a, b, rng_f):
        rng = _float_to_rng(rng_f)
        rng_fwd, rng_bwd = jax.random.split(rng)
        ab = jnp.concatenate([a, b], axis=-1)
        ab_int, e_ab = float_to_int(ab, bits, rng_fwd)
        n = a.shape[-1]
        c_int = ab_int[..., :n] + ab_int[..., n:]
        c_float = int_to_float(c_int, e_ab, bits)
        return c_float, (_rng_to_float(rng_bwd),)

    def _core_bwd(res, g):
        rng_bwd_f, = res
        rng_bwd = _float_to_rng(rng_bwd_f)
        rng1, rng2 = jax.random.split(rng_bwd)
        g_int1, e_g1 = float_to_int(g, bits, rng1)
        da = int_to_float(g_int1, e_g1, bits)
        g_int2, e_g2 = float_to_int(g, bits, rng2)
        db = int_to_float(g_int2, e_g2, bits)
        return (da, db, jnp.zeros_like(rng_bwd_f))

    _core.defvjp(_core_fwd, _core_bwd)

    def int_add(a, b, rng):
        return _core(a, b, _rng_to_float(rng))

    return int_add


# ===========================================================================
# Module 3: IntegerPrimeExpLSTM (Flax module)
# ===========================================================================

class IntegerPrimeExpLSTM(nn.Module):
    """LSTM with prime-exponent weights and optional integer arithmetic.

    Mirrors PrimeExpLSTM exactly in architecture (108 parameters for
    H=I=O=3), but replaces LSTM matmuls with int_matmul when
    use_integer=True.

    When use_integer=False, falls back to standard float matmuls,
    providing an SGD baseline for fair comparison.

    Args:
        hidden_size: LSTM hidden dimension (default 3).
        input_size: input dimension / vocabulary size (default 3).
        output_size: output dimension (default 3).
        P: number of primes in basis (default 6).
        init_std: initialization std for z/u parameters (default 0.01).
        clamp_logmag: clamp range for log-magnitude (default 10.0).
        use_integer: if True, use integer arithmetic for matmuls.
        int_bits: bit-width for integer representation (default 8).
    """
    hidden_size: int = 3
    input_size: int = 3
    output_size: int = 3
    P: int = 6
    init_std: float = 0.01
    clamp_logmag: float = 10.0
    use_integer: bool = True
    int_bits: int = 8

    @nn.compact
    def __call__(self, x, rng=None):
        B_batch, T = x.shape
        H = self.hidden_size
        I = self.input_size
        P = self.P
        bits = self.int_bits

        # Parameter counts -- same layout as PrimeExpLSTM
        n_lstm_w = 4 * I * H + 4 * H * H   # 72
        n_lstm_b = 4 * H + 4 * H            # 24
        n_out_w = H * self.output_size       # 9
        n_out_b = self.output_size           # 3
        n_total = n_lstm_w + n_lstm_b + n_out_w + n_out_b  # 108

        log_primes = get_log_primes(P)

        z = self.param('z_exponents', _normal_init(self.init_std),
                       (n_total, P))
        u = self.param('u_signs', _normal_init(self.init_std),
                       (n_total,))

        all_weights = reconstruct_weight(z, u, log_primes, self.clamp_logmag)

        # --- Unpack weights (identical layout to PrimeExpLSTM) ---
        offset = 0

        def take(size):
            nonlocal offset
            w = all_weights[offset:offset + size]
            offset += size
            return w

        W_ii = take(I * H).reshape(I, H)
        W_if = take(I * H).reshape(I, H)
        W_ig = take(I * H).reshape(I, H)
        W_io = take(I * H).reshape(I, H)

        W_hi = take(H * H).reshape(H, H)
        W_hf = take(H * H).reshape(H, H)
        W_hg = take(H * H).reshape(H, H)
        W_ho = take(H * H).reshape(H, H)

        b_ii = take(H)
        b_if = take(H)
        b_ig = take(H)
        b_io = take(H)
        b_hi = take(H)
        b_hf = take(H)
        b_hg = take(H)
        b_ho = take(H)

        W_out = take(H * self.output_size).reshape(H, self.output_size)
        b_out = take(self.output_size)

        assert offset == n_total

        # --- One-hot encode input ---
        x_onehot = jax.nn.one_hot(x, I)  # (B, T, I)

        # --- Build integer ops (factories close over bit-width) ---
        use_int = self.use_integer
        if use_int:
            _int_matmul = make_int_matmul(bits)
            _int_mul = make_int_elemwise_mul(bits)
            _int_add = make_int_elemwise_add(bits)

        def _matmul(A, B, sub_rng):
            """Gate matmul: integer or float."""
            if use_int:
                return _int_matmul(A, B, sub_rng)
            else:
                return A @ B

        # --- LSTM scan ---
        # Pre-split rng: T step keys + 1 output key
        # Per timestep: 8 gate matmuls + 4 element-wise ops = 12 sub-keys
        if use_int and rng is not None:
            step_keys = jax.random.split(rng, T + 1)
            scan_keys = step_keys[:T]
            output_key = step_keys[T]
        else:
            scan_keys = jnp.zeros((T, 2), dtype=jnp.uint32)
            output_key = jrandom.PRNGKey(0)

        def lstm_step(carry, inputs):
            h, c = carry
            x_t, step_key = inputs

            if use_int:
                # 12 sub-keys: 8 gate matmuls + 4 element-wise ops
                sub_keys = jax.random.split(step_key, 12)
            else:
                sub_keys = jnp.zeros((12, 2), dtype=jnp.uint32)

            # Gate computations: x_t @ W_q + b_qi + h @ W_qh + b_qh
            pre_i = (_matmul(x_t, W_ii, sub_keys[0]) + b_ii
                     + _matmul(h, W_hi, sub_keys[1]) + b_hi)
            pre_f = (_matmul(x_t, W_if, sub_keys[2]) + b_if
                     + _matmul(h, W_hf, sub_keys[3]) + b_hf)
            pre_g = (_matmul(x_t, W_ig, sub_keys[4]) + b_ig
                     + _matmul(h, W_hg, sub_keys[5]) + b_hg)
            pre_o = (_matmul(x_t, W_io, sub_keys[6]) + b_io
                     + _matmul(h, W_ho, sub_keys[7]) + b_ho)

            # Nonlinearities stay in float (Ghaffari Section 5)
            i_t = jax.nn.sigmoid(pre_i)
            f_t = jax.nn.sigmoid(pre_f)
            g_t = jnp.tanh(pre_g)
            o_t = jax.nn.sigmoid(pre_o)

            # Cell and hidden state update — integer element-wise ops
            if use_int:
                fc = _int_mul(f_t, c, sub_keys[8])
                ig = _int_mul(i_t, g_t, sub_keys[9])
                c = _int_add(fc, ig, sub_keys[10])
                h = _int_mul(o_t, jnp.tanh(c), sub_keys[11])
            else:
                c = f_t * c + i_t * g_t
                h = o_t * jnp.tanh(c)

            return (h, c), h

        h0 = jnp.zeros((B_batch, H))
        c0 = jnp.zeros((B_batch, H))
        x_seq = jnp.transpose(x_onehot, (1, 0, 2))  # (T, B, I)

        (h_final, c_final), h_seq = jax.lax.scan(
            lstm_step, (h0, c0), (x_seq, scan_keys),
        )
        h_seq = jnp.transpose(h_seq, (1, 0, 2))  # (B, T, H)

        # --- Output layer ---
        if use_int and rng is not None:
            logits = _matmul(h_seq, W_out, output_key) + b_out
        else:
            logits = h_seq @ W_out + b_out  # (B, T, output_size)

        mdl_pen = compute_mdl_penalty(z, log_primes)

        aux = {
            "mdl_penalty": mdl_pen,
            "z_exponents": z,
            "u_signs": u,
            "n_params": n_total,
        }
        return logits, aux


# ===========================================================================
# Module 4: Integer SGD & Training Loop
# ===========================================================================

class IntTrainState(NamedTuple):
    """Minimal JIT-compatible train state for integer SGD.

    Unlike flax.training.TrainState, this does NOT use optax —
    the weight update is performed manually in integer arithmetic.
    """
    step: jnp.ndarray   # scalar int32
    params: dict
    momentum: dict


def _int_sgd_leaf(w, g, m, lr, beta, sgd_bits, rng):
    """Integer SGD update for a single parameter leaf.

    m_new = beta * m + g            (integer add at sgd_bits)
    w_new = w - lr * m_new          (integer subtract at sgd_bits)

    Uses concatenation trick to align exponents before integer add/sub.

    Reference: Ghaffari et al. (NeurIPS 2022), Appendix D, Eq. 18.
    """
    r1, r2 = jax.random.split(rng)

    # --- Momentum update: m_new = beta*m + g ---
    bm = beta * m  # scalar * array, float
    bm_g = jnp.concatenate([bm.ravel(), g.ravel()])
    bm_g_int, e_bm_g = float_to_int(bm_g, sgd_bits, r1)
    n = bm.size
    m_new_int = bm_g_int[:n] + bm_g_int[n:]
    m_new = int_to_float(m_new_int, e_bm_g, sgd_bits).reshape(m.shape)

    # --- Weight update: w_new = w - lr * m_new ---
    update = lr * m_new  # scalar * array, float
    w_upd = jnp.concatenate([w.ravel(), update.ravel()])
    w_upd_int, e_wu = float_to_int(w_upd, sgd_bits, r2)
    nw = w.size
    w_new_int = w_upd_int[:nw] - w_upd_int[nw:]
    w_new = int_to_float(w_new_int, e_wu, sgd_bits).reshape(w.shape)

    return w_new, m_new


def int_sgd_update(params, grads, momentum, lr, beta, sgd_bits, rng):
    """Integer SGD with momentum over a full param pytree.

    All arithmetic (momentum accumulation, weight update) is performed
    on the integer grid at sgd_bits (paper uses int16, Section 5).

    Reference: Ghaffari et al. (NeurIPS 2022), Appendix D.
    """
    flat_w, treedef = jax.tree.flatten(params)
    flat_g, _ = jax.tree.flatten(grads)
    flat_m, _ = jax.tree.flatten(momentum)
    leaf_rngs = jax.random.split(rng, len(flat_w))

    new_w, new_m = [], []
    for w, g, m, lrng in zip(flat_w, flat_g, flat_m, leaf_rngs):
        w_new, m_new = _int_sgd_leaf(w, g, m, lr, beta, sgd_bits, lrng)
        new_w.append(w_new)
        new_m.append(m_new)

    return treedef.unflatten(new_w), treedef.unflatten(new_m)


def create_int_anbn_train_state(model, rng, seq_len, batch_size):
    """Initialize train state for integer SGD (no optax)."""
    dummy_x = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    rng_init, rng_model = jax.random.split(rng)
    params = model.init(rng_init, dummy_x, rng=rng_model)["params"]
    # Zero-initialize momentum buffer matching param shapes
    momentum = jax.tree.map(jnp.zeros_like, params)
    return IntTrainState(
        step=jnp.int32(0),
        params=params,
        momentum=momentum,
    )


def make_int_anbn_train_step(lambda_mdl, n_train, P, use_integer, int_bits,
                             sgd_bits, lr, beta, model_apply_fn):
    """Create JIT-compiled training step with integer forward, backward, and SGD.

    The custom_vjp on int_matmul/int_elemwise_* ensures gradients are
    computed in integer arithmetic. Then int_sgd_update applies the weight
    update in integer arithmetic at sgd_bits (int16 per paper Section 5).

    Reference: Ghaffari et al. (NeurIPS 2022), Sections 3-4, Appendix D.
    """
    log_primes = get_log_primes(P)

    @jax.jit
    def train_step(state, x, y, mask, rng):
        rng_fwd, rng_sgd = jax.random.split(rng)

        def loss_fn(params):
            logits, aux = model_apply_fn(
                {"params": params}, x, rng=rng_fwd,
            )
            data_nll = cross_entropy_bits(logits, y, mask)
            mdl_reg = aux["mdl_penalty"]
            loss = data_nll + (lambda_mdl / n_train) * mdl_reg
            return loss, {
                "data_nll": data_nll,
                "mdl_reg": mdl_reg,
                "z_exponents": aux["z_exponents"],
            }

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params,
        )

        # Integer SGD weight update
        if use_integer:
            new_params, new_momentum = int_sgd_update(
                state.params, grads, state.momentum,
                lr, beta, sgd_bits, rng_sgd,
            )
        else:
            # Float SGD baseline (for comparison)
            new_momentum = jax.tree.map(
                lambda m, g: beta * m + g, state.momentum, grads,
            )
            new_params = jax.tree.map(
                lambda w, m: w - lr * m, state.params, new_momentum,
            )

        new_state = IntTrainState(
            step=state.step + 1,
            params=new_params,
            momentum=new_momentum,
        )
        return new_state, loss, aux

    return train_step


# ===========================================================================
# Module 5: Evaluation
# ===========================================================================

def evaluate_int_anbn_accuracy(model, params, test_inputs, test_targets,
                               rng, batch_size=64):
    """Evaluate integer-mode accuracy on a^n b^n test set.

    Uses a fixed rng for deterministic integer evaluation.
    Processes all strings in a single forward pass to avoid repeated
    JIT compilation and redundant scan steps from per-batch padding.
    """
    from src.mdl.data import SYMBOL_B
    N = len(test_inputs)
    max_len = max(len(s) for s in test_inputs)

    x_pad = np.zeros((N, max_len), dtype=np.int32)
    y_pad = np.zeros((N, max_len), dtype=np.int32)
    det_mask = np.zeros((N, max_len), dtype=np.float32)

    for i, (inp, tgt) in enumerate(zip(test_inputs, test_targets)):
        L = len(inp)
        x_pad[i, :L] = inp
        y_pad[i, :L] = tgt
        for t in range(L):
            if inp[t] == SYMBOL_B:
                det_mask[i, t] = 1.0

    x_jnp = jnp.array(x_pad)
    y_jnp = jnp.array(y_pad)
    det_mask_jnp = jnp.array(det_mask)

    logits, _ = model.apply({"params": params}, x_jnp, rng=rng)
    preds = jnp.argmax(logits, axis=-1)

    correct = (preds == y_jnp).astype(jnp.float32)
    n_det = jnp.sum(det_mask_jnp, axis=-1)
    n_correct = jnp.sum(correct * det_mask_jnp, axis=-1)
    accs = np.array(jnp.where(n_det > 0, n_correct / n_det, 1.0))

    perfect = accs >= 1.0 - 1e-6
    n_perfect = int(np.sum(perfect))

    gen_n = 0
    for i in range(N):
        if perfect[i]:
            gen_n = i + 1
        else:
            break

    first_failure = None
    for i in range(N):
        if not perfect[i]:
            first_failure = i + 1
            break

    return {
        "mean_accuracy": float(np.mean(accs)),
        "n_perfect": n_perfect,
        "gen_n": gen_n,
        "first_failure_n": first_failure,
        "per_string_acc": accs,
    }


def make_int_forward_fn(model, params, rng):
    """Create forward_fn for integer model, compatible with evaluation utils."""
    def forward_fn(x_batch):
        logits, _ = model.apply({"params": params}, x_batch, rng=rng)
        return logits
    return forward_fn


def make_float_forward_fn_from_int_model(model_float, params):
    """Create forward_fn for float-mode model (use_integer=False)."""
    def forward_fn(x_batch):
        logits, _ = model_float.apply({"params": params}, x_batch, rng=None)
        return logits
    return forward_fn


# ===========================================================================
# Module 6: Configuration
# ===========================================================================

@dataclass
class IntegerTrainingConfig:
    """Configuration for integer-arithmetic prime-exponent training."""
    task: str = "anbn"
    P: int = 6
    hidden_size: int = 3
    lambda_mdl: float = 100.0
    lr: float = 0.01
    epochs: int = 10000
    seed: int = 42
    init_std: float = 0.01
    clamp_logmag: float = 10.0
    eval_every: int = 200
    log_every: int = 200
    num_train: int = 1000
    p: float = 0.3
    data_seed: int = 0
    test_max_n: int = 1500

    # Integer training fields
    int_bits: int = 8
    sgd_bits: int = 16  # Ghaffari Section 5: int16 for SGD weight updates
    momentum: float = 0.9
    use_integer: bool = True


# ===========================================================================
# Main experiment
# ===========================================================================

def run_int_anbn_experiment(config):
    """Run ANBN experiment with integer-arithmetic prime-exponent LSTM."""
    mode_str = f"int{config.int_bits}" if config.use_integer else "float_sgd"
    print("=" * 60)
    print(f"Integer-Arithmetic ANBN Experiment ({mode_str})")
    print("=" * 60)

    run_dir = make_experiment_dir(
        "prime_rationals_int",
        f"{mode_str}_P{config.P}_lam{config.lambda_mdl}_lr{config.lr}",
    )
    print(f"Results dir: {run_dir}")
    save_config(run_dir, {
        f.name: getattr(config, f.name) for f in fields(config)
    })

    rng = jrandom.PRNGKey(config.seed)

    # --- Data ---
    train_inputs, train_targets = make_anbn_dataset(
        num_strings=config.num_train, p=config.p, seed=config.data_seed,
    )
    x_train, y_train, mask_train = sequences_to_padded_arrays(
        train_inputs, train_targets,
    )
    n_train = len(train_inputs)
    test_inputs, test_targets = make_test_set(max_n=config.test_max_n)

    print(f"Train: {n_train} strings")
    print(f"Test: {len(test_inputs)} strings (max_n={config.test_max_n})")
    print(f"P={config.P}, lambda={config.lambda_mdl}, lr={config.lr}, "
          f"momentum={config.momentum}")
    print(f"use_integer={config.use_integer}, int_bits={config.int_bits}")

    # --- Model ---
    model = IntegerPrimeExpLSTM(
        hidden_size=config.hidden_size,
        input_size=NUM_SYMBOLS,
        output_size=NUM_SYMBOLS,
        P=config.P,
        init_std=config.init_std,
        clamp_logmag=config.clamp_logmag,
        use_integer=config.use_integer,
        int_bits=config.int_bits,
    )

    # Also create a float-mode copy of the model for float evaluation
    model_float = IntegerPrimeExpLSTM(
        hidden_size=config.hidden_size,
        input_size=NUM_SYMBOLS,
        output_size=NUM_SYMBOLS,
        P=config.P,
        init_std=config.init_std,
        clamp_logmag=config.clamp_logmag,
        use_integer=False,
        int_bits=config.int_bits,
    )

    rng, init_rng = jax.random.split(rng)
    state = create_int_anbn_train_state(
        model, init_rng, x_train.shape[1], x_train.shape[0],
    )
    n_model_params = sum(p.size for p in jax.tree.leaves(state.params))
    print(f"Model parameters: {n_model_params} "
          f"(108 weights x P={config.P} exponents + 108 signs)")
    print(f"sgd_bits={config.sgd_bits}")

    train_step = make_int_anbn_train_step(
        config.lambda_mdl, n_train, config.P,
        config.use_integer, config.int_bits,
        config.sgd_bits, config.lr, config.momentum,
        model.apply,
    )

    # --- Metrics log ---
    metrics_log = []

    # --- Training loop ---
    t0 = time.monotonic()
    best_gen_n = -1
    best_params = state.params

    for epoch in range(1, config.epochs + 1):
        # Fresh rng per epoch for stochastic rounding
        epoch_rng = jax.random.fold_in(rng, epoch)

        state, loss, aux = train_step(
            state, x_train, y_train, mask_train, epoch_rng,
        )

        if epoch % config.log_every == 0 or epoch == 1:
            elapsed = time.monotonic() - t0
            mean_abs_z = float(jnp.mean(jnp.abs(aux["z_exponents"])))
            print(f"  epoch {epoch:5d} | loss={float(loss):.6f} | "
                  f"nll={float(aux['data_nll']):.4f} bits | "
                  f"mdl={float(aux['mdl_reg']):.2f} | "
                  f"mean|z|={mean_abs_z:.4f} | "
                  f"time={elapsed:.1f}s")

        if epoch % config.eval_every == 0 or epoch == config.epochs:
            eval_rng = jrandom.PRNGKey(config.seed + 1)

            # Integer eval (or float eval if use_integer=False)
            if config.use_integer:
                int_result = evaluate_int_anbn_accuracy(
                    model, state.params, test_inputs, test_targets,
                    eval_rng, batch_size=64,
                )
            else:
                int_result = evaluate_anbn_accuracy(
                    model_float, state.params, test_inputs, test_targets,
                    batch_size=64,
                )

            # Float eval (always)
            float_result = evaluate_anbn_accuracy(
                model_float, state.params, test_inputs, test_targets,
                batch_size=64,
            )

            # Discretized eval
            disc_params = discretize_params(state.params, get_log_primes(config.P))
            disc_result = evaluate_anbn_accuracy(
                model_float, disc_params, test_inputs, test_targets,
                batch_size=64,
            )

            # Hypothesis bits
            h_bits = compute_h_bits(state.params, config.P)
            mean_abs_z = float(jnp.mean(jnp.abs(aux["z_exponents"])))

            # Float train loss (for comparison)
            float_logits, float_aux = model_float.apply(
                {"params": state.params}, x_train, rng=None,
            )
            float_train_nll = float(cross_entropy_bits(
                float_logits, y_train, mask_train,
            ))

            metrics_row = {
                "epoch": epoch,
                "int_train_loss": float(loss),
                "float_train_loss": float_train_nll,
                "int_test_acc": int_result["mean_accuracy"],
                "float_test_acc": float_result["mean_accuracy"],
                "disc_test_acc": disc_result["mean_accuracy"],
                "gen_n_int": int_result["gen_n"],
                "gen_n_float": float_result["gen_n"],
                "gen_n_disc": disc_result["gen_n"],
                "h_bits": h_bits,
                "mean_abs_z": mean_abs_z,
            }
            metrics_log.append(metrics_row)

            current_gen_n = int_result["gen_n"] if config.use_integer else float_result["gen_n"]
            is_best = current_gen_n > best_gen_n
            best_tag = "  * NEW BEST" if is_best else ""

            print(f"  [eval] int_acc={int_result['mean_accuracy']:.4f} | "
                  f"float_acc={float_result['mean_accuracy']:.4f} | "
                  f"disc_acc={disc_result['mean_accuracy']:.4f} | "
                  f"gen_n(int/float/disc)="
                  f"{int_result['gen_n']}/{float_result['gen_n']}/"
                  f"{disc_result['gen_n']} | "
                  f"|H|={h_bits:.2f}{best_tag}")

            if is_best:
                best_gen_n = current_gen_n
                best_params = jax.tree.map(lambda x: x.copy(), state.params)
                save_checkpoint(
                    {"params": state.params},
                    str(checkpoint_path(run_dir, "best.npz")),
                )

    elapsed = time.monotonic() - t0
    print(f"Training complete in {elapsed:.1f}s")

    # Save final checkpoint
    save_checkpoint({"params": state.params},
                    str(checkpoint_path(run_dir, "final.npz")))

    # --- Save metrics CSV ---
    import csv
    csv_path = Path(run_dir) / "metrics.csv"
    if metrics_log:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_log[0].keys())
            writer.writeheader()
            writer.writerows(metrics_log)
        print(f"  Metrics saved to {csv_path}")

    # --- Final evaluation ---
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # Use best params
    eval_rng = jrandom.PRNGKey(config.seed + 2)
    if config.use_integer:
        final_int = evaluate_int_anbn_accuracy(
            model, best_params, test_inputs, test_targets, eval_rng,
        )
    else:
        final_int = evaluate_anbn_accuracy(
            model_float, best_params, test_inputs, test_targets,
        )
    final_float = evaluate_anbn_accuracy(
        model_float, best_params, test_inputs, test_targets,
    )
    disc_params = discretize_params(best_params, get_log_primes(config.P))
    final_disc = evaluate_anbn_accuracy(
        model_float, disc_params, test_inputs, test_targets,
    )

    print(f"  Integer test accuracy: {final_int['mean_accuracy']:.4f} "
          f"(gen_n={final_int['gen_n']})")
    print(f"  Float test accuracy:   {final_float['mean_accuracy']:.4f} "
          f"(gen_n={final_float['gen_n']})")
    print(f"  Discrete test accuracy: {final_disc['mean_accuracy']:.4f} "
          f"(gen_n={final_disc['gen_n']})")

    h_bits = compute_h_bits(best_params, config.P)
    print(f"  |H| = {h_bits:.4f} bits")

    # --- Save results ---
    results = {
        "mode": mode_str,
        "int_test_accuracy": final_int["mean_accuracy"],
        "int_gen_n": final_int["gen_n"],
        "float_test_accuracy": final_float["mean_accuracy"],
        "float_gen_n": final_float["gen_n"],
        "disc_test_accuracy": final_disc["mean_accuracy"],
        "disc_gen_n": final_disc["gen_n"],
        "h_bits": h_bits,
    }
    save_results(run_dir, results)
    print(f"\nResults saved to {run_dir}")
    return state, results


# ===========================================================================
# CLI
# ===========================================================================

def _build_arg_parser(defaults=None):
    """Build argument parser. YAML defaults seed argparse so CLI overrides them."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Integer-arithmetic prime-exponent training (Ghaffari et al. 2022)",
    )
    parser.add_argument(
        "config", nargs="?", default=None,
        help="Optional YAML config path. CLI flags override YAML values.",
    )

    # Task
    parser.add_argument("--task", type=str, default="anbn",
                        help="Experiment task (default: anbn)")
    # Prime basis
    parser.add_argument("--P", type=int, default=6,
                        help="Number of primes in basis (default: 6)")
    # Architecture
    parser.add_argument("--hidden_size", type=int, default=3,
                        help="LSTM hidden size (3 matches Lan et al.)")
    # Training
    parser.add_argument("--lambda_mdl", type=float, default=100.0,
                        help="MDL regularization weight")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="SGD learning rate")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--init_std", type=float, default=0.01,
                        help="Std for exponent/sign parameter initialization")
    parser.add_argument("--clamp_logmag", type=float, default=10.0,
                        help="Clamp range for log-magnitude before exp()")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")
    # Data
    parser.add_argument("--num_train", type=int, default=1000,
                        help="Number of training strings")
    parser.add_argument("--p", type=float, default=0.3,
                        help="PCFG termination probability")
    parser.add_argument("--data_seed", type=int, default=0,
                        help="Seed for data generation")
    # Evaluation
    parser.add_argument("--test_max_n", type=int, default=1500,
                        help="Max n for test set")
    parser.add_argument("--eval_every", type=int, default=200,
                        help="Evaluate every N epochs")
    parser.add_argument("--log_every", type=int, default=200,
                        help="Log training metrics every N epochs")
    # Integer training
    parser.add_argument("--int_bits", type=int, default=8,
                        help="Bit width for integer GEMM")
    parser.add_argument("--sgd_bits", type=int, default=16,
                        help="Bit width for integer SGD (Ghaffari: int16)")
    parser.add_argument("--use_integer",
                        type=lambda v: v if isinstance(v, bool) else v.lower() in ('true', '1', 'yes'),
                        default=True,
                        help="Use integer arithmetic (False = float SGD baseline)")

    if defaults:
        valid_dests = {a.dest for a in parser._actions}
        unknown = sorted(k for k in defaults if k not in valid_dests)
        if unknown:
            print(f"Warning: ignoring unknown config keys: {', '.join(unknown)}")
            defaults = {k: v for k, v in defaults.items() if k in valid_dests}
        parser.set_defaults(**defaults)
    return parser


def main():
    """CLI entry point: python prime_rationals_int.py [config.yaml] [--flag ...]"""
    # Parse config path first so YAML defaults can seed argparse.
    pre_config_arg = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        pre_config_arg = sys.argv[1]

    yaml_defaults = {}
    if pre_config_arg is not None:
        config_path = Path(pre_config_arg)
        if config_path.exists() and config_path.suffix in (".yaml", ".yml"):
            with open(config_path) as f:
                yaml_defaults = yaml.safe_load(f) or {}

    parser = _build_arg_parser(defaults=yaml_defaults)
    args = parser.parse_args()

    config = IntegerTrainingConfig(**{
        f.name: getattr(args, f.name)
        for f in fields(IntegerTrainingConfig)
        if hasattr(args, f.name)
    })

    print(f"Config: {config}")
    print(f"Primes ({config.P}): {first_primes(config.P)}")

    if config.task == "anbn":
        run_int_anbn_experiment(config)
    else:
        print(f"Unsupported task: {config.task}")
        sys.exit(1)


if __name__ == "__main__":
    main()
