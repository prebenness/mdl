"""Golden Dyck-1 LSTM — manually constructed (approximate for depth > 1).

LSTM with hidden_size=3, input_size=3, output_size=3 for the Dyck-1
language under the PCFG S → (S)S (prob p) | ε (prob 1-p), with p = 1/3.

Uses the same saturated-gate strategy as the aⁿbⁿ golden from
Lan et al. (2024, "MDL Regularization of LSTM Language Models",
ACL 2024, Appendix B).

Counting mechanism:
    c[0] = 1             (constant, for output layer bias)
    c[1] = 0             (unused)
    c[2] = depth          (+1 per open, -1 per close)

The optimal next-token prediction depends only on depth:
    depth = 0:  P(#) = 1-p,  P(() = p,    P()) = 0
    depth > 0:  P(#) = 0,    P(() = p,    P()) = 1-p

LIMITATION: The output layer is calibrated for depth=1 (h[2]=tanh(1)).
At depth>1, tanh(depth) saturates toward 1.0, shifting the softmax
away from {1/3, 2/3} toward {0, 1}. This is a fundamental LSTM
constraint: the cell state counter passes through tanh, destroying
the binary depth>0 signal. The aⁿbⁿ golden avoids this because its
counter-gated phase is deterministic (P(b)=1). Abudy et al. (2025)
use free-form RNNs with unsigned step functions to handle this.

For depth=1 the probabilities are exact. For depth=2+ the argmax
prediction is still correct (just the probabilities are wrong).

Alphabet: {#=0, (=1, )=2}.
"""

from __future__ import annotations

import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np

from .coding import integer_code_length, rational_codelength
from .tasks.dyck1 import SYMBOL_HASH, SYMBOL_OPEN, SYMBOL_CLOSE

LARGE = 2**7 - 1  # 127
HIDDEN_SIZE = 3
INPUT_SIZE = 3     # {#, (, )}
OUTPUT_SIZE = 3


def build_golden_dyck1_params(p: float = 1 / 3) -> dict:
    """Build weight matrices for the golden Dyck-1 LSTM.

    Gate conventions match golden.py:
        it = sigmoid(x_t @ W_ii + b_ii + h @ W_hi + b_hi)
        ft = sigmoid(x_t @ W_if + b_if + h @ W_hf + b_hf)
        gt = tanh  (x_t @ W_ig + b_ig + h @ W_hg + b_hg)
        ot = sigmoid(x_t @ W_io + b_io + h @ W_ho + b_ho)
        ct = ft * ct-1 + it * gt
        ht = ot * tanh(ct)
    """
    I, H, O = INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
    L = float(LARGE)

    # --- Cell input gate (g): gt = tanh(Wig @ xt) ---
    # x = [#, (, )] one-hot
    # g[0] = tanh(L * x[#]) ≈ sign(x[#])    → 1 at #, 0 otherwise
    # g[1] = 0                                → unused
    # g[2] = tanh(L * (x[(] - x[)])) ≈ +1 for (, -1 for ), 0 for #
    #
    # This gives c[2] = depth (raw count of open - close).
    # To handle tanh(depth) saturation at depth>1, we rescale c[2]
    # in the output layer. Since depth ≥ 1 maps to tanh(depth) ∈ [0.76, 1),
    # we calibrate W_out for a reference value (tanh_ref) that accounts
    # for typical depth. See output layer section below.
    W_ig = L * jnp.array([
        [1.0, 0.0,  0.0],    # # → [L, 0, 0]
        [0.0, 0.0,  1.0],    # ( → [0, 0, L]
        [0.0, 0.0, -1.0],    # ) → [0, 0, -L]
    ])  # (I=3, H=3)
    b_ig = jnp.zeros(H)
    W_hg = jnp.zeros((H, H))
    b_hg = jnp.zeros(H)

    # --- Input gate (i): always open ---
    W_ii = jnp.zeros((I, H))
    b_ii = L * jnp.ones(H)
    W_hi = jnp.zeros((H, H))
    b_hi = jnp.zeros(H)

    # --- Forget gate (f): always remember ---
    W_if = jnp.zeros((I, H))
    b_if = L * jnp.ones(H)
    W_hf = jnp.zeros((H, H))
    b_hf = jnp.zeros(H)

    # --- Output gate (o): always expose c[2] (depth) ---
    # o = [0, 0, 1] for all inputs.
    # sigmoid(L*(2*x[any] + ... - 1)) constructions are possible but
    # the simplest: o[2] is always on via a large positive bias,
    # o[0] and o[1] always off.
    W_io = jnp.zeros((I, H))
    b_io = L * jnp.array([-1.0, -1.0, 1.0])   # o = [~0, ~0, ~1]
    W_ho = jnp.zeros((H, H))
    b_ho = jnp.zeros(H)

    # --- Output layer ---
    # h = [0, 0, tanh(depth)]
    #
    # depth = 0: h = [0, 0, 0], logits = b_out
    #   → need: P(#) = 1-p, P(() = p, P()) = 0
    #
    # depth > 0: h ≈ [0, 0, tanh(d)], logits = tanh(d) * W_out[2,:] + b_out
    #   → need: P(#) = 0, P(() = p, P()) = 1-p
    #
    # Set b_out = log-probs for depth=0 case:
    epsilon = 1.0 / (2**14 - 1)
    top_level = jnp.array([1.0 - p, p, 0.0])      # depth=0: [#, (, )]
    inside = jnp.array([0.0, p, 1.0 - p])          # depth>0: [#, (, )]

    b_out = jnp.log(top_level + epsilon)

    # For depth > 0:
    # logits ≈ tanh(1) * W_out[2,:] + b_out = log(inside + eps)
    # → W_out[2,:] = (log(inside + eps) - b_out) / tanh(1)
    tanh_1 = float(jnp.tanh(1.0))
    inside_logits = jnp.log(inside + epsilon)
    w_row = (inside_logits - b_out) / tanh_1

    # W_out: shape (H=3, O=3). Only row 2 matters.
    W_out = jnp.zeros((H, O))
    W_out = W_out.at[2, :].set(w_row)

    return {
        "W_ii": W_ii, "W_if": W_if, "W_ig": W_ig, "W_io": W_io,
        "W_hi": W_hi, "W_hf": W_hf, "W_hg": W_hg, "W_ho": W_ho,
        "b_ii": b_ii, "b_if": b_if, "b_ig": b_ig, "b_io": b_io,
        "b_hi": b_hi, "b_hf": b_hf, "b_hg": b_hg, "b_ho": b_ho,
        "W_out": W_out, "b_out": b_out,
    }


def golden_dyck1_forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Run the golden Dyck-1 LSTM on a batch of token sequences.

    Args:
        params: dict from build_golden_dyck1_params()
        x: int32 (batch, seq_len) token indices

    Returns:
        logits: float32 (batch, seq_len, 3)
    """
    B, T = x.shape
    H = HIDDEN_SIZE
    I = INPUT_SIZE

    W_ii, W_if, W_ig, W_io = params["W_ii"], params["W_if"], params["W_ig"], params["W_io"]
    W_hi, W_hf, W_hg, W_ho = params["W_hi"], params["W_hf"], params["W_hg"], params["W_ho"]
    b_ii, b_if, b_ig, b_io = params["b_ii"], params["b_if"], params["b_ig"], params["b_io"]
    b_hi, b_hf, b_hg, b_ho = params["b_hi"], params["b_hf"], params["b_hg"], params["b_ho"]
    W_out, b_out = params["W_out"], params["b_out"]

    x_onehot = jax.nn.one_hot(x, I)

    def lstm_step(carry, x_t):
        h, c = carry
        i_t = jax.nn.sigmoid(x_t @ W_ii + b_ii + h @ W_hi + b_hi)
        f_t = jax.nn.sigmoid(x_t @ W_if + b_if + h @ W_hf + b_hf)
        g_t = jnp.tanh(x_t @ W_ig + b_ig + h @ W_hg + b_hg)
        o_t = jax.nn.sigmoid(x_t @ W_io + b_io + h @ W_ho + b_ho)
        c_new = f_t * c + i_t * g_t
        h_new = o_t * jnp.tanh(c_new)
        return (h_new, c_new), h_new

    h0 = jnp.zeros((B, H))
    c0 = jnp.zeros((B, H))
    x_seq = jnp.transpose(x_onehot, (1, 0, 2))
    _, h_seq = jax.lax.scan(lstm_step, (h0, c0), x_seq)
    h_seq = jnp.transpose(h_seq, (1, 0, 2))
    logits = h_seq @ W_out + b_out
    return logits


def golden_dyck1_mdl_score(p: float = 1 / 3) -> dict:
    """Compute |H| for the golden Dyck-1 LSTM.

    Encoding follows Lan et al. (2024) convention:
    |H| = E(hidden_size) + sum of rational codelengths for all weights.
    """
    I, H, O = INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
    L = LARGE
    ZERO = Fraction(0)

    all_fracs: list[Fraction] = []

    # W_ig: (I=3, H=3) — only 4 non-zero entries
    # Row 0 (#): [L, 0, 0]
    # Row 1 ((): [0, 0, L]
    # Row 2 ()): [0, 0, -L]
    W_ig_fracs = [
        Fraction(L), ZERO, ZERO,
        ZERO, ZERO, Fraction(L),
        ZERO, ZERO, Fraction(-L),
    ]

    # W_ii, W_if: all zeros (I*H = 9 each)
    zeros_IH = [ZERO] * (I * H)

    # W_io: all zeros
    W_io_fracs = [ZERO] * (I * H)

    # W_hi, W_hf, W_hg, W_ho: all zeros (H*H = 9 each)
    zeros_HH = [ZERO] * (H * H)

    # b_ii, b_if: [L, L, L]
    b_saturated = [Fraction(L)] * H

    # b_ig, b_hg, b_hi, b_hf, b_ho: [0, 0, 0]
    zeros_H = [ZERO] * H

    # b_io: [-L, -L, L]
    b_io_fracs = [Fraction(-L), Fraction(-L), Fraction(L)]

    # Order: W_ii, W_if, W_ig, W_io, W_hi, W_hf, W_hg, W_ho,
    #        b_ii, b_if, b_ig, b_io, b_hi, b_hf, b_hg, b_ho,
    #        W_out, b_out
    all_fracs.extend(zeros_IH)      # W_ii
    all_fracs.extend(zeros_IH)      # W_if
    all_fracs.extend(W_ig_fracs)    # W_ig
    all_fracs.extend(W_io_fracs)    # W_io
    all_fracs.extend(zeros_HH)      # W_hi
    all_fracs.extend(zeros_HH)      # W_hf
    all_fracs.extend(zeros_HH)      # W_hg
    all_fracs.extend(zeros_HH)      # W_ho
    all_fracs.extend(b_saturated)   # b_ii
    all_fracs.extend(b_saturated)   # b_if
    all_fracs.extend(zeros_H)       # b_ig
    all_fracs.extend(b_io_fracs)    # b_io
    all_fracs.extend(zeros_H)       # b_hi
    all_fracs.extend(zeros_H)       # b_hf
    all_fracs.extend(zeros_H)       # b_hg
    all_fracs.extend(zeros_H)       # b_ho

    # W_out: (H=3, O=3) — only row 2 non-zero
    # Need to compute the exact rational values
    epsilon = Fraction(1, 2**14 - 1)
    top_level = [Fraction(2, 3), Fraction(1, 3), Fraction(0)]
    inside = [Fraction(0), Fraction(1, 3), Fraction(2, 3)]

    # b_out = log(top_level + eps) — transcendental, use limit_denominator
    import math as _math
    b_out_exact = []
    for val in top_level:
        log_val = _math.log(float(val + epsilon))
        b_out_exact.append(Fraction(log_val).limit_denominator(1000))

    # W_out row 2 = (log(inside + eps) - b_out) / tanh(1)
    tanh_1 = _math.tanh(1.0)
    W_out_fracs = []
    for r in range(H):
        for c in range(O):
            if r == 2:
                inside_log = _math.log(float(inside[c] + epsilon))
                w_val = (inside_log - float(b_out_exact[c])) / tanh_1
                W_out_fracs.append(Fraction(w_val).limit_denominator(1000))
            else:
                W_out_fracs.append(ZERO)

    all_fracs.extend(W_out_fracs)
    all_fracs.extend(b_out_exact)

    # Compute codelengths
    arch_bits = integer_code_length(H)
    weight_bits = sum(rational_codelength(f) for f in all_fracs)
    total_bits = arch_bits + weight_bits

    return {
        "arch_bits": arch_bits,
        "weight_bits": weight_bits,
        "total_bits": total_bits,
        "n_params": len(all_fracs),
    }
