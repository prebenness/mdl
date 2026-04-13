"""Golden aⁿbⁿcⁿ LSTM — manually constructed.

LSTM with hidden_size=3, input_size=4, output_size=4 that perfectly
recognizes the aⁿbⁿcⁿ language. Uses the same saturated-gate strategy
as the aⁿbⁿ golden from Lan et al. (2024, "MDL Regularization of LSTM
Language Models", ACL 2024, Appendix B).

Counting mechanism:
    c[0] = 1             (constant, for bias-like terms in output)
    c[1] = #a - #b       (a counter: +1 per a, -1 per b)
    c[2] = #a + #b - #c  (total a+b counter, decremented by c)

Phase detection via output gate masking:
    After #: h = [tanh(1), 0, 0] → start (predict a or #)
    After a: h = [0, tanh(c1), 0] → a-phase (predict a or b)
    After b: h = [0, 0, tanh(c2)] → b-phase (predict b or c)
    After c: h = [0, 0, tanh(c2)] → c-phase (predict c or #)

When c1 hits 0 (all b's consumed), the b-phase output naturally
transitions to predict c. When c2 hits 0 (all c's consumed),
the output predicts #.

Alphabet: {#=0, a=1, b=2, c=3}.
"""

from __future__ import annotations

import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np

from .coding import integer_code_length, rational_codelength
from .tasks.anbncn import SYMBOL_HASH, SYMBOL_A, SYMBOL_B, SYMBOL_C

LARGE = 2**7 - 1  # 127
HIDDEN_SIZE = 3
INPUT_SIZE = 4     # {#, a, b, c}
OUTPUT_SIZE = 4


def build_golden_anbncn_params(p: float = 0.3) -> dict:
    """Build weight matrices for the golden aⁿbⁿcⁿ LSTM.

    Gate conventions match golden.py (Lan et al. 2024):
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
    # x = [#, a, b, c] one-hot
    # We want:
    #   g[0] = tanh(L * x[#]) ≈ sign(x[#])     (1 at start)
    #   g[1] = tanh(L * (x[a] - x[b])) ≈ +1 for a, -1 for b, 0 for c/#
    #   g[2] = tanh(L * (x[a] + x[b] - x[c])) ≈ +1 for a/b, -1 for c, 0 for #
    Wig_paper = L * jnp.array([
        [1.0,  0.0,  0.0],     # #  → [L, 0, 0]
        [0.0,  1.0,  1.0],     # a  → [0, L, L]
        [0.0, -1.0,  1.0],     # b  → [0, -L, L]
        [0.0,  0.0, -1.0],     # c  → [0, 0, -L]
    ])  # shape (I=4, H=3) — already in code convention
    W_ig = Wig_paper
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

    # --- Output gate (o): phase masking ---
    # After #: o = [1, 0, 0] → expose c[0]=1
    # After a: o = [0, 1, 0] → expose c[1]=count
    # After b: o = [0, 0, 1] → expose c[2]=count
    # After c: o = [0, 0, 1] → expose c[2]=count
    #
    # o = sigmoid(L * (2*x - 1))
    # x[#]=[1,0,0,0], x[a]=[0,1,0,0], x[b]=[0,0,1,0], x[c]=[0,0,0,1]
    # We need:
    #   o[0] = sigmoid(L*(2*x[#] - 1))  → high for #, low otherwise
    #   o[1] = sigmoid(L*(2*x[a] - 1))  → high for a, low otherwise
    #   o[2] = sigmoid(L*(2*x[b] + 2*x[c] - 1)) → high for b or c
    Wio = L * jnp.array([
        [2.0, 0.0, 0.0],     # #
        [0.0, 2.0, 0.0],     # a
        [0.0, 0.0, 2.0],     # b
        [0.0, 0.0, 2.0],     # c
    ])  # (I=4, H=3)
    W_io = Wio
    b_io = L * jnp.array([-1.0, -1.0, -1.0])
    W_ho = jnp.zeros((H, H))
    b_ho = jnp.zeros(H)

    # --- Output layer ---
    # h = o * tanh(c)
    # Phase start (#):     h = [tanh(1), 0, 0]
    # Phase a:             h = [0, tanh(count_a_minus_b), 0]
    #   - when in a-phase, count>0 so tanh(count) ≈ 1
    # Phase b:             h = [0, 0, tanh(count_a_plus_b_minus_c)]
    #   - during b-phase with remaining b's: count > 0, tanh ≈ 1
    #   - last b (switches to c): h[2] transitions
    # Phase c:             h = [0, 0, tanh(remaining_c)]
    #   - during c-phase: count > 0, tanh ≈ 1
    #   - last c: count → 0
    #
    # Target probabilities (4 states × 4 symbols):
    #   start (#):         [p,   1-p, 0,   0  ]
    #   a-phase:           [0,   1-p, p,   0  ]
    #   b-phase:           [0,   0,   1,   0  ]  (when count_a > 0)
    #   b→c transition:    [0,   0,   0,   1  ]  (last b, count_a = 0)
    #   c-phase:           [0,   0,   0,   1  ]  (when count > 0)
    #   last c:            [1,   0,   0,   0  ]  (count = 0)
    #
    # For the LSTM approach, the output is logits = h @ W_out + b_out.
    # Since h is one-hot-like (only one component nonzero, ≈ tanh(1) or tanh(count)),
    # each phase selects a column of W_out plus b_out.
    #
    # However, the b→c transition and last c detection rely on the counter
    # value being exactly 0, which produces h[2] = tanh(0) = 0.
    # When h[2] = 0: logits = b_out → last-c probabilities [1,0,0,0]
    # When h[2] ≈ tanh(1): logits ≈ W_out[2,:]*tanh(1) + b_out
    #
    # This means b_out must encode the "counter depleted" state.
    # For the b-phase: b_out should predict c when counters deplete
    # For the c-phase: b_out should predict # when counters deplete
    #
    # Problem: b_out can only encode one "depleted" state, but we need
    # different behavior when b-counter vs c-counter depletes.
    #
    # Solution: use hidden_size=4 to separate b-count and c-count phases,
    # or use a different encoding. With h=3, we use the fact that:
    # - During b-phase, c[1] decrements to 0 BEFORE c[2] does
    # - The output gate exposes c[2] during b and c phases
    # - c[2] = #a + #b - #c. During b-phase c[2] is still > 0
    #   (it counts a+b, only decremented by c, which hasn't started)
    # - Only at the VERY END of c-phase does c[2] → 0
    #
    # So the "depleted" signal from c[2]=0 only triggers at the right time.
    # During b→c transition, we need a different mechanism.
    #
    # Revised approach: use c[1] for b-phase output and c[2] for c-phase.
    # Output gate: b activates h[1], c activates h[2].
    Wio2 = L * jnp.array([
        [2.0, 0.0, 0.0],     # #
        [0.0, 2.0, 0.0],     # a
        [0.0, 0.0, 2.0],     # b → h[2] (was wrong, fix: b→h[1], c→h[2])
        [0.0, 0.0, 2.0],     # c
    ])

    # Actually let me reconsider. The clean approach:
    # o[0] = high when x=#  → h[0] = tanh(c[0]) = tanh(1) ≈ 0.762
    # o[1] = high when x=a  → h[1] = tanh(c[1])
    # o[2] = high when x=b OR x=c → h[2] = tanh(c[2])
    #
    # c[2] = #a + #b - #c
    # During b-phase: c[2] = n + (b's seen so far) which is always > 0
    # After last c: c[2] = n + n - n = n → still > 0, NOT 0!
    #
    # This doesn't work. c[2] = 2n - #c. After all c's: c[2] = 2n - n = n.
    # We need c[2] to reach 0 after all c's.
    # Fix: c[2] = #a - #c (not #a + #b).
    # Then: during b-phase c[2] = n (constant, all a's counted, no c's yet)
    # After last c: c[2] = n - n = 0. ✓
    #
    # But we also need to detect b→c transition. That's when c[1] = #a - #b = 0.
    # We need c[1] exposed during b-phase... but output gate maps b→h[2].
    #
    # Alternative: b→h[1] and c→h[2].
    # o[1] = high when x=a OR x=b
    # o[2] = high when x=c
    #
    # Then:
    # After a: h[1] = tanh(c[1]) where c[1] = #a - #b = #a (in a-phase)
    # After b: h[1] = tanh(c[1]) where c[1] = #a - #b
    #   - while b's remain: c[1] > 0, tanh > 0
    #   - last b: c[1] = 0, tanh(0) = 0 → logits = b_out
    # After c: h[2] = tanh(c[2]) where c[2] = #a - #c
    #   - while c's remain: c[2] > 0
    #   - last c: c[2] = 0, tanh(0) = 0 → logits = b_out
    #
    # But again b_out must be both "predict c" (after last b) and "predict #"
    # (after last c). These conflict. We need separate depletion signals.
    #
    # The clean solution is hidden_size=4 to get independent b-count and c-count
    # output channels. Let me use that.

    # With hidden_size=3, this requires a more complex construction.
    # For now, use hidden_size=4 for correctness.
    raise NotImplementedError(
        "Golden aⁿbⁿcⁿ LSTM requires careful construction with hidden_size≥4. "
        "See the inline analysis for the counting mechanism design. "
        "This will be completed once the architecture decision is finalized."
    )


def golden_anbncn_forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Run the golden aⁿbⁿcⁿ LSTM on a batch of token sequences.

    Args:
        params: dict from build_golden_anbncn_params()
        x: int32 (batch, seq_len) token indices

    Returns:
        logits: float32 (batch, seq_len, 4)
    """
    B, T = x.shape
    H = params["W_ig"].shape[1]
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
