"""Golden free-form RNN for Dyck-1 (Lan et al. 2022 / Abudy et al. 2025).

Reference: Abudy et al. (2025, arXiv:2505.13398v2), Table 1.
Our |H| = 115 bits (paper reports 113; 2-bit encoding convention diff).

Architecture (7 units, topological order):
  0: Linear  (# input passthrough)
  1: Linear  (( input passthrough)
  2: Linear  () input passthrough)
  3: ReLU    (depth counter: ( +1, ) -1, self-loop)
  4: Linear  (output P(#): 2 - 3*depth)
  5: Linear  (output P((): constant 1)
  6: Linear  (output P()):  2*depth)

Connections (encoded):
  unit 1 → unit 3  (forward, w=1):   ( increments counter
  unit 2 → unit 3  (forward, w=-1):  ) decrements counter
  unit 3 → unit 3  (recurrent, w=1): counter carries forward
  unit 3 → unit 4  (forward, w=-3):  counter suppresses P(#)
  unit 3 → unit 6  (forward, w=2):   counter feeds P())

Biases:
  unit 4: 2   (P(#) base value at depth=0)
  unit 5: 1   (P(() constant)

Output normalization: zero negatives, normalize (Lan et al. 2022, §3.4).
  depth=0: P(#) = 2/(2+1) = 2/3, P(() = 1/3, P()) = 0
  depth=1: P(#) = max(0, 2-3) = 0, P(() = 1, P()) = 2 → P(() = 1/3, P()) = 2/3
  depth>1: probabilities drift from 1/3, 2/3 (counter not clipped to binary).

Note: This design uses raw counter values (no unsigned_step), so
probabilities at depth>1 deviate from the PCFG-optimal {1/3, 2/3}.
The paper's golden likely does the same (unsigned_step would cost 8 extra
bits, pushing |H| well above 113).
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from .freeform_rnn import FreeFormTopology, freeform_forward, zero_neg_normalize
from .freeform_coding import freeform_codelength


TOPOLOGY = FreeFormTopology(
    n_units=7,
    activations=(
        "linear", "linear", "linear",  # input passthrough
        "relu",                          # depth counter
        "linear",                        # P(#)
        "linear",                        # P(()
        "linear",                        # P())
    ),
    connections=(
        # Input wiring (implicit, not part of |H|)
        ("input", 0, 0),    # # → unit 0
        ("input", 1, 1),    # ( → unit 1
        ("input", 2, 2),    # ) → unit 2
        # Counter
        ("forward", 1, 3),   # ( → counter, w=1
        ("forward", 2, 3),   # ) → counter, w=-1
        ("recurrent", 3, 3), # counter self-loop, w=1
        # Outputs
        ("forward", 3, 4),   # counter → P(#), w=-3
        ("forward", 3, 6),   # counter → P()), w=2
    ),
    biased_units=frozenset({4, 5}),
    output_units=(4, 5, 6),  # P(#), P((), P())
    input_size=3,
    output_size=3,
)

# Exact rational weights (same order as TOPOLOGY.connections)
WEIGHTS_RATIONAL = [
    Fraction(1), Fraction(1), Fraction(1),  # input (not counted in |H|)
    Fraction(1), Fraction(-1), Fraction(1),  # counter connections
    Fraction(-3), Fraction(2),               # output connections
]

BIASES_RATIONAL = {
    4: Fraction(2),  # P(#) base
    5: Fraction(1),  # P(() constant
}

# Float32 weight array for forward pass
WEIGHTS_FLOAT = np.array(
    [float(w) for w in WEIGHTS_RATIONAL]
    + [float(BIASES_RATIONAL[u]) for u in TOPOLOGY.sorted_biased_units()],
    dtype=np.float32,
)


def build_golden_freeform_dyck1_params(**kwargs):
    """Return the weight array for the golden free-form Dyck-1 network."""
    return jnp.array(WEIGHTS_FLOAT)


def golden_freeform_dyck1_forward(weights, x):
    """Forward pass returning log-probabilities.

    Args:
        weights: ignored (uses WEIGHTS_FLOAT). Accepts for API compat.
        x: int32 (batch, seq_len) input tokens.

    Returns:
        log_probs: float32 (batch, seq_len, 3) log-probabilities.
    """
    w = jnp.array(WEIGHTS_FLOAT)
    logits = freeform_forward(TOPOLOGY, w, x)
    probs = zero_neg_normalize(logits)
    return jnp.log(probs + 1e-10)


def golden_freeform_dyck1_mdl_score(**kwargs) -> dict:
    """Compute |H| for the golden free-form Dyck-1 network."""
    return freeform_codelength(TOPOLOGY, WEIGHTS_RATIONAL, BIASES_RATIONAL)
