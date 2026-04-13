"""Golden free-form RNN for aⁿbⁿ (Lan et al. 2022 / Abudy et al. 2025).

Network diagram: Abudy et al. (2025, arXiv:2505.13398v2), Appendix Figure 7.
Reference: |H| = 137 bits (paper reports 139; 2-bit difference likely
from encoding convention). Test |D:H| = 2.94 bits (Abudy et al. 2025).

Architecture (7 units, topological order):
  0: Linear  (# input passthrough)
  1: Linear  (a input passthrough)
  2: Linear  (b input passthrough)
  3: ReLU    (counter: accumulates a's, decrements for b's)
  4: Sigmoid (output P(#): sigmoid(-15) ≈ 0 always)
  5: Linear  (output P(a): -3*x_b + 7/3)
  6: Step    (output P(b): step(counter))

Connections:
  unit 1 → unit 3  (forward, w=1):    a input adds 1 to counter
  unit 2 → unit 3  (forward, w=-1):   b input subtracts 1 from counter
  unit 3 → unit 3  (recurrent, w=1):  counter carries forward
  unit 3 → unit 6  (forward, w=1):    counter feeds P(b) step function
  unit 2 → unit 5  (forward, w=-3):   b input suppresses P(a)

Biases:
  unit 4: -15  (P(#) always ≈ 0 via sigmoid)
  unit 5: 7/3  (P(a) calibrated so P(a)/(P(a)+P(b)) = 0.7 when p=0.3)

Output normalization: zero negatives, normalize (Lan et al. 2022, §3.4).
  During a-phase: P(a) = (7/3) / (7/3 + 1) = 0.7, P(b) = 0.3
  During b-phase: P(a) = max(0, -3+7/3) = 0, P(b) = step(counter) = 1
  After last b: counter = 0, step(0) = 0, all outputs ≈ 0 → uniform
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from .freeform_rnn import FreeFormTopology, freeform_forward, zero_neg_normalize
from .freeform_coding import freeform_codelength


TOPOLOGY = FreeFormTopology(
    n_units=7,
    activations=(
        "linear", "linear", "linear",  # input units
        "relu",                          # counter
        "sigmoid",                       # P(#)
        "linear",                        # P(a)
        "unsigned_step",                 # P(b)
    ),
    connections=(
        # Input wiring (implicit in encoding, explicit for forward pass)
        ("input", 0, 0),    # # → unit 0
        ("input", 1, 1),    # a → unit 1
        ("input", 2, 2),    # b → unit 2
        # Counter
        ("forward", 1, 3),   # a → counter, w=2
        ("forward", 2, 3),   # b → counter, w=-1
        ("recurrent", 3, 3), # counter self-loop, w=1
        # Outputs
        ("forward", 3, 6),   # counter → P(b), w=1
        ("forward", 2, 5),   # b → P(a), w=-3
    ),
    biased_units=frozenset({4, 5}),
    output_units=(4, 5, 6),  # P(#), P(a), P(b)
    input_size=3,
    output_size=3,
)

# Exact rational weights (same order as TOPOLOGY.connections)
WEIGHTS_RATIONAL = [
    Fraction(1), Fraction(1), Fraction(1),   # input (not counted in |H|)
    Fraction(1), Fraction(-1), Fraction(1),  # counter connections
    Fraction(1), Fraction(-3),               # output connections
]

BIASES_RATIONAL = {
    4: Fraction(-15),   # P(#)
    5: Fraction(7, 3),  # P(a)
}

# Float32 weight array for forward pass
WEIGHTS_FLOAT = np.array(
    [float(w) for w in WEIGHTS_RATIONAL]
    + [float(BIASES_RATIONAL[u]) for u in TOPOLOGY.sorted_biased_units()],
    dtype=np.float32,
)


def build_golden_freeform_anbn_params(**kwargs):
    """Return the weight array for the golden free-form aⁿbⁿ network."""
    return jnp.array(WEIGHTS_FLOAT)


def golden_freeform_anbn_forward(weights, x):
    """Forward pass returning log-probabilities (for NLL computation).

    Uses zero-negative normalization (Lan et al. 2022), then log.

    Args:
        weights: ignored (uses WEIGHTS_FLOAT).  Accepts for API compat.
        x: int32 (batch, seq_len) input tokens.

    Returns:
        log_probs: float32 (batch, seq_len, 3) log-probabilities.
    """
    w = jnp.array(WEIGHTS_FLOAT)
    logits = freeform_forward(TOPOLOGY, w, x)
    probs = zero_neg_normalize(logits)
    return jnp.log(probs + 1e-10)


def golden_freeform_anbn_mdl_score(**kwargs) -> dict:
    """Compute |H| for the golden free-form aⁿbⁿ network."""
    return freeform_codelength(TOPOLOGY, WEIGHTS_RATIONAL, BIASES_RATIONAL)
