"""Golden free-form RNN for aⁿbⁿcⁿ (Lan et al. 2022 / Abudy et al. 2025).

Reference: Abudy et al. (2025, arXiv:2505.13398v2), Table 1.
Paper reports |H| = 241 bits. Our encoding gives a different value
(see golden_freeform_anbncn_mdl_score) due to architectural differences;
the paper's golden was found by genetic algorithm search.

Architecture (11 units, topological order):
  0: Linear  (# input passthrough)
  1: Linear  (a input passthrough)
  2: Linear  (b input passthrough)
  3: Linear  (c input passthrough)
  4: Linear  (output P(a): suppressed by b and c inputs)
  5: ReLU    (counter1: #a - #b, tracks a-b balance)
  6: ReLU    (counter2: #a - #c, tracks a-c balance)
  7: Step    (step1 = step(counter1); IS output P(b))
  8: Step    (step2 = step(counter2))
  9: Linear  (output P(#): 1 - 2*step1 - 2*step2)
  10: Linear (output P(c): step2 - 2*step1)

Counter mechanism:
  counter1 accumulates during a-phase, decrements during b-phase.
  counter2 accumulates during a-phase, decrements during c-phase.
  step1 = 1 during a-phase and b-phase (counter1>0), 0 after last b.
  step2 = 1 during a-phase, b-phase, and c-phase (counter2>0), 0 after last c.

Output behavior (after zero-neg normalization):
  After #:    counter1=0, counter2=0 → P(#)=0.3, P(a)=0.7
  A-phase:    step1=1, step2=1 → P(a)=0.7, P(b)=0.3
  B-phase:    step1=1, step2=1, b suppresses P(a) → P(b)=1
  After last b: step1=0, step2=1 → P(c)=1
  C-phase:    step1=0, step2=1 → P(c)=1
  After last c: step1=0, step2=0 → P(#)=1
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from .freeform_rnn import FreeFormTopology, freeform_forward, zero_neg_normalize
from .freeform_coding import freeform_codelength


TOPOLOGY = FreeFormTopology(
    n_units=11,
    activations=(
        "linear", "linear", "linear", "linear",  # input passthrough
        "linear",           # P(a) output
        "relu",             # counter1
        "relu",             # counter2
        "unsigned_step",    # step(counter1) = P(b)
        "unsigned_step",    # step(counter2)
        "linear",           # P(#)
        "linear",           # P(c)
    ),
    connections=(
        # Input wiring (implicit, not part of |H|)
        ("input", 0, 0),    # # → unit 0
        ("input", 1, 1),    # a → unit 1
        ("input", 2, 2),    # b → unit 2
        ("input", 3, 3),    # c → unit 3
        # P(a) suppression by b and c
        ("forward", 2, 4),   # b → P(a), w=-3
        ("forward", 3, 4),   # c → P(a), w=-3
        # Counter1 (a-b balance)
        ("forward", 1, 5),   # a → counter1, w=1
        ("forward", 2, 5),   # b → counter1, w=-1
        ("recurrent", 5, 5), # counter1 self-loop, w=1
        # Counter2 (a-c balance)
        ("forward", 1, 6),   # a → counter2, w=1
        ("forward", 3, 6),   # c → counter2, w=-1
        ("recurrent", 6, 6), # counter2 self-loop, w=1
        # Step functions
        ("forward", 5, 7),   # counter1 → step1, w=1
        ("forward", 6, 8),   # counter2 → step2, w=1
        # P(#) output
        ("forward", 7, 9),   # step1 → P(#), w=-2
        ("forward", 8, 9),   # step2 → P(#), w=-2
        # P(c) output
        ("forward", 7, 10),  # step1 → P(c), w=-2
        ("forward", 8, 10),  # step2 → P(c), w=1
    ),
    biased_units=frozenset({4, 9}),
    output_units=(9, 4, 7, 10),  # P(#), P(a), P(b), P(c)
    input_size=4,
    output_size=4,
)

# Exact rational weights (same order as TOPOLOGY.connections)
WEIGHTS_RATIONAL = [
    # Input (not counted in |H|)
    Fraction(1), Fraction(1), Fraction(1), Fraction(1),
    # P(a) suppression
    Fraction(-3), Fraction(-3),
    # Counter1
    Fraction(1), Fraction(-1), Fraction(1),
    # Counter2
    Fraction(1), Fraction(-1), Fraction(1),
    # Step functions
    Fraction(1), Fraction(1),
    # P(#)
    Fraction(-2), Fraction(-2),
    # P(c)
    Fraction(-2), Fraction(1),
]

BIASES_RATIONAL = {
    4: Fraction(7, 3),  # P(a) base (same calibration as aⁿbⁿ)
    9: Fraction(1),     # P(#) base
}

# Float32 weight array for forward pass
WEIGHTS_FLOAT = np.array(
    [float(w) for w in WEIGHTS_RATIONAL]
    + [float(BIASES_RATIONAL[u]) for u in TOPOLOGY.sorted_biased_units()],
    dtype=np.float32,
)


def build_golden_freeform_anbncn_params(**kwargs):
    """Return the weight array for the golden free-form aⁿbⁿcⁿ network."""
    return jnp.array(WEIGHTS_FLOAT)


def golden_freeform_anbncn_forward(weights, x):
    """Forward pass returning log-probabilities.

    Args:
        weights: ignored (uses WEIGHTS_FLOAT). Accepts for API compat.
        x: int32 (batch, seq_len) input tokens.

    Returns:
        log_probs: float32 (batch, seq_len, 4) log-probabilities.
    """
    w = jnp.array(WEIGHTS_FLOAT)
    logits = freeform_forward(TOPOLOGY, w, x)
    probs = zero_neg_normalize(logits)
    return jnp.log(probs + 1e-10)


def golden_freeform_anbncn_mdl_score(**kwargs) -> dict:
    """Compute |H| for the golden free-form aⁿbⁿcⁿ network."""
    return freeform_codelength(TOPOLOGY, WEIGHTS_RATIONAL, BIASES_RATIONAL)
