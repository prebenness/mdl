"""MDL encoding for free-form RNNs (Lan et al. 2022, TACL, arXiv:2111.00600).

Encoding scheme (Section 3.3):
  |H| = E(n_units) + sum_i encoding(unit_i)

Per-unit encoding:
  activation_cost + E(n_outgoing) + sum_j encoding(connection_j) + [bias]

Per-connection encoding:
  E(target_unit) + 1 bit (forward/recurrent) + weight_encoding

Weight encoding:
  sign (1 bit) + E(|numerator|) + E(denominator)
"""

from fractions import Fraction

from .coding import integer_code_length, rational_codelength
from .freeform_rnn import FreeFormTopology, ACTIVATION_BITS


def freeform_unit_encoding_bits(
    topology: FreeFormTopology,
    weights: list[Fraction],
    biases: dict[int, Fraction],
) -> list[dict]:
    """Compute per-unit encoding cost.

    Args:
        topology: the network topology.
        weights: list of Fraction weights, one per connection (same order
            as topology.connections).
        biases: dict mapping unit_idx -> Fraction bias value.

    Returns:
        list of dicts, one per unit, each with:
            activation_bits, n_outgoing_bits, connection_bits, bias_bits,
            total_bits.
    """
    N = topology.n_units

    # Build outgoing connection list per source unit.
    # Input connections ("input" type) are NOT encoded — they represent the
    # implicit one-hot input wiring to input units (Lan et al. 2022, §3.4:
    # "inputs and outputs are one-hot encoded over n input units").
    outgoing: list[list[tuple[int, bool, Fraction]]] = [[] for _ in range(N)]
    for ci, (src_type, src_idx, dst_idx) in enumerate(topology.connections):
        w = weights[ci]
        if src_type == "input":
            pass  # Implicit wiring, not part of |H|
        elif src_type == "forward":
            outgoing[src_idx].append((dst_idx, False, w))
        elif src_type == "recurrent":
            outgoing[src_idx].append((dst_idx, True, w))

    results = []
    for uid in range(N):
        act_bits = ACTIVATION_BITS.get(topology.activations[uid], 0)

        out_conns = outgoing[uid]
        n_outgoing_bits = integer_code_length(len(out_conns))

        conn_bits = 0
        for target, is_rec, w in out_conns:
            conn_bits += integer_code_length(target)  # E(target)
            conn_bits += 1  # forward/recurrent bit
            conn_bits += rational_codelength(w)  # weight

        # Bias
        bias_bits = 0
        if uid in biases and biases[uid] != Fraction(0):
            bias_bits = rational_codelength(biases[uid])

        total = act_bits + n_outgoing_bits + conn_bits + bias_bits
        results.append({
            "unit": uid,
            "activation_bits": act_bits,
            "n_outgoing_bits": n_outgoing_bits,
            "connection_bits": conn_bits,
            "bias_bits": bias_bits,
            "total_bits": total,
        })

    return results


def freeform_codelength(
    topology: FreeFormTopology,
    weights: list[Fraction],
    biases: dict[int, Fraction],
) -> dict:
    """Compute total |H| for a free-form RNN.

    Args:
        topology: the network topology.
        weights: Fraction weights, one per connection.
        biases: dict mapping biased unit_idx -> Fraction bias.

    Returns:
        dict with total_bits, unit_count_bits, per_unit details,
        n_params, arch_bits, weight_bits.
    """
    unit_count_bits = integer_code_length(topology.n_units)
    per_unit = freeform_unit_encoding_bits(topology, weights, biases)
    unit_total = sum(u["total_bits"] for u in per_unit)

    total_bits = unit_count_bits + unit_total

    # Separate arch vs weight bits for comparison with LSTM encoding
    # "arch" = topology structure (unit count + activations + connectivity)
    # "weight" = actual weight/bias values
    arch_bits = unit_count_bits
    weight_bits = 0
    for u in per_unit:
        arch_bits += u["activation_bits"] + u["n_outgoing_bits"]
        # Connection bits include target + type (arch) + weight (value)
        # We can't easily separate these further without re-computing,
        # so we count the full connection + bias bits as weight_bits.
        weight_bits += u["connection_bits"] + u["bias_bits"]

    return {
        "total_bits": total_bits,
        "unit_count_bits": unit_count_bits,
        "per_unit": per_unit,
        "arch_bits": arch_bits,
        "weight_bits": weight_bits,
        "n_params": topology.n_weights,
    }
