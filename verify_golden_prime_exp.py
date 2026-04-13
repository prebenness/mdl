#!/usr/bin/env python3.12
"""Verify: can the prime-exponent parameterization represent the golden network?

Encodes the Lan et al. (2024) golden ANBN LSTM as prime exponents and evaluates.
Gate-saturation weights (127) are replaced with powers of 2 (e.g., 8 = 2^3).
Output weights (transcendental) are approximated by closest prime-power rationals.

This is a diagnostic: if gen_n=1500 and |H| is reasonable, the parameterization
works and the problem is purely optimization. If not, the approach needs rethinking.
"""

import math
from itertools import product as iprod

import jax
import jax.numpy as jnp
import numpy as np

from prime_rationals import (
    first_primes,
    get_log_primes,
    reconstruct_weight,
    compute_h_bits,
    PrimeExpLSTM,
    evaluate_anbn_accuracy,
)
from src.mdl.data import make_test_set
from src.mdl.golden import build_golden_network_params


P = 6
PRIMES = first_primes(P)  # [2, 3, 5, 7, 11, 13]
LOG_PRIMES = [math.log(p) for p in PRIMES]
H = I = O = 3


def factorize_to_primes(value, max_exp=5):
    """Find integer exponent vector z such that prod(p_r^z_r) ≈ |value|.

    Returns (sign, z_vector, approx_value, abs_error).
    """
    if abs(value) < 1e-10:
        return 1, [0] * P, 1.0, 1.0

    sign = 1 if value > 0 else -1
    log_target = math.log(abs(value))

    best_z = [0] * P
    best_err = abs(log_target)

    # Exhaustive search over integer exponent grid
    ranges = [range(-max_exp, max_exp + 1) for _ in PRIMES]
    for z_tuple in iprod(*ranges):
        log_val = sum(z * lp for z, lp in zip(z_tuple, LOG_PRIMES))
        err = abs(log_val - log_target)
        if err < best_err:
            best_err = err
            best_z = list(z_tuple)

    approx = sign * math.exp(sum(z * lp for z, lp in zip(best_z, LOG_PRIMES)))
    return sign, best_z, approx, abs(approx - value)


def build_golden_prime_exp(saturation=8, zero_exp=-5, p=0.3):
    """Encode the golden network as (z_exponents, u_signs).

    Args:
        saturation: power-of-2 for gate saturation (replaces 127).
        zero_exp: exponent on prime 2 for "zero" weights. E.g., -5 gives
            2^{-5} = 1/32 ≈ 0.031 as approximation to zero.
            Set to 0 to leave zeros as ±1 (no approximation).

    Returns z (108, P), u (108,), and a diagnostic report.
    """
    golden = build_golden_network_params(p=p)

    # Packing order must match PrimeExpLSTM
    pack_order = (
        [('W_ii', I, H), ('W_if', I, H), ('W_ig', I, H), ('W_io', I, H),
         ('W_hi', H, H), ('W_hf', H, H), ('W_hg', H, H), ('W_ho', H, H)] +
        [('b_ii', H), ('b_if', H), ('b_ig', H), ('b_io', H),
         ('b_hi', H), ('b_hf', H), ('b_hg', H), ('b_ho', H)] +
        [('W_out', H, O), ('b_out', O)]
    )

    names, targets = [], []
    for item in pack_order:
        key = item[0]
        w = np.array(golden[key]).flatten()
        for i, v in enumerate(w):
            names.append(f"{key}[{i}]")
            targets.append(float(v))

    assert len(targets) == 108

    z = np.zeros((108, P), dtype=np.float32)
    u = np.ones(108, dtype=np.float32)
    report = []

    # Pre-factorize the saturation value
    _, sat_z, _, _ = factorize_to_primes(saturation, max_exp=8)

    for idx, (name, target) in enumerate(zip(names, targets)):
        if abs(target) < 1e-10:
            # Zero weight: approximate with 2^zero_exp (small value)
            if zero_exp != 0:
                z[idx] = [zero_exp, 0, 0, 0, 0, 0]
                u[idx] = 1.0  # sign doesn't matter much for tiny values
                approx = 2.0 ** zero_exp
            else:
                z[idx] = 0
                u[idx] = 1.0
                approx = 1.0
            report.append((name, target, approx, list(z[idx].astype(int)), abs(approx)))
            continue

        # Gate saturation weights: replace 127→saturation, 254→2*saturation
        is_gate = not (name.startswith('W_out') or name.startswith('b_out'))
        if is_gate and abs(target) > 10:
            s = 1 if target > 0 else -1
            if abs(abs(target) - 127) < 1:
                z[idx] = sat_z
            elif abs(abs(target) - 254) < 1:
                z2 = list(sat_z)
                z2[0] += 1  # multiply by 2
                z[idx] = z2
            u[idx] = s
            approx = s * math.exp(sum(z[idx][r] * LOG_PRIMES[r] for r in range(P)))
            report.append((name, target, approx, list(z[idx].astype(int)), abs(approx - target)))
        else:
            # Output layer or small gate weight: find best approximation
            s, z_vec, approx, err = factorize_to_primes(target, max_exp=5)
            z[idx] = z_vec
            u[idx] = s
            report.append((name, target, approx, z_vec, err))

    return z, u, report


def main():
    print("=" * 70)
    print("Golden Network -> Prime-Exponent Verification")
    print("=" * 70)
    print(f"Primes (P={P}): {PRIMES}")

    # Build test set once
    test_inputs, test_targets = make_test_set(max_n=1500)

    configs = [
        (8,   0, "sat=8,  zeros=±1"),
        (8,  -3, "sat=8,  zeros=2^-3=0.125"),
        (8,  -5, "sat=8,  zeros=2^-5=0.031"),
        (8, -10, "sat=8,  zeros=2^-10=0.001"),
        (16, -5, "sat=16, zeros=2^-5=0.031"),
        (16,-10, "sat=16, zeros=2^-10=0.001"),
    ]
    for saturation, zero_exp, desc in configs:
        print(f"\n{'='*70}")
        print(f"Config: {desc}")
        print(f"{'='*70}")

        z, u, report = build_golden_prime_exp(saturation=saturation, zero_exp=zero_exp)

        # Show non-trivial approximations
        print(f"\nNon-zero weight encodings:")
        n_zero = sum(1 for _, t, _, _, _ in report if abs(t) < 1e-10)
        max_rel = 0
        for name, target, approx, z_vec, err in report:
            if abs(target) < 1e-10:
                continue
            rel = err / abs(target)
            max_rel = max(max_rel, rel)
            z_short = [int(v) for v in z_vec]
            # Only print non-zero z entries
            z_str = str(z_short) if any(v != 0 for v in z_short) else "[0]*6"
            print(f"  {name:<18} target={target:>10.4f}  "
                  f"approx={approx:>10.4f}  z={z_str:<30} "
                  f"err={rel*100:.1f}%")

        print(f"\nZero-target weights: {n_zero}/108 (approximated as +1)")
        print(f"Max relative error (non-zero): {max_rel*100:.2f}%")

        # Compute |H|
        z_int = np.round(z).astype(np.float32)
        exp_bits = float(np.sum(np.abs(z_int) * np.array(LOG_PRIMES)))
        h_bits = 108 + exp_bits
        print(f"|H| = {h_bits:.1f} bits (108 sign + {exp_bits:.1f} exponent)")

        # Create model and inject params
        model = PrimeExpLSTM(hidden_size=H, P=P, clamp_logmag=20.0)
        params = {'z_exponents': jnp.array(z), 'u_signs': jnp.array(u)}

        print(f"\nEvaluating (float, z as-is)...")
        float_result = evaluate_anbn_accuracy(
            model, params, test_inputs, test_targets,
        )
        print(f"  acc={float_result['mean_accuracy']:.4f}, "
              f"gen_n={float_result['gen_n']}, "
              f"n_perfect={float_result['n_perfect']}")

        # Discretized (round z to integers)
        disc_params = {'z_exponents': jnp.array(z_int), 'u_signs': jnp.array(u)}
        print(f"Evaluating (discrete, z rounded)...")
        disc_result = evaluate_anbn_accuracy(
            model, disc_params, test_inputs, test_targets,
        )
        print(f"  acc={disc_result['mean_accuracy']:.4f}, "
              f"gen_n={disc_result['gen_n']}, "
              f"n_perfect={disc_result['n_perfect']}")


if __name__ == "__main__":
    main()
