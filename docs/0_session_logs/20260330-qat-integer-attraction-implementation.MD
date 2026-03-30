# 2026-03-30: QAT and Integer-Attraction Penalty Implementation

## Summary

Implemented quantization-aware training (QAT) and integer-attraction penalty support for the prime-exponent MDL codebase in `prime_rationals.py`.

## Changes

### New utility functions (in `prime_rationals.py`):
- `round_ste(z)`: Straight-through estimator rounding (forward = round, backward = identity)
- `integer_attraction_penalty(z)`: sin^2(pi * z) penalty, zero at integers, smooth gradient
- `integer_distance_penalty(z)`: (z - round(z))^2 alternative for ablation
- `get_forward_mode(frac, ...)`: Schedule returning "continuous" / "rounded" / "frozen_rounded"
- `get_integer_mu(frac, mu_max, ...)`: Linear ramp schedule for penalty weight
- `compute_qat_diagnostics(params, P)`: Computes d_int, d_logw, f_eps, int_penalty across all z_* params
- `clamp_exponents_in_params(params, E_max)`: Clamps all z_* to [-E_max, E_max]
- `make_anbn_qat_train_step(...)`: JIT'd train step with mode, mu, and gradient clipping

### Modified existing code:
- `reconstruct_weight`: added `mode` parameter ("continuous"/"rounded"/"frozen_rounded")
- `PrimeExpLinear.__call__`: added `mode` parameter, passed through to `reconstruct_weight`
- `PrimeExpMLP.__call__`: added `mode` parameter, passed to child layers
- `PrimeExpLSTM.__call__`: added `mode` parameter, passed to `reconstruct_weight`
- `PrimeRationalConfig`: added 9 new QAT fields (qat_enabled, int_attraction_mu_max, etc.)
- `_build_arg_parser`: added corresponding CLI arguments
- `run_anbn_experiment`: QAT training loop with mode switching, LR drops, clamping, diagnostics

### New config files:
- `config/prime_rationals/anbn_qat_A.yaml`: baseline (qat_enabled: false)
- `config/prime_rationals/anbn_qat_B.yaml`: integer-attraction only (no rounding)
- `config/prime_rationals/anbn_qat_C.yaml`: STE rounding only (no attraction)
- `config/prime_rationals/anbn_qat_D.yaml`: combined (rounding + attraction)

### New tests:
- `tests/test_prime_rationals.py`: 43 tests covering all new functions, modes, schedules, diagnostics, and backward compatibility

## Design decisions
- QAT train steps are cached by (mode, mu_bucket) to avoid excessive JIT recompilation
- Gradient clipping uses manual global norm computation (not optax transform chain) for simplicity in the step function
- When `qat_enabled=False`, behavior is identical to original code (all new code paths are gated)
- LR drop occurs at mode boundaries by rebuilding the optimizer with dropped LR

## Test results
- All 43 new tests pass (`JAX_PLATFORMS=cpu`)
- All 36 existing `test_mdl.py` tests pass (backward compatibility confirmed)
