# 2026-03-31: sign_ste fix, hyperparameter sweep, experiment battery

## Changes

1. **Replaced `tanh(u)` with `sign_ste(u)`** in `prime_rationals.py`:
   - New `sign_ste()` function: forward=sign(u), backward=identity (STE)
   - `reconstruct_weight()` now uses `sign_ste(u)` — u encodes only sign, not magnitude
   - `compute_mdl_penalty()` includes 1 bit per weight for sign: `N + sum |z| * log(p)`
   - `compute_h_bits()` updated accordingly

2. **Added argparse CLI** to `prime_rationals_int.py` (matching `prime_rationals.py` pattern). Supports `--lambda_mdl`, `--lr`, etc. as overrides on top of YAML config.

3. **Fixed QAT JIT bug**: `make_anbn_qat_train_step` returned `"mode": mode` (a string) in the aux dict from a JIT'd function. JAX cannot trace strings. Removed the string from the return dict.

4. **Tuned lambda_mdl** via 2000-epoch sweep:
   - Adam (`prime_rationals.py`): 100.0 -> 1.0
   - SGD/int (`prime_rationals_int.py`): 100.0 -> 0.1
   - All YAML configs updated

5. **Full experiment battery** (10k epochs, ANBN H=3 P=6):
   - Adam baseline: gen_n=226, disc_gen_n=1
   - QAT A-D: all worse than baseline (D completely fails)
   - Float SGD: gen_n=57, disc_gen_n=3
   - Int8: gen_n=6, Int6: gen_n=7

## Diagnosis

The z-underuse problem persists: even with sign_ste, exponents stay near zero (|H| = 108 = sign bits only). The model learns through tiny continuous z perturbations which break under discretization. QAT cannot help when z is already near zero (rounding is a no-op).

Root cause: the loss landscape provides insufficient gradient pressure on z. With weights ≈ ±1, the LSTM can solve ANBN through sign patterns alone, and even lambda=1.0 MDL penalty is enough to pin z at zero.

## Next steps (documented in `docs/3_future_roadmap/prime_rationals_todo.md`)

1. Lambda annealing / warm-up (delay MDL penalty)
2. Per-layer scale factor rho
3. Xavier-informed z initialization
4. Nullspace control
