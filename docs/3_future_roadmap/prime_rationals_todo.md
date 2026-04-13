# Prime-Exponent Relaxation: Deferred Extensions

Tracking future improvements to the prime-exponent continuous relaxation for MDL-style rational regularization.

## ~~Sign Handling~~ — Done (2026-03-31)

Replaced `tanh(u)` with `sign_ste(u)` in `reconstruct_weight()`. Forward pass uses `sign(u) ∈ {-1, +1}`, backward passes gradient through via STE. `compute_mdl_penalty()` now includes 1 bit per weight for sign: `|H| = N * 1 bit + sum |z_{i,r}| * log(p_r)`. This ensures `u` encodes only sign (not continuous magnitude), forcing all magnitude through the prime-exponent structure `z`.

## Experiment Results (2026-03-31): sign_ste + Hyperparameter Sweep

After the `sign_ste` change, ran a full hyperparameter sweep to find good `lambda_mdl` values:
- **Adam** (`prime_rationals.py`): lambda=1.0 is optimal. lambda>=10 kills learning entirely.
- **SGD** (`prime_rationals_int.py`): lambda=0.1 is optimal. lambda>=1 hurts significantly.
- `init_std=0.01` remains best for both; larger values (0.5, 1.0) cause |H| explosion.

Configs updated with tuned values. Full 10k-epoch results (ANBN, H=3, P=6):

| Experiment | lambda | gen_n | disc_gen_n | \|H\| (bits) | test_acc | disc_acc |
|---|---|---|---|---|---|
| Adam baseline | 1.0 | 226 | 1 | 108 | 0.992 | 0.005 |
| QAT-B (int-attraction) | 1.0 | 55 | 0 | 108 | 0.756 | 0.000 |
| QAT-C (STE rounding) | 1.0 | 3 | 1 | 108 | 0.187 | 0.005 |
| QAT-D (combined) | 1.0 | 0 | 0 | 108 | 0.000 | 0.000 |
| Float SGD | 0.1 | 57 | 3 | 108 | 0.931 | 0.134 |
| Int8 | 0.1 | 6 | 0 | 337 | 0.122 | 0.000 |
| Int6 | 0.1 | 7 | 1 | 116 | 0.076 | 0.005 |

### Key findings

1. **|H| ≈ 108 bits (sign bits only) across most runs.** Exponents z stay near zero; the model encodes information through tiny continuous z perturbations around 0, not through integer exponents. Even with `sign_ste`, the MDL gradient on z dominates the data gradient: `d_w/d_z = w * log(p) ≈ 1 * log(p)` (since weights ≈ ±1), meaning even lambda=1.0 is enough to pin z near zero.

2. **Discretization destroys generalization.** Float gen_n up to 226, but disc_gen_n is 0-3. Rounding z to 0 removes the tiny perturbations the model depends on, collapsing all weights to exactly ±1.

3. **QAT hurts rather than helps** with the current parameterization. QAT-D (combined rounding + int-attraction) completely fails (gen_n=0). The rounding schedule interacts badly with near-zero z: `round(z) = 0` throughout training, so the model in "rounded" mode sees constant ±1 weights and cannot learn.

4. **Integer arithmetic (Int8) pushes z further from zero** (|H|=337, mean|z|=0.34) due to stochastic rounding noise, but this degrades rather than improves accuracy.

### Diagnosis: fundamental z-underuse problem

The root cause is not sign handling — it is that the loss landscape provides almost no gradient pressure to move z away from zero. With all weights ≈ ±1, the LSTM can learn the ANBN task through sign patterns alone, and the MDL penalty (even at lambda=1.0) is sufficient to keep z pinned near zero. The `sign_ste` fix was necessary (it removed the unpenalized `tanh(u)` channel) but not sufficient — further work is needed to make z usage attractive to the optimizer.

### Diagnosis: local minimum, not gradient magnitude

The sign_ste fix resolved the gradient attenuation problem (weights are now ±1 not ±0.01), but the real issue is a **local minimum**: the ±1 basin (z=0) is attractive for both training loss and MDL simultaneously. With p=0.3, training strings are mostly short (median n ≈ 3), and approximate ±1 counting suffices — there's little gradient pressure toward the golden-network weights (which include 1/2, requiring z=[-1,0,0,0,0,0]). Lambda annealing would remove the MDL barrier but may not create new gradient signal toward non-trivial z.

### Next steps (ordered)

**~~Step 1: Verify the golden network in prime-exponent form~~ — Done (2026-03-31)**

Encoded the Lan et al. (2024, "MDL Regularization of LSTM Language Models", ACL, Appendix B) golden network as prime exponents. Script: `verify_golden_prime_exp.py`. Key findings:

- **The parameterization CAN represent the golden network.** With saturation=16 (2^4, replacing 127 for gate saturation) and zero weights approximated as 2^{-5}=1/32: **gen_n=1500 (perfect), |H|=755 bits, float=discrete (no gap).**
- **Zero handling is critical.** The golden network has 82/108 zero weights. With z=0 (weight=±1), gen_n=0. Approximating zeros as 2^{-5} or 2^{-10} restores functionality.
- **Saturation=8 is insufficient** — tanh(8) is close to 1 but not close enough for the counting mechanism. Saturation=16 works.
- **|H|=755 bits** (vs Lan et al.'s 1137 bits with their universal integer code). The prime-exponent encoding is actually more compact, but 82 "zero" weights cost 82×5×log(2)=284 bits. Adding a zero gate (1 bit per weight to indicate zero, as in Lan et al.) would reduce this to 82 bits + 26×(1 bit sign + exponent cost) ≈ much less.

**Conclusion:** The problem is purely optimization — the target exists and is reachable. The optimizer never finds it because of the ±1 local minimum. This validates steps 2-3 and strongly motivates zero gates (from the Sparsity section above).

**Step 2: Lambda annealing (obvious but uncertain)**

Train with lambda=0 initially, ramp up. Removes one barrier but may not escape the ±1 basin if the task loss alone is already minimized there. Worth testing because it's cheap, but expectations should be moderate.

**Step 3: Warm-start from a standard LSTM (likely effective, less principled)**

Train a normal LSTM on ANBN, get real-valued weights, factorize each as `±prod(p_r^{e_r})` to get integer z, then fine-tune with QAT + MDL. Bypasses the local minimum entirely. Effective but not end-to-end differentiable MDL — more of a "train then compress" workflow. May be the right approach for demonstrating that the prime-exponent MDL framework produces good compressed models.

**Step 4: Rethink continuous vs. categorical**

The Gumbel-Softmax categorical approach (`differentiable_mdl.py`) doesn't have this problem — it optimizes directly over a discrete rational grid with no continuous-to-discrete gap. The prime-exponent relaxation was meant to be a more elegant continuous alternative, but continuous optimization fundamentally resists snapping to integers. If steps 1-3 don't resolve the z-underuse problem, effort may be better directed at the categorical approach which works by construction.

## Initialization

- **Xavier/He-informed initialization**: Instead of `N(0, 0.01)` for all exponent coordinates, compute a per-layer target weight scale (e.g., He init: `sqrt(2/fan_in)`) and set `z` values so that `exp(z^T log(primes))` matches that scale. May require a per-layer scalar scale factor `rho`.

## Sparsity & Zero Handling

- **Zero gates (`g_i`)**: Add a learnable gate parameter `q_i` per weight, with `g_i = sigmoid(q_i)`, so that `s_i = g_i * sigma_i * exp(a_i)`. Include a gate cost `(1 - g_i) * c_0` in the regularizer to allow exact zeros without infinite exponent penalty.

## Scale Factors

- **Per-layer scalar scale factor `rho`**: Factor each layer as `W = rho * W_tilde` where `rho` is a learned real scalar and `W_tilde` uses the prime-exponent structure. Lets the network set overall magnitude easily while MDL shapes the fine structure.

## Alternative Parameterizations

- **Positive/negative exponent split**: Replace single real exponent `z_{i,r}` with two nonneg variables: `z_{i,r} = u_{i,r} - v_{i,r}` where `u, v >= 0`. Regularizer becomes `lambda * sum(u + v) * log(p_r) + gamma * sum(u * v)`. Makes numerator/denominator mass explicit and penalizes simultaneous use at the same prime.

## Architecture

- **Layerwise prime basis**: Allow different layers to use different numbers of primes `P` or different prime sets. E.g., early layers might need only P=4 while later layers benefit from P=8.

## Nullspace Control

The task loss depends on exponent vector `z` only through the scalar `a = z^T log(p)`. This means the task term is blind to the `(P-1)`-dimensional nullspace orthogonal to `c = (log p_1, ..., log p_P)`. While the MDL regularizer does depend on individual coordinates, optimizer noise can drive unconstrained drift in nullspace directions, widening the gap between continuous and discretized exponents at deployment.

**Soft nullspace penalty:** Add a penalty on the component of `z` orthogonal to the log-primes vector:

```
z_parallel = (c^T z / ||c||^2) * c
z_perp = z - z_parallel
R_null(z) = eta * mean(z_perp^2)
```

Full objective becomes `L = L_task(s) + lambda * R_mdl(z) + eta * R_null(z)`.

**Implementation (PyTorch/JAX):**
1. Precompute `alpha = 1 / ||c||^2` where `c = log_primes`.
2. For each exponent vector `z` (last dim `P`): `coeff = (z * c).sum(-1, keepdim=True) * alpha`, then `z_parallel = coeff * c`, `z_perp = z - z_parallel`.
3. `null_penalty = eta * z_perp.pow(2).mean()`.

**Design notes:**
- This is a *soft* penalty, not a hard reparameterization. Collapsing to `z = beta * c` (one scalar DOF) is too restrictive — it removes the ability of different exponent coordinates to take different signs, which is needed for rationals with both numerator and denominator prime factors.
- Best used together with a separate integer-attraction mechanism (e.g., `sin^2(pi z)` penalty or rounded-forward QAT).
- Expected effects: reduced gratuitous exponent drift, more stable continuous exponent vectors, smaller train-to-discretize gap.
- **Tuning `eta`:** Start with a small sweep, e.g. `eta in {1e-4, 1e-3, 1e-2}`. The penalty should be subordinate to both the task loss and MDL regularizer — its job is cleanup, not shaping.

## ~~Quantization-Aware Training (QAT) with Rounded-Forward Exponents~~ — Done (2026-03-30)

Implemented in `prime_rationals.py`. Added `round_ste()` (STE via `jax.lax.stop_gradient`), `reconstruct_weight` with 3 modes (continuous/rounded/frozen_rounded), phased training schedule (`get_forward_mode`, `get_integer_mu`), exponent clamping (`clamp_exponents_in_params`), QAT diagnostics (`compute_qat_diagnostics`), and `make_anbn_qat_train_step`. All Flax modules (`PrimeExpLinear`, `PrimeExpMLP`, `PrimeExpLSTM`) accept `mode` parameter. 9 new config fields on `PrimeRationalConfig`. Fully backward compatible when `qat_enabled=False`. 4 ablation YAML configs: `config/prime_rationals/anbn_qat_{A,B,C,D}.yaml`. 43 tests in `tests/test_prime_rationals.py`.

## ~~Integer-Attraction Penalty~~ — Done (2026-03-30)

Implemented alongside QAT in `prime_rationals.py`. `integer_attraction_penalty(z)` = `mean(sin^2(pi*z))`, `integer_distance_penalty(z)` for ablation. Annealing schedule via `get_integer_mu(frac, mu_max, ...)`. Integrated into `make_anbn_qat_train_step` weighted by `mu`. Config fields: `int_attraction_mu_max`, `mu_start_frac`, `mu_full_frac`.

**Experiments run (2026-03-31):** 4-run ablation (A/B/C/D) completed with tuned lambda_mdl=1.0. QAT degraded performance — see "Experiment Results" section above. The QAT mechanism itself works (bug in JIT string-return fixed 2026-03-31), but the near-zero z regime makes rounding a no-op (`round(0.02) = 0`). QAT will need re-evaluation once the z-underuse problem is addressed.

## Training Schedule

- **Lambda annealing**: Start with low `lambda_mdl` to let the model learn the task, then increase to encourage simplification. Analogous to tau annealing in the Gumbel-Softmax approach.
