# Prime-Exponent Relaxation: Deferred Extensions

Tracking future improvements to the prime-exponent continuous relaxation for MDL-style rational regularization.

## Sign Handling

- **Hard sign with STE**: Replace `tanh(u_i)` with binary `sign(u_i)` in the forward pass, using a straight-through estimator for gradients. This would make the sign discrete during training, which may help convergence to clean rational weights. The current smooth `tanh` sign was chosen for simplicity in v1.

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

**Still pending — experiments:** Run the 4-run ablation matrix (A/B/C/D) and tune hyperparameters (`mu_max`, `round_warmup_frac`, `lr_drop_at_switch`). See `config/prime_rationals/anbn_qat_*.yaml`.

## Training Schedule

- **Lambda annealing**: Start with low `lambda_mdl` to let the model learn the task, then increase to encourage simplification. Analogous to tau annealing in the Gumbel-Softmax approach.
