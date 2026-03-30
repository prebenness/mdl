# Issue: Restart-on-stagnation mechanism is a degenerate replay loop

## Observed behaviour

In the most recent `dst_fixed` run (`20260323_191905`, 12,000 epochs, `restart_patience=200`, `eval_every=50`):

- 56 restarts fired across the entire run
- Restarts #2–#27 (26 consecutive): all reverted to the same checkpoint (epoch 450)
- Restarts #28–#56 (29 consecutive): all reverted to epoch 6100
- Only 2 actual improvements across all 56 restarts (|H| 721 → 709, a 12-bit gain)
- ~93% of the training budget was spent on near-identical replays from the same checkpoint

The mechanism is supposed to prevent stagnation. Instead, it enforces a Sisyphean loop: restore best checkpoint, train for `patience` epochs, fail to improve, restore the same checkpoint again.

## Why it is useless

The restart (`_maybe_restart` in `differentiable_mdl.py:203`) restores `best['params']` and zeros the Adam moments. In `dst_fixed` mode, this produces near-deterministic replays:

1. **Same starting point every time.** The best params are restored verbatim. No perturbation, no noise, no exploration.

2. **Near-deterministic training dynamics.** With `deterministic_st=True`, the forward pass is pure argmax (no Gumbel noise). The only stochasticity is mini-batch ordering from the advancing RNG, which provides insufficient variation to escape the attractor basin. In a full-batch regime (`bs >= N`), restarts would be literally identical — zero stochasticity.

3. **Adam count preservation with zeroed moments.** `_zero_adam_moments` zeros mu/nu but preserves the optimizer step count. The bias correction denominators `(1 - beta_2^count)` expect accumulated statistics but receive fresh zeros, producing artificially large initial updates (~2.7x effective LR) that likely destabilise the model immediately after restart. This may explain the frequent 50/50 → 0/50 collapses visible in the logs.

4. **No escalation.** After 27 failed restarts to the same checkpoint, the mechanism does nothing differently on the 28th attempt.

## Affected code paths

- `differentiable_mdl.py:203-214` — `_maybe_restart()`
- `differentiable_mdl.py:193-200` — `_zero_adam_moments()`
- `differentiable_mdl.py:860-870` — `run_training_dst_fixed()` fused path restart integration
- `differentiable_mdl.py:897-907` — `run_training_dst_fixed()` mini-batch path restart integration
- `prime_rationals.py:829-841` — duplicate `_maybe_restart()` with same issue

## Scope

This issue applies to all training modes that use `_maybe_restart`, not just `dst_fixed`. However, the problem is most acute under `dst_fixed` because `deterministic_st=True` eliminates the Gumbel noise that would otherwise provide some trajectory variation between restarts.
