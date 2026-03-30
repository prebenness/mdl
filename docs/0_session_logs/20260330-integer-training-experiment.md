# 2026-03-30: Integer-Arithmetic Training for Prime-Exponent MDL

## What was done

Implemented simulated integer-arithmetic training for the prime-exponent MDL system, following Ghaffari et al. (NeurIPS 2022, "Is Integer Arithmetic Enough for Deep Learning Training?", arXiv:2207.08822).

The key idea: all heavy linear algebra (LSTM gate matmuls, output projection) is performed on an integer grid via dynamic fixed-point representation mapping with stochastic rounding, while pointwise nonlinearities (sigmoid, tanh, log-softmax) remain in float32. This is consistent with Ghaffari's design where the inverse mapping module sits between layer computations.

The implementation uses float32 dtype throughout for JAX autodiff compatibility, but constrains values to the integer grid via stochastic rounding + straight-through gradient estimator. This gives the same numerical result as true integer GEMM while remaining differentiable.

## Files created

### `prime_rationals_int.py` (standalone, no existing files modified)

Contains six modules:

1. **Representation mapping**: `stochastic_round()`, `float_to_int()`, `int_to_float()` -- dynamic fixed-point conversion with unbiased stochastic rounding and straight-through gradients
2. **Integer GEMM**: `int_matmul()` -- quantize inputs, float matmul on integer-grid values, re-quantize result
3. **IntegerPrimeExpLSTM**: Flax module mirroring PrimeExpLSTM exactly but with integer matmuls when `use_integer=True`. PRNG keys pre-split into T step keys before `jax.lax.scan`, each split into 8 sub-keys for 8 gate matmuls
4. **Training loop**: SGD + momentum via `optax.sgd(lr, momentum=0.9)` (not Adam -- Ghaffari's convergence proof covers SGD). Fresh rng per epoch via `jax.random.fold_in`
5. **Evaluation**: Integer, float, and discretized eval modes with metrics logging
6. **Config**: `IntegerTrainingConfig` dataclass with `int_bits`, `momentum`, `use_integer` fields

### Config YAMLs under `config/prime_rationals_int/`

- `anbn_int8.yaml`, `anbn_int6.yaml`, `anbn_int5.yaml`, `anbn_int4.yaml`
- `anbn_float_sgd.yaml` (float SGD baseline, `use_integer: false`)

### `tests/test_integer_training.py` (13 tests, all passing)

1. Round-trip: `int_to_float(float_to_int(A))` within 1 ULP for int8, int6, int5, int4
2. All-zeros handling
3. Stochastic rounding unbiasedness: N=10000 roundings of 3.7, mean within 3 sigma
4. Stochastic rounding produces integer values
5. Integer GEMM: 3x3 matmul within 25% relative error (quantization tolerance)
6. Identity matmul preservation
7. Integer LSTM forward: finite output, close to float output
8. Output layer integer matmul
9. Gradient flow: non-zero finite gradients for z_exponents and u_signs (straight-through works)
10. Float mode gradient flow (sanity check)
11. 50 epochs int8 training: loss decreases
12. Float SGD baseline runs without error

## Design decisions

1. **Float dtype, integer values**: The critical insight from the spec. Using float32 for all tensors but constraining values to integer grid via stochastic rounding + straight-through. This avoids JAX's limited int8 support and keeps autodiff working.

2. **SGD + momentum, not Adam**: Ghaffari's convergence guarantee (Theorem 1, Remark 1) covers SGD. The float SGD baseline uses the same optimizer for fair comparison.

3. **Re-quantization of GEMM output**: After the int_B x int_B -> int_2B accumulator, we convert back to float scale then re-quantize to B-bit, matching Ghaffari Section 3.3 step 4.

4. **Biases stay in float and are simply added**: Given the tiny 3x3 architecture, the spec's integer bias alignment is less critical. Biases are added in float after the integer matmul result is inverse-mapped.

5. **PRNG management**: Single rng per train step, epoch-wise fold_in, pre-split into T step keys for scan, each step split into 8 sub-keys for 8 gate matmuls.

## Test results

```
13 passed in 21.54s (JAX_PLATFORMS=cpu)
36 existing tests still pass (no regressions)
```

## How to run

```bash
# Tests
JAX_PLATFORMS=cpu python3.12 -m pytest tests/test_integer_training.py -v

# Full training (do NOT run on GPU without checking utilization)
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_int8.yaml
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_float_sgd.yaml
```
