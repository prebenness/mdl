# Differentiable MDL for Rational-Weight Networks

Gradient-based training of neural networks with Minimum Description Length (MDL) regularization, biasing weights toward simple rational numbers.

Three complementary approaches to the same goal:

| Entry point | Approach | Weight parameterization |
|---|---|---|
| `differentiable_mdl.py` | Gumbel-Softmax categorical relaxation | Categorical distribution over finite rational grid |
| `prime_rationals.py` | Continuous prime-exponent relaxation | `s = tanh(u) * exp(z^T log(primes))` with L1-weighted exponent penalty |
| `prime_rationals_int.py` | Integer-arithmetic training (Ghaffari et al. 2022) | Same as above, but linear algebra simulated in integer fixed-point |

All three target the **a^n b^n** formal language recognition task (LSTM, hidden_size=3) from Lan et al. (2024, ACL).

## Quick start

```bash
# Install dependencies (JAX, Flax, optax, PyYAML)
pip install jax jaxlib flax optax pyyaml

# Run the categorical MDL experiment
python3.12 differentiable_mdl.py config/anbn_mdl/basic_train.yaml

# Run the prime-exponent experiment
python3.12 prime_rationals.py config/prime_rationals/anbn.yaml

# Run tests
pytest tests/ -v
```

## Experiments

### Categorical MDL (`differentiable_mdl.py`)

Gumbel-Softmax straight-through estimation over a categorical weight parameterization. Each weight is a distribution over a finite rational grid; the coding term (expected codelength) is computed exactly.

```bash
python3.12 differentiable_mdl.py config/anbn_mdl/basic_train.yaml
python3.12 differentiable_mdl.py config/anbn_mdl/basic_train.yaml --mode shared  # shared-weight extension
```

### Prime-exponent relaxation (`prime_rationals.py`)

Each weight is `s_i = sign(u_i) * exp(z_i^T log(primes))` with trainable real exponent coordinates `z` and sign parameters `u`. The sign is discretized via STE so `u` encodes only +/-1 (1 bit per weight); all magnitude must come through `z`. The MDL regularizer is `N * 1 bit (signs) + sum |z_{i,r}| * log(p_r)`. After training, exponents are rounded to integers to recover exact rationals.

```bash
# ANBN language task
python3.12 prime_rationals.py config/prime_rationals/anbn.yaml

# XOR toy classification
python3.12 prime_rationals.py config/prime_rationals/xor.yaml
```

#### QAT + Integer-Attraction Ablation

Quantization-aware training (rounded-forward exponents via STE) and integer-attraction penalty (`sin^2(pi*z)`) to close the gap between continuous training and discretized deployment. Four ablation configs isolate each component:

```bash
# A: Baseline — continuous forward, MDL only, post-hoc rounding
python3.12 prime_rationals.py config/prime_rationals/anbn_qat_A.yaml

# B: Integer-attraction only — continuous forward + sin^2 penalty
python3.12 prime_rationals.py config/prime_rationals/anbn_qat_B.yaml

# C: QAT only — rounded-forward after warmup, no integer-attraction
python3.12 prime_rationals.py config/prime_rationals/anbn_qat_C.yaml

# D: Combined — rounded-forward + integer-attraction (expected best)
python3.12 prime_rationals.py config/prime_rationals/anbn_qat_D.yaml
```

### Integer-arithmetic training (`prime_rationals_int.py`)

Simulated integer-only linear algebra for the prime-exponent LSTM, following Ghaffari et al. (NeurIPS 2022, "Is Integer Arithmetic Enough for Deep Learning Training?", arXiv:2207.08822). All LSTM matmuls use dynamic fixed-point with stochastic rounding; nonlinearities remain in float. Tests the hypothesis that MDL regularization (which pushes weights toward simple rationals) improves robustness to low-bit-width quantization.

```bash
# Float SGD baseline (isolates optimizer effect from integer quantization)
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_float_sgd.yaml

# Integer bit-width ablation
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_int8.yaml
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_int6.yaml
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_int5.yaml
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_int4.yaml
```

## Project structure

```
differentiable_mdl.py          # Categorical MDL (Gumbel-Softmax)
prime_rationals.py             # Prime-exponent relaxation + QAT
prime_rationals_int.py         # Integer-arithmetic training (Ghaffari)
src/
  mdl/
    data.py                    # ANBN data pipeline
    evaluation.py              # Metrics: NLL, grammar-weighted eval, recognition accuracy
  utils/
    checkpointing.py           # Run management, logging, checkpoints
config/
  anbn_mdl/                    # Configs for differentiable_mdl.py
  prime_rationals/             # Configs for prime_rationals.py (incl. QAT ablation)
  prime_rationals_int/         # Configs for prime_rationals_int.py (bit-width ablation)
tests/                         # pytest suite
docs/                          # Session logs, roadmap, active discussions
```

## References

- Lan et al. (2024), "MDL Regularization of LSTM Language Models for Formal Language Learning", ACL 2024
- Abudy et al. (2025), "Learning Formal Languages with Small RNNs Using MDL Regularization", NeurIPS 2025 submission, arXiv:2505.13398
- Lan, Chemla & Katzir (2022), "Minimum description length recurrent neural networks", TACL 10:785-799, arXiv:2111.00600
- Ghaffari et al. (2022), "Is Integer Arithmetic Enough for Deep Learning Training?", NeurIPS 2022, arXiv:2207.08822
