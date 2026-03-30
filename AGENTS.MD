# AGENTS.md

This file provides operational guidance for tool-driven work in this repository.

## Always Do

1. Start by reading the relevant documents in `docs/`.
2. At the end of every major piece of work, add a timestamped file to `docs/0_session_logs/`.

## Documentation Entry Point

- `docs/README.MD`

## Core Commands

Run tests:

```bash
pytest tests/test_mdl.py -v
pytest tests/test_prime_rationals.py -v
pytest tests/test_integer_training.py -v
```

Run ANBN MDL:

```bash
python3.12 differentiable_mdl.py config/anbn_mdl/basic_train.yaml
```

Run full experiment battery:

```bash
bash run_experiments.sh
```

### Prime-Exponent QAT + Integer-Attraction Experiments

Quantization-aware training (rounded-forward exponents via STE) and integer-attraction penalty (`sin^2(pi*z)`) for the prime-exponent relaxation. 4-run ablation to isolate the effect of each component. Spec: `agent-context/integer-exp (1).tex`.

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

### Integer-Arithmetic Training (Ghaffari et al. 2022)

Simulated integer-only linear algebra for the prime-exponent LSTM, following Ghaffari et al. (NeurIPS 2022, "Is Integer Arithmetic Enough for Deep Learning Training?", arXiv:2207.08822). Ablates across int8/int6/int5/int4 bit-widths. Spec: `agent-context/integer_training_plan.tex`.

```bash
# Float SGD baseline (isolates optimizer effect from integer quantization)
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_float_sgd.yaml

# Integer bit-width ablation
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_int8.yaml
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_int6.yaml
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_int5.yaml
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_int4.yaml
```

## Important Note

Keep root-level docs minimal. Stable reference material, active technical status, session logs, and planning now belong in `docs/`.
