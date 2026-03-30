#!/bin/bash
# Full experiment battery with tuned hyperparameters.
# Runs sequentially to stay under 40% GPU usage.
set -e

# Limit JAX GPU memory to 35% to stay well under 40% total
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35

echo "============================================================"
echo "FULL EXPERIMENT BATTERY — $(date)"
echo "GPU memory limit: ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo "============================================================"

# 1. Baseline Adam (no QAT)
echo -e "\n>>> [1/8] Baseline Adam (anbn.yaml)"
python3.12 prime_rationals.py config/prime_rationals/anbn.yaml 2>&1 | tail -20

# 2. QAT A: baseline (no QAT, same as above but explicit)
echo -e "\n>>> [2/8] QAT-A: No QAT baseline"
python3.12 prime_rationals.py config/prime_rationals/anbn_qat_A.yaml 2>&1 | tail -20

# 3. QAT B: integer-attraction only
echo -e "\n>>> [3/8] QAT-B: Integer-attraction only"
python3.12 prime_rationals.py config/prime_rationals/anbn_qat_B.yaml 2>&1 | tail -20

# 4. QAT C: STE rounding only
echo -e "\n>>> [4/8] QAT-C: STE rounding only"
python3.12 prime_rationals.py config/prime_rationals/anbn_qat_C.yaml 2>&1 | tail -20

# 5. QAT D: combined
echo -e "\n>>> [5/8] QAT-D: Combined"
python3.12 prime_rationals.py config/prime_rationals/anbn_qat_D.yaml 2>&1 | tail -20

# 6. Float SGD baseline
echo -e "\n>>> [6/8] Float SGD baseline"
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_float_sgd.yaml 2>&1 | tail -20

# 7. Int8
echo -e "\n>>> [7/8] Int8"
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_int8.yaml 2>&1 | tail -20

# 8. Int6
echo -e "\n>>> [8/8] Int6"
python3.12 prime_rationals_int.py config/prime_rationals_int/anbn_int6.yaml 2>&1 | tail -20

echo -e "\n============================================================"
echo "ALL EXPERIMENTS COMPLETE — $(date)"
echo "============================================================"

# Collect summary from results dirs
echo -e "\nSUMMARY TABLE:"
echo "--------------------------------------------------------------"
printf "%-40s %8s %8s %8s\n" "Run" "gen_n" "disc_gn" "|H|"
echo "--------------------------------------------------------------"

for dir in results/prime_rationals_anbn/P6_lam1.0_lr0.01 \
           results/prime_rationals_anbn/P6_lam1.0_lr0.01_qat_A \
           results/prime_rationals_anbn/P6_lam1.0_lr0.01_qat_B \
           results/prime_rationals_anbn/P6_lam1.0_lr0.01_qat_C \
           results/prime_rationals_anbn/P6_lam1.0_lr0.01_qat_D \
           results/prime_rationals_int/float_sgd_P6_lam0.1_lr0.01 \
           results/prime_rationals_int/int8_P6_lam0.1_lr0.01 \
           results/prime_rationals_int/int6_P6_lam0.1_lr0.01; do
    if [ -f "$dir/results.json" ]; then
        python3.12 -c "
import json, sys
d = json.load(open('$dir/results.json'))
name = '$dir'.split('/')[-1]
gn = d.get('test_gen_n', d.get('float_gen_n', '?'))
dgn = d.get('disc_test_gen_n', d.get('disc_gen_n', '?'))
h = d.get('h_bits', d.get('disc_h_bits', '?'))
print(f'{name:<40} {gn:>8} {dgn:>8} {h:>8.1f}')
" 2>/dev/null || echo "$dir: parse error"
    fi
done
echo "--------------------------------------------------------------"
