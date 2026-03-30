#!/usr/bin/env python3.12
"""Quick hyperparameter sweep for prime-exponent experiments after sign_ste change.

Runs short (2000-epoch) experiments varying lambda_mdl and init_std,
collects gen_n and |H| from stdout, and prints a summary table.
"""
import subprocess
import sys
import re
from itertools import product

SWEEP_EPOCHS = 2000
EVAL_EVERY = 500
LOG_EVERY = 500

def run_experiment(script, base_config, overrides, label):
    """Run one experiment and extract key metrics from stdout."""
    cmd = [
        "python3.12", script, base_config,
        "--epochs", str(SWEEP_EPOCHS),
        "--eval_every", str(EVAL_EVERY),
        "--log_every", str(LOG_EVERY),
    ]
    for k, v in overrides.items():
        cmd.extend([f"--{k}", str(v)])

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr

    # Extract best gen_n and final |H| from eval lines
    # [eval] ... gen_n(int/float/disc)=X/Y/Z or gen_n=X ... |H|=N.NN
    best_gen_n = 0
    last_h = 0.0
    best_disc_acc = 0.0
    last_nll = 999.0

    for line in output.split("\n"):
        # prime_rationals.py format: gen_n=X ... |H|=N.NN
        m = re.search(r'gen_n=(\d+)', line)
        if m:
            gn = int(m.group(1))
            best_gen_n = max(best_gen_n, gn)

        # prime_rationals_int.py format: gen_n(int/float/disc)=X/Y/Z
        m = re.search(r'gen_n\(int/float/disc\)=(\d+)/(\d+)/(\d+)', line)
        if m:
            for g in m.groups():
                best_gen_n = max(best_gen_n, int(g))

        m = re.search(r'\|H\|=([0-9.]+)', line)
        if m:
            last_h = float(m.group(1))

        m = re.search(r'disc_acc=([0-9.]+)', line)
        if m:
            best_disc_acc = max(best_disc_acc, float(m.group(1)))

        # float acc for prime_rationals.py
        m = re.search(r'acc=([0-9.]+)', line)
        if m:
            best_disc_acc = max(best_disc_acc, float(m.group(1)))

        m = re.search(r'nll=([0-9.]+)', line)
        if m:
            last_nll = float(m.group(1))

    # Print last few lines for context
    lines = [l for l in output.strip().split("\n") if l.strip()]
    for l in lines[-8:]:
        print(f"  {l}")

    return {
        "label": label,
        "best_gen_n": best_gen_n,
        "last_h": last_h,
        "best_disc_acc": best_disc_acc,
        "last_nll": last_nll,
    }


def main():
    results = []

    # --- Phase 1: prime_rationals.py (Adam) lambda_mdl sweep ---
    print("\n" + "#"*60)
    print("# Phase 1: prime_rationals.py — lambda_mdl sweep")
    print("#"*60)

    for lam in [1.0, 10.0, 50.0, 100.0]:
        r = run_experiment(
            "prime_rationals.py",
            "config/prime_rationals/anbn.yaml",
            {"lambda_mdl": lam},
            f"prime_rationals Adam | lambda={lam} init_std=0.01",
        )
        results.append(r)

    # --- Phase 2: prime_rationals_int.py float SGD lambda_mdl sweep ---
    print("\n" + "#"*60)
    print("# Phase 2: prime_rationals_int.py float SGD — lambda_mdl sweep")
    print("#"*60)

    for lam in [0.1, 1.0, 10.0, 100.0]:
        r = run_experiment(
            "prime_rationals_int.py",
            "config/prime_rationals_int/anbn_float_sgd.yaml",
            {"lambda_mdl": lam},
            f"prime_rationals_int float SGD | lambda={lam} init_std=0.01",
        )
        results.append(r)

    # --- Phase 3: init_std sweep with best lambdas ---
    print("\n" + "#"*60)
    print("# Phase 3: init_std sweep (best lambda from phases 1-2)")
    print("#"*60)

    # Pick best lambda for each script (by gen_n, then by lowest |H|)
    p1 = [r for r in results[:4]]
    p2 = [r for r in results[4:8]]

    best_lam_adam = max(p1, key=lambda r: (r["best_gen_n"], -r["last_h"]))
    best_lam_sgd = max(p2, key=lambda r: (r["best_gen_n"], -r["last_h"]))

    lam_adam = float(best_lam_adam["label"].split("lambda=")[1].split()[0])
    lam_sgd = float(best_lam_sgd["label"].split("lambda=")[1].split()[0])

    print(f"\nBest lambda (Adam): {lam_adam} (gen_n={best_lam_adam['best_gen_n']}, |H|={best_lam_adam['last_h']:.1f})")
    print(f"Best lambda (SGD):  {lam_sgd} (gen_n={best_lam_sgd['best_gen_n']}, |H|={best_lam_sgd['last_h']:.1f})")

    for std in [0.1, 0.5, 1.0]:
        r = run_experiment(
            "prime_rationals.py",
            "config/prime_rationals/anbn.yaml",
            {"lambda_mdl": lam_adam, "init_std": std},
            f"prime_rationals Adam | lambda={lam_adam} init_std={std}",
        )
        results.append(r)

    for std in [0.1, 0.5, 1.0]:
        r = run_experiment(
            "prime_rationals_int.py",
            "config/prime_rationals_int/anbn_float_sgd.yaml",
            {"lambda_mdl": lam_sgd, "init_std": std},
            f"prime_rationals_int float SGD | lambda={lam_sgd} init_std={std}",
        )
        results.append(r)

    # --- Summary ---
    print("\n" + "="*80)
    print("SWEEP SUMMARY")
    print("="*80)
    print(f"{'Label':<55} {'gen_n':>6} {'|H|':>8} {'nll':>8}")
    print("-"*80)
    for r in results:
        print(f"{r['label']:<55} {r['best_gen_n']:>6} {r['last_h']:>8.2f} {r['last_nll']:>8.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
