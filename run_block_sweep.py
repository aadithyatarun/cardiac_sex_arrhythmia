"""
Block-sweep simulation using Numba-JIT ORd model (~100x faster than scipy-only).

Generates data/results/block_sweep.json, ead_thresholds.json, rate_dependence.json.
"""

import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.parameters import MALE, FEMALE
from models.ord_model import initial_conditions
from simulations.fast_pacing import run_ss_fast
from simulations.pacing import (compute_apd, detect_ead, compute_triangulation,
                                 detect_repolarization_failure, detect_critical_apd)

os.makedirs("data/results", exist_ok=True)

CYCLE_LENGTHS   = [500.0, 1000.0, 2000.0]
BLOCK_FRACTIONS = np.array([
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95
])
N_WARMUP = 60   # beats to approach steady-state (very fast with numba)
N_SWEEP  = 40   # beats per block level from warm IC
N_SAVE   = 1

print("=== ORd Block Sweep (Numba-accelerated) ===")
print(f"Cycle lengths: {[int(c) for c in CYCLE_LENGTHS]} ms")
print(f"Block fractions ({len(BLOCK_FRACTIONS)}): {BLOCK_FRACTIONS}")
print(f"Warmup: {N_WARMUP} beats, Sweep: {N_SWEEP} beats/point\n", flush=True)

# Trigger numba JIT compilation before timing anything
print("Compiling Numba JIT (one-time, ~20s)...", flush=True)
t_compile = time.time()
_y0_tmp = initial_conditions()
_tmp = run_ss_fast(MALE, cl=1000.0, n_beats=1, n_save=1, y0=_y0_tmp)
print(f"  done in {time.time()-t_compile:.0f}s\n", flush=True)


def warmup(params, cl, label):
    t0 = time.time()
    print(f"  Warmup {label} CL={int(cl)} ms ... ", end="", flush=True)
    t, y = run_ss_fast(params, cl=cl, n_beats=N_WARMUP, n_save=1)
    elapsed = time.time() - t0
    vm = y[0]
    apd = compute_apd(t, vm, 0.90)
    print(f"done in {elapsed:.1f}s  APD90={apd:.1f} ms", flush=True)
    return y[:, -1]


block_results = {}
ead_thresholds = {}        # classic EAD oscillations (rarely triggered in ORd)
rep_failure_thresholds = {}  # APD90 > 95% CL or NaN
critical_apd_thresholds = {} # APD90 > 500 ms
rate_dep = {"male": {"cl": [], "apd90": [], "tri": []},
            "female": {"cl": [], "apd90": [], "tri": []}}

t_total = time.time()

for sex, params in [("male", MALE), ("female", FEMALE)]:
    block_results[sex] = {}
    ead_thresholds[sex] = {}
    rep_failure_thresholds[sex] = {}
    critical_apd_thresholds[sex] = {}

    for cl in CYCLE_LENGTHS:
        key = str(int(cl))
        y0_warm = warmup(params, cl, f"{sex}")

        block_results[sex][key] = {
            "block": [], "apd90": [], "apd50": [], "apd30": [],
            "tri": [], "ead": [], "rep_failure": [], "critical_apd": []
        }
        ead_thresholds[sex][key] = None
        rep_failure_thresholds[sex][key] = None
        critical_apd_thresholds[sex][key] = None

        for blk in BLOCK_FRACTIONS:
            t0 = time.time()
            try:
                t, y = run_ss_fast(params, cl=cl, n_beats=N_SWEEP, n_save=N_SAVE,
                                   ikr_block=blk, y0=y0_warm)
                vm = y[0, :]
                apd90 = compute_apd(t, vm, 0.90)
                apd50 = compute_apd(t, vm, 0.50)
                apd30 = compute_apd(t, vm, 0.30)
                tri   = compute_triangulation(t, vm)
                ead   = detect_ead(t, vm)
                rep_f = detect_repolarization_failure(t, vm, cl, failure_threshold=0.95)
                crit  = detect_critical_apd(t, vm, threshold_ms=500.0)

                block_results[sex][key]["block"].append(float(blk))
                block_results[sex][key]["apd90"].append(float(apd90) if not np.isnan(apd90) else None)
                block_results[sex][key]["apd50"].append(float(apd50) if not np.isnan(apd50) else None)
                block_results[sex][key]["apd30"].append(float(apd30) if not np.isnan(apd30) else None)
                block_results[sex][key]["tri"].append(float(tri)   if not np.isnan(tri)   else None)
                block_results[sex][key]["ead"].append(int(ead))
                block_results[sex][key]["rep_failure"].append(int(rep_f))
                block_results[sex][key]["critical_apd"].append(int(crit))

                if ead_thresholds[sex][key] is None and ead:
                    ead_thresholds[sex][key] = float(blk)
                if rep_failure_thresholds[sex][key] is None and rep_f:
                    rep_failure_thresholds[sex][key] = float(blk)
                if critical_apd_thresholds[sex][key] is None and crit:
                    critical_apd_thresholds[sex][key] = float(blk)

                # Save AP traces at key block levels
                if abs(blk) < 0.01 or abs(blk - 0.50) < 0.04 or abs(blk - 0.80) < 0.04:
                    np.savez_compressed(
                        f"data/results/ap_{sex}_cl{int(cl)}_blk{int(blk*100)}.npz",
                        t=t, vm=vm)

                elapsed = time.time() - t0
                risk = "FAIL" if rep_f else ("CRIT" if crit else "ok  ")
                print(f"  {sex} CL={int(cl)} blk={blk*100:4.0f}%: "
                      f"APD90={apd90:6.1f}ms  tri={tri:5.1f}ms  "
                      f"risk={risk}  ({elapsed:.2f}s)", flush=True)

            except Exception as e:
                print(f"  FAILED {sex} CL={int(cl)} blk={blk:.2f}: {e}", flush=True)
                block_results[sex][key]["block"].append(float(blk))
                for k in ("apd90","apd50","apd30","tri"):
                    block_results[sex][key][k].append(None)
                for k in ("ead","rep_failure","critical_apd"):
                    block_results[sex][key][k].append(0)

    # Rate-dependence at 0% block
    for cl in CYCLE_LENGTHS:
        key = str(int(cl))
        if block_results[sex][key]["apd90"]:
            rate_dep[sex]["cl"].append(int(cl))
            rate_dep[sex]["apd90"].append(block_results[sex][key]["apd90"][0])
            rate_dep[sex]["tri"].append(block_results[sex][key]["tri"][0])

# ── Save results ───────────────────────────────────────────────────────────────
with open("data/results/block_sweep.json", "w") as f:
    json.dump(block_results, f, indent=2)
with open("data/results/ead_thresholds.json", "w") as f:
    json.dump(ead_thresholds, f, indent=2)
with open("data/results/rep_failure_thresholds.json", "w") as f:
    json.dump(rep_failure_thresholds, f, indent=2)
with open("data/results/critical_apd_thresholds.json", "w") as f:
    json.dump(critical_apd_thresholds, f, indent=2)
with open("data/results/rate_dependence.json", "w") as f:
    json.dump(rate_dep, f, indent=2)

print(f"\n=== Block sweep complete in {(time.time()-t_total)/60:.1f} minutes ===")
print("\nRepolarization failure thresholds (APD90 > 95% CL or NaN):")
for sex in ("male", "female"):
    for cl in CYCLE_LENGTHS:
        th = rep_failure_thresholds[sex][str(int(cl))]
        s = f"{th*100:.0f}%" if th is not None else "not reached"
        print(f"  {sex} CL={int(cl)}ms: {s}")

print("\nCritical APD thresholds (APD90 > 500 ms):")
for sex in ("male", "female"):
    for cl in CYCLE_LENGTHS:
        th = critical_apd_thresholds[sex][str(int(cl))]
        s = f"{th*100:.0f}%" if th is not None else "not reached"
        print(f"  {sex} CL={int(cl)}ms: {s}")
