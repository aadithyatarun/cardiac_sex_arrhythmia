"""
Master simulation script.

Generates all simulation data for:
  1. Baseline validation (AP morphology vs. pacing rate)
  2. APD rate-dependence: male vs. female at CL = 500, 1000, 2000 ms
  3. IKr block sweep: APD90 vs. block% at 3 rates
  4. EAD detection: minimum block% to trigger EAD by sex and rate
  5. Ionic current decomposition at critical block

Saves results to data/results/ as .npz files.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
from models.parameters import MALE, FEMALE
from simulations.pacing import run_steady_state, compute_apd, detect_ead, compute_triangulation

os.makedirs("data/results", exist_ok=True)

CYCLE_LENGTHS   = [500.0, 1000.0, 2000.0]
BLOCK_FRACTIONS = np.linspace(0.0, 0.95, 15)
N_BEATS_SS      = 80
N_BEATS_SAVE    = 1

# ── Warm-start cache: pre-run 20 beats at each CL/sex to skip cold start ─────
_warm_cache = {}   # key: (sex, cl) → y0_warm after 20 beats

def get_warm_y0(params, cl, sex_label):
    key = (sex_label, cl)
    if key not in _warm_cache:
        print(f"  [warm-start] {sex_label} CL={cl:.0f} ...", flush=True)
        t, y = run_steady_state(params, cl=cl, n_beats=20, n_save=1,
                                 ikr_block=0.0)
        _warm_cache[key] = y[:, -1]
        print(f"  [warm-start] done", flush=True)
    return _warm_cache[key]


def sim_one(params, cl, ikr_block, n_beats=N_BEATS_SS, label="", sex_label=""):
    """Run one pacing scenario and return metrics."""
    try:
        # Use warm-started IC if available (skips slow cold-start beat)
        y0 = get_warm_y0(params, cl, sex_label) if sex_label else None
        remain = max(1, n_beats - 20) if sex_label else n_beats
        t, y = run_steady_state(params, cl=cl, n_beats=remain,
                                n_save=N_BEATS_SAVE, ikr_block=ikr_block,
                                y0=y0)
        vm = y[0, :]
        apd90 = compute_apd(t, vm, threshold_frac=0.90)
        apd50 = compute_apd(t, vm, threshold_frac=0.50)
        apd30 = compute_apd(t, vm, threshold_frac=0.30)
        tri   = compute_triangulation(t, vm)
        ead   = detect_ead(t, vm)
        vmax  = float(np.max(vm))
        vrest = float(vm[0])
        print(f"  {label} CL={cl:.0f} block={ikr_block:.2f}: "
              f"APD90={apd90:.1f} EAD={ead}")
        return dict(apd90=apd90, apd50=apd50, apd30=apd30,
                    tri=tri, ead=int(ead), vmax=vmax, vrest=vrest,
                    t=t, vm=vm)
    except Exception as e:
        print(f"  FAILED {label} CL={cl} block={ikr_block}: {e}")
        return None


# ── 1. Baseline AP traces (male and female, 3 rates) ─────────────────────────
print("=== 1. Baseline AP traces ===")
baseline_results = {}
for sex, params in [("male", MALE), ("female", FEMALE)]:
    baseline_results[sex] = {}
    for cl in CYCLE_LENGTHS:
        r = sim_one(params, cl=cl, ikr_block=0.0,
                    label=f"{sex} baseline", sex_label=sex)
        if r is not None:
            baseline_results[sex][str(int(cl))] = {
                "apd90": r["apd90"], "apd50": r["apd50"], "apd30": r["apd30"],
                "tri": r["tri"], "vmax": r["vmax"], "vrest": r["vrest"]
            }
            np.savez_compressed(f"data/results/ap_baseline_{sex}_cl{int(cl)}.npz",
                                t=r["t"], vm=r["vm"])

with open("data/results/baseline_metrics.json", "w") as f:
    json.dump(baseline_results, f, indent=2)
print("  Saved baseline metrics.")


# ── 2. IKr block sweep: APD90 and EAD by sex and rate ─────────────────────────
print("\n=== 2. IKr block sweep ===")
block_results = {}
for sex, params in [("male", MALE), ("female", FEMALE)]:
    block_results[sex] = {}
    for cl in CYCLE_LENGTHS:
        key = str(int(cl))
        block_results[sex][key] = {"block": [], "apd90": [], "apd50": [],
                                   "tri": [], "ead": []}
        for blk in BLOCK_FRACTIONS:
            r = sim_one(params, cl=cl, ikr_block=blk,
                        label=f"{sex}", sex_label=sex)
            if r is not None:
                block_results[sex][key]["block"].append(float(blk))
                block_results[sex][key]["apd90"].append(
                    float(r["apd90"]) if not np.isnan(r["apd90"]) else None)
                block_results[sex][key]["apd50"].append(
                    float(r["apd50"]) if not np.isnan(r["apd50"]) else None)
                block_results[sex][key]["tri"].append(
                    float(r["tri"]) if not np.isnan(r["tri"]) else None)
                block_results[sex][key]["ead"].append(r["ead"])

                # Save AP trace at key block levels (0%, 50%, 80%)
                if abs(blk) < 0.01 or abs(blk - 0.50) < 0.03 or abs(blk - 0.80) < 0.03:
                    np.savez_compressed(
                        f"data/results/ap_{sex}_cl{int(cl)}_blk{int(blk*100)}.npz",
                        t=r["t"], vm=r["vm"])

with open("data/results/block_sweep.json", "w") as f:
    json.dump(block_results, f, indent=2)
print("  Saved block sweep results.")


# ── 3. EAD threshold: minimum block% to trigger EAD ──────────────────────────
print("\n=== 3. EAD thresholds ===")
ead_thresholds = {}
for sex, params in [("male", MALE), ("female", FEMALE)]:
    ead_thresholds[sex] = {}
    for cl in CYCLE_LENGTHS:
        key = str(int(cl))
        data = block_results[sex][key]
        thresh = None
        for blk, ead in zip(data["block"], data["ead"]):
            if ead == 1:
                thresh = blk
                break
        ead_thresholds[sex][key] = thresh
        print(f"  {sex} CL={cl}: EAD threshold = {thresh}")

with open("data/results/ead_thresholds.json", "w") as f:
    json.dump(ead_thresholds, f, indent=2)


# ── 4. Rate-dependence summary (APD90 vs CL at 0% block) ─────────────────────
print("\n=== 4. Rate-dependence (APD90 vs CL) ===")
cls_fine = [500, 700, 1000, 1200, 1500, 2000]
rate_dep = {}
for sex, params in [("male", MALE), ("female", FEMALE)]:
    rate_dep[sex] = {"cl": [], "apd90": [], "tri": []}
    for cl in cls_fine:
        r = sim_one(params, cl=float(cl), ikr_block=0.0, label=f"{sex} rate-dep", sex_label=sex)
        if r is not None:
            rate_dep[sex]["cl"].append(cl)
            rate_dep[sex]["apd90"].append(
                float(r["apd90"]) if not np.isnan(r["apd90"]) else None)
            rate_dep[sex]["tri"].append(
                float(r["tri"]) if not np.isnan(r["tri"]) else None)

with open("data/results/rate_dependence.json", "w") as f:
    json.dump(rate_dep, f, indent=2)
print("  Saved rate-dependence data.")


print("\n=== All simulations complete ===")
print("Results in data/results/")
