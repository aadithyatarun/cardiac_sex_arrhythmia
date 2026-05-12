"""
Fast parallel block-sweep using multiprocessing.
Runs all (sex, CL, block) combinations concurrently.
"""

import sys, os, json, time
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("data/results", exist_ok=True)

BLOCK_FRACTIONS = np.linspace(0.0, 0.95, 12)
CYCLE_LENGTHS   = [500.0, 1000.0, 2000.0]
N_WARMUP        = 20
N_BEATS         = 25   # enough for convergence; ~10× faster than 250


def _worker(args):
    sex_label, cl, blk = args
    # Must import inside worker (multiprocessing forks)
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    from models.parameters import MALE, FEMALE
    from simulations.pacing import run_steady_state, compute_apd, detect_ead, compute_triangulation

    params = MALE if sex_label == "male" else FEMALE
    try:
        t, y = run_steady_state(params, cl=cl, n_beats=N_WARMUP + N_BEATS,
                                n_save=1, ikr_block=blk)
        vm = y[0, :]
        apd90 = compute_apd(t, vm, 0.90)
        apd50 = compute_apd(t, vm, 0.50)
        apd30 = compute_apd(t, vm, 0.30)
        tri   = compute_triangulation(t, vm)
        ead   = detect_ead(t, vm)

        # Save AP trace at key block levels
        if abs(blk) < 0.01 or abs(blk - 0.50) < 0.06 or abs(blk - 0.80) < 0.06:
            np.savez_compressed(
                f"data/results/ap_{sex_label}_cl{int(cl)}_blk{int(blk*100)}.npz",
                t=t, vm=vm)

        return (sex_label, cl, blk, apd90, apd50, apd30, tri, ead, None)
    except Exception as e:
        return (sex_label, cl, blk, np.nan, np.nan, np.nan, np.nan, False, str(e))


if __name__ == "__main__":
    t0 = time.time()
    tasks = [(sex, cl, blk)
             for sex in ("male", "female")
             for cl  in CYCLE_LENGTHS
             for blk in BLOCK_FRACTIONS]

    n_workers = min(cpu_count(), len(tasks), 8)
    print(f"Running {len(tasks)} simulations with {n_workers} workers...", flush=True)

    block_results   = {sex: {str(int(cl)): {"block":[],"apd90":[],"apd50":[],"apd30":[],"tri":[],"ead":[]}
                             for cl in CYCLE_LENGTHS}
                       for sex in ("male","female")}
    ead_thresholds  = {sex: {str(int(cl)): None for cl in CYCLE_LENGTHS}
                       for sex in ("male","female")}

    with Pool(n_workers) as pool:
        for res in pool.imap_unordered(_worker, tasks):
            sex, cl, blk, apd90, apd50, apd30, tri, ead, err = res
            key = str(int(cl))
            d = block_results[sex][key]
            d["block"].append(float(blk))
            d["apd90"].append(float(apd90) if not np.isnan(apd90) else None)
            d["apd50"].append(float(apd50) if not np.isnan(apd50) else None)
            d["apd30"].append(float(apd30) if not np.isnan(apd30) else None)
            d["tri"].append(float(tri)   if not np.isnan(tri)   else None)
            d["ead"].append(int(ead))
            if err:
                print(f"  ERROR {sex} CL={int(cl)} blk={blk:.2f}: {err}", flush=True)
            else:
                print(f"  {sex} CL={int(cl)} blk={blk:.2f}: APD90={apd90:.1f} EAD={ead}", flush=True)

    # Sort each list by block fraction
    for sex in ("male","female"):
        for key in block_results[sex]:
            d = block_results[sex][key]
            order = np.argsort(d["block"])
            for field in ("block","apd90","apd50","apd30","tri","ead"):
                d[field] = [d[field][i] for i in order]
            # Find EAD threshold
            for blk, ead in zip(d["block"], d["ead"]):
                if ead == 1 and ead_thresholds[sex][key] is None:
                    ead_thresholds[sex][key] = float(blk)

    with open("data/results/block_sweep.json",    "w") as f: json.dump(block_results,  f, indent=2)
    with open("data/results/ead_thresholds.json", "w") as f: json.dump(ead_thresholds, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.")
    print("EAD thresholds:", json.dumps(ead_thresholds, indent=2))
