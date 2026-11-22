"""Detection at a 3 m displacement. The published curve skipped this point.

Facade B, 3.5 GHz, N = 10, material known. Same statistic as step5_detect.trial
(kind "model"). 200 trials per class. Scored at the existing 1% thresholds from
step5b_detect_ci.npz (2000 no-change trials). Seeds start at 160000, disjoint
from step 5b.

Usage: python3 step16_delta3.py
"""
import math
import os
import time
import numpy as np
import torch
import step5_detect as s5

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
N_TRIALS = 200
DELTA = 3.0
WORKERS = int(os.environ.get("WORKERS", "12"))


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def one(task):
    torch.set_num_threads(1)
    cls_name, seed = task
    key, t = s5.trial(("model", cls_name, 10, DELTA, seed))
    return cls_name, t


def main():
    from multiprocessing import get_context
    z = np.load(os.path.join(ART, "step5b_detect_ci.npz"))
    thr = {c: float(z[f"{c}_pfa1_thr"]) for c in ("LOS", "NLOS")}
    tasks, seed = [], 160000
    for cls_name in ("LOS", "NLOS"):
        for _ in range(N_TRIALS):
            seed += 1
            tasks.append((cls_name, seed))
    t0 = time.time()
    with get_context("fork").Pool(WORKERS) as p:
        out = p.map(one, tasks, chunksize=1)
    lines = [f"Delta {DELTA:g} m, N=10, {N_TRIALS} trials, wall {time.time() - t0:.0f}s"]
    save = {"delta": np.array([DELTA])}
    for cls_name in ("LOS", "NLOS"):
        t = np.array([v for c, v in out if c == cls_name])
        k = int((t > thr[cls_name]).sum())
        lo, hi = wilson(k, len(t))
        save[f"{cls_name}_T"] = t
        save[f"{cls_name}_thr"] = np.array([thr[cls_name]])
        lines.append(
            f"{cls_name}: threshold {thr[cls_name]:.2f} (step5b, 2000 H0), "
            f"P_D {k / len(t):.3f} [{lo:.3f}, {hi:.3f}]  ({k}/{len(t)}), median T {np.median(t):.2f}"
        )
    text = "\n".join(lines)
    print(text, flush=True)
    open(os.path.join(ART, "step16_delta3.txt"), "w").write(text + "\n")
    np.savez_compressed(os.path.join(ART, "step16_delta3.npz"), **save)


if __name__ == "__main__":
    main()
