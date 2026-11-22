"""M4: for every MAP fit, compare the cost at the estimate with the cost at the truth.

Same design as step4b_mle_far.py (facade B, 3.5 GHz, N=10, sigma_u=1 m, truth moved by
U[-6, 6] m and U[-2, 2] deg, starts at the map). The truth cost holds (d, phi, material)
at the truth and optimises only the position offsets and wall-path phases.
A distant fit is one with |d error| > 3 sqrt(CRB).

Usage: python3 step4d_costs.py [draws]
  draws defaults to 4 (the paper). Use 2 to finish sooner.
"""
import math
import os
import sys
import time
import numpy as np
import torch
from step4_mle import fit, setup
import rsrp_model as rm

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SIGMA_U, N_USERS = 1.0, 10
SETS = 30
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SIGMA_E = [0.25, 1.0]
D_STARTS, EPS_STARTS = [0.0, -3.0, 3.0, -6.0, 6.0], [3.0, 8.0]


def best_fit(cfg, r, u_hat, h0, th_map):
    best = (float("inf"), None)
    for dd in D_STARTS:
        for eps in EPS_STARTS:
            init = th_map.clone()
            init[0] += dd
            init[2], init[3] = eps, -1.5
            th, c = fit(cfg, r, u_hat, h0, SIGMA_U, init, False)
            if c < best[0]:
                best = (c, th)
    return best


def trial(task):
    torch.set_num_threads(1)
    name, se, sid, idx, seed = task
    cfg, th0, U, H0, _ = setup("B")
    cfg.sigma_e = se
    u, h0 = U[idx], H0[idx]
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    th_true = th0.clone()
    th_true[0] += rng.uniform(-6, 6)
    th_true[1] += math.radians(rng.uniform(-2, 2))
    J = rm.fisher_robust(cfg, th_true, u, h0, sigma_u=SIGMA_U).sum(0).numpy()
    crb = float(np.linalg.inv(J)[0, 0])
    mu = rm.rsrp_mean_phase(cfg, th_true, u, h0, torch.zeros(len(idx)))
    r = mu + se * torch.randn(mu.shape, generator=gen)
    u_hat = u + torch.cat([SIGMA_U * torch.randn(len(idx), 2, generator=gen), torch.zeros(len(idx), 1)], 1)
    c_hat, th = best_fit(cfg, r, u_hat, h0, th0)
    # nuisances only, wall held at the truth
    _, c_true = fit(cfg, r, u_hat, h0, SIGMA_U, th_true.clone(), False, free=(0.0, 0.0, 0.0, 0.0))
    err = float((th - th_true)[0]) if th is not None else float("nan")
    # RMS gap of the two fitted means, in dB, on the reported beams
    return name, se, sid, err, crb, c_hat, c_true


def main():
    from multiprocessing import get_context
    _, _, _, _, los = setup("B")
    pools = {"LOS": np.where(los)[0], "NLOS": np.where(~los)[0]}
    rng = np.random.default_rng(7)          # same user sets as step4b
    tasks, seed = [], 40000
    for name, pool in pools.items():
        sets = [rng.choice(pool, N_USERS, replace=False) for _ in range(SETS)]
        for se in SIGMA_E:
            for sid, idx in enumerate(sets):
                for _ in range(DRAWS):
                    seed += 1
                    tasks.append((name, se, sid, idx, seed))
    t0 = time.time()
    ctx = get_context("fork")
    with ctx.Pool(16) as p:
        out = p.map(trial, tasks, chunksize=1)
    lines = [f"draws {DRAWS}, sets {SETS}, wall {time.time() - t0:.0f}s"]
    res = {}
    for name, se, sid, err, crb, c_hat, c_true in out:
        res.setdefault((name, se), []).append((err, crb, c_hat, c_true, sid))
    for key, rows in sorted(res.items()):
        err = np.array([r[0] for r in rows])
        crb = np.array([r[1] for r in rows])
        ch = np.array([r[2] for r in rows])
        ct = np.array([r[3] for r in rows])
        far = np.abs(err) > 3 * np.sqrt(crb)
        # cost is ||r-mu||^2/(2 sigma^2) + prior. Lower is better.
        better = ch < ct - 1e-3
        worse = ch > ct + 1e-3
        gap = ch - ct
        lines.append(
            f"{key[0]} {key[1]:.2f} dB: n {len(rows)} distant {far.mean():.2f}  "
            f"of distant, C(hat)<C(true) {better[far].mean() if far.any() else float('nan'):.2f}, "
            f"C(hat)>C(true) {worse[far].mean() if far.any() else float('nan'):.2f}, "
            f"median gap {np.median(gap[far]) if far.any() else float('nan'):.2f}  "
            f"of all fits, C(hat)<C(true) {better.mean():.2f}")
    text = "\n".join(lines)
    print(text, flush=True)
    open(os.path.join(ART, "step4d_costs.txt"), "w").write(text + "\n")
    np.savez(os.path.join(ART, "step4d_costs.npz"),
             rows=np.array(out, dtype=object))


if __name__ == "__main__":
    main()
