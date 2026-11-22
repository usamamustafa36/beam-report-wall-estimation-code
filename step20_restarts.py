"""Richer multi-start on the LOS noise sweep of step4b.

Same facade, same user sets, same noise seeds as the LOS rows of step4b
(rng seed 7, trial seeds starting at 40000, LOS generated first). The search
is wider: d starts at 0, +-2, +-4, +-6, +-8 m, eps_r in {3, 5, 7, 9}, and the
wall-phase grid has 32 points. sigma_e in {0.25, 1, 4} dB, 30 sets x 4 draws.

Also records the cost at the estimate and the cost with the wall held at the
truth, so a fit that stopped short of the truth can be counted.

Usage: python3 step20_restarts.py
Resumes artifacts/step20_partial.npz.
"""
import math
import os
import time
import numpy as np
import torch
import rsrp_model as rm
from step4_mle import fit, setup

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SIGMA_U, N_USERS = 1.0, 10
SETS, DRAWS = 30, 4
SIGMA_E = [0.25, 1.0, 4.0]
D_STARTS = [0.0, -2.0, 2.0, -4.0, 4.0, -6.0, 6.0, -8.0, 8.0]
EPS_STARTS = [3.0, 5.0, 7.0, 9.0]
WORKERS = int(os.environ.get("WORKERS", "12"))
PARTIAL = os.path.join(ART, "step20_partial.npz")


def best_fit(cfg, r, u_hat, h0, th_map):
    best = (float("inf"), None)
    for dd in D_STARTS:
        for eps in EPS_STARTS:
            init = th_map.clone()
            init[0] += dd
            init[2], init[3] = eps, -1.5
            th, c = fit(cfg, r, u_hat, h0, SIGMA_U, init, False, nphase=32)
            if c < best[0]:
                best = (c, th)
    return best


def trial(task):
    torch.set_num_threads(1)
    se, sid, idx, seed = task
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
    _, c_true = fit(cfg, r, u_hat, h0, SIGMA_U, th_true.clone(), False, free=(0.0, 0.0, 0.0, 0.0), nphase=32)
    err = float((th - th_true)[0]) if th is not None else float("nan")
    return se, sid, seed, err, crb, c_hat, c_true


def save_rows(rows):
    np.savez_compressed(
        PARTIAL,
        se=np.array([r[0] for r in rows]),
        sid=np.array([r[1] for r in rows]),
        seed=np.array([r[2] for r in rows]),
        err=np.array([r[3] for r in rows]),
        crb=np.array([r[4] for r in rows]),
        chat=np.array([r[5] for r in rows]),
        ctrue=np.array([r[6] for r in rows]),
    )


def main():
    from multiprocessing import get_context
    _, _, _, _, los = setup("B")
    pool = np.where(los)[0]
    rng = np.random.default_rng(7)
    # Match step4b: LOS sets are drawn first, then NLOS sets (discarded here).
    sets = [rng.choice(pool, N_USERS, replace=False) for _ in range(SETS)]
    _nlos = np.where(~los)[0]
    _ = [rng.choice(_nlos, N_USERS, replace=False) for _ in range(SETS)]
    tasks, seed = [], 40000
    for se in SIGMA_E:
        for sid, idx in enumerate(sets):
            for _k in range(DRAWS):
                seed += 1
                tasks.append((se, sid, idx, seed))
    rows = []
    if os.path.exists(PARTIAL):
        z = np.load(PARTIAL)
        rows = list(zip(z["se"], z["sid"], z["seed"], z["err"], z["crb"], z["chat"], z["ctrue"]))
        have = set(int(s) for s in z["seed"])
        tasks = [t for t in tasks if t[3] not in have]
        print(f"resume {len(rows)} done, {len(tasks)} left", flush=True)
    t0 = time.time()
    if tasks:
        with get_context("fork").Pool(WORKERS) as p:
            for k, row in enumerate(p.imap_unordered(trial, tasks, chunksize=1), 1):
                rows.append(row)
                if k % 10 == 0 or k == len(tasks):
                    save_rows(rows)
                    print(f"progress {k}/{len(tasks)} new, {len(rows)} stored, {time.time() - t0:.0f}s", flush=True)
    lines = [
        "LOS richer multi-start. Previous step4b LOS RMSE: 0.25 dB 3.66 m, 1 dB 2.75 m, 4 dB 4.77 m.",
        f"starts d {D_STARTS}, eps {EPS_STARTS}, phase grid 32, wall {time.time() - t0:.0f}s",
    ]
    rmses = []
    for se in SIGMA_E:
        sub = [r for r in rows if abs(r[0] - se) < 1e-9]
        err = np.array([r[3] for r in sub])
        crb = np.array([r[4] for r in sub])
        ch = np.array([r[5] for r in sub])
        ct = np.array([r[6] for r in sub])
        ok = np.isfinite(err)
        rmse = float(np.sqrt(np.nanmean(err ** 2)))
        rmses.append(rmse)
        far = ok & (np.abs(err) > 3 * np.sqrt(crb))
        short = far & (ch > ct)
        worse = far & (ch < ct)
        gap = ch[short] - ct[short]
        lines.append(
            f"LOS {se:.2f} dB: d RMSE {rmse:.2f} m / sqrt(mean CRB) {np.sqrt(crb.mean()):.2f}; "
            f"distant {int(far.sum())}/{len(sub)}; of distant, C(hat)>C(true) {int(short.sum())} "
            f"median gap {np.median(gap) if short.any() else float('nan'):.2f}; "
            f"C(hat)<C(true) {int(worse.sum())}  [{ok.sum()} fits]"
        )
    mono = rmses[0] < rmses[1] < rmses[2]
    lines.append(f"RMSE increases with sigma_e: {bool(mono)}  ({rmses[0]:.2f}, {rmses[1]:.2f}, {rmses[2]:.2f})")
    text = "\n".join(lines)
    print(text, flush=True)
    open(os.path.join(ART, "step20_restarts.txt"), "w").write(text + "\n")
    save_rows(rows)


if __name__ == "__main__":
    main()
