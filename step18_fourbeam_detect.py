"""Likelihood test on the four-beam report.

Same costs as step5_detect, but the observations are the four-beam reports of
step17_fourbeam_map.py. Beam indices are fixed from the noise-free mean at the
true wall. Threshold is the empirical 99th percentile of four-beam no-change
trials. It is not the 16-beam threshold 9.6.

Facade B, 3.5 GHz, N = 10, material known, sigma_e = 1 dB.
1000 no-change trials and 200 trials at 2, 3 and 4 m, each class.
H0_TRIALS and H1_TRIALS override those counts. A smaller run is a
different experiment: its threshold is the 99th percentile of its own
no-change trials, not the 16-beam value 9.6, and the paper must say
the counts.

Usage: python3 step18_fourbeam_detect.py
Resumes artifacts/step18_fourbeam_partial.npz if it exists.
"""
import math
import os
import time
import numpy as np
import torch
import rsrp_model as rm
from step4_mle import setup
from step17_fourbeam_map import SIGMA_U, fit_top4, make_reports, reduce_mu, report_std

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
H0_TRIALS = int(os.environ.get("H0_TRIALS", "1000"))
H1_TRIALS = int(os.environ.get("H1_TRIALS", "200"))
DELTAS = [0.0, 2.0, 3.0, 4.0]
D_STARTS = [0.0, -3.0, 3.0, -6.0, 6.0]
WORKERS = int(os.environ.get("WORKERS", "12"))
PARTIAL = os.path.join(ART, "step18_fourbeam_partial.npz")


def nuisance_grid(cfg, th, r, u_hat, h0, pick, step=0.2, half=2.5):
    g = torch.arange(-half, half + 1e-9, step)
    gx, gy = [a.reshape(-1) for a in torch.meshgrid(g, g, indexing="ij")]
    m, ps = len(gx), torch.arange(16) * (2 * math.pi / 16)
    std = report_std(cfg.sigma_e)
    prior = (gx ** 2 + gy ** 2) / (2 * SIGMA_U ** 2)
    dxy, psi = torch.zeros(len(u_hat), 2), torch.zeros(len(u_hat))
    with torch.no_grad():
        for n in range(len(u_hat)):
            u = u_hat[n].repeat(m, 1)
            u[:, 0] += gx
            u[:, 1] += gy
            cols = []
            pk = pick[n:n + 1].expand(m, 4)
            for p in ps:
                mu = rm.rsrp_mean_phase(cfg, th, u, h0[n].repeat(m, 1), p.repeat(m))
                y = reduce_mu(mu, pk)
                cols.append(((r[n] - y) ** 2 / std ** 2).sum(1) / 2 + prior)
            k = int(torch.stack(cols, 1).argmin())
            dxy[n] = torch.stack([gx[k // 16], gy[k // 16]])
            psi[n] = ps[k % 16]
    return dxy, psi


def costs(cfg, th_map, r, u_hat, h0, pick):
    held, freed = (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)
    _, c0, nu0 = fit_top4(cfg, r, u_hat, h0, pick, th_map.clone(), free=held, ret_nuis=True)
    _, c0g, nu0g = fit_top4(cfg, r, u_hat, h0, pick, th_map.clone(), free=held,
                            nuis=nuisance_grid(cfg, th_map, r, u_hat, h0, pick), ret_nuis=True)
    if c0g < c0:
        c0, nu0 = c0g, nu0g
    c1, th1, nu1 = c0, th_map.clone(), nu0
    for dd in D_STARTS:
        init = th_map.clone()
        init[0] += dd
        _th, c, nu = fit_top4(cfg, r, u_hat, h0, pick, init, free=freed, nuis=nu0, ret_nuis=True)
        if c < c1:
            c1, th1, nu1 = c, _th, nu
    if c1 < c0:
        _th, c, nu = fit_top4(cfg, r, u_hat, h0, pick, th1.clone(), free=freed,
                              nuis=nuisance_grid(cfg, th1, r, u_hat, h0, pick), ret_nuis=True)
        if c < c1:
            c1, th1, nu1 = c, _th, nu
        _, c0b = fit_top4(cfg, r, u_hat, h0, pick, th_map.clone(), free=held, nuis=nu1)
        c0 = min(c0, c0b)
    return c0, min(c1, c0)


def one(task):
    torch.set_num_threads(1)
    cls_name, delta, seed = task
    cfg, th0, U, H0, los = setup("B")
    cfg.sigma_e = 1.0
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    pool = np.where(los if cls_name == "LOS" else ~los)[0]
    idx = rng.choice(pool, 10, replace=False)
    u, h0 = U[idx], H0[idx]
    th_true = th0.clone()
    th_true[0] += delta
    r, pick = make_reports(cfg, th_true, u, h0, gen)
    u_hat = u + torch.cat([SIGMA_U * torch.randn(10, 2, generator=gen), torch.zeros(10, 1)], 1)
    c0, c1 = costs(cfg, th0, r, u_hat, h0, pick)
    return cls_name, delta, seed, 2 * (c0 - c1)


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def save_rows(rows):
    np.savez_compressed(
        PARTIAL,
        name=np.array([r[0] for r in rows]),
        delta=np.array([r[1] for r in rows]),
        seed=np.array([r[2] for r in rows]),
        T=np.array([r[3] for r in rows]),
    )


def main():
    from multiprocessing import get_context
    tasks, seed = [], 180000
    for cls_name in ("LOS", "NLOS"):
        for delta in DELTAS:
            n = H0_TRIALS if delta == 0.0 else H1_TRIALS
            for _ in range(n):
                seed += 1
                tasks.append((cls_name, delta, seed))
    rows = []
    if os.path.exists(PARTIAL):
        z = np.load(PARTIAL, allow_pickle=True)
        rows = list(zip(z["name"], z["delta"], z["seed"], z["T"]))
        have = set(int(s) for s in z["seed"])
        tasks = [t for t in tasks if t[2] not in have]
        print(f"resume {len(rows)} done, {len(tasks)} left", flush=True)
    t0 = time.time()
    if tasks:
        with get_context("fork").Pool(WORKERS) as p:
            for k, row in enumerate(p.imap_unordered(one, tasks, chunksize=1), 1):
                rows.append(row)
                if k % 10 == 0 or k == len(tasks):
                    save_rows(rows)
                    print(f"progress {k}/{len(tasks)} new, {len(rows)} stored, {time.time() - t0:.0f}s", flush=True)
    lines = [f"four-beam likelihood test, N=10, H0 {H0_TRIALS}, H1 {H1_TRIALS}, wall {time.time() - t0:.0f}s"]
    save = {}
    for cls_name in ("LOS", "NLOS"):
        t0s = np.array([r[3] for r in rows if r[0] == cls_name and r[1] == 0.0])
        thr = float(np.percentile(t0s, 99))
        save[f"{cls_name}_T0"] = t0s
        save[f"{cls_name}_thr"] = np.array([thr])
        parts = []
        for delta in DELTAS[1:]:
            t1 = np.array([r[3] for r in rows if r[0] == cls_name and r[1] == delta])
            k = int((t1 > thr).sum())
            lo, hi = wilson(k, len(t1))
            save[f"{cls_name}_{delta:g}_T"] = t1
            parts.append(f"{delta:g}m:{k / len(t1):.3f}[{lo:.3f},{hi:.3f}]")
        lines.append(f"{cls_name}: threshold {thr:.2f} ({len(t0s)} H0, median T {np.median(t0s):.2f}); P_D " + " ".join(parts))
    text = "\n".join(lines)
    print(text, flush=True)
    open(os.path.join(ART, "step18_fourbeam.txt"), "w").write(text + "\n")
    save_rows(rows)
    np.savez_compressed(os.path.join(ART, "step18_fourbeam.npz"), **save)


if __name__ == "__main__":
    main()
