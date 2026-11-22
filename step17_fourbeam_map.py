"""MAP on the four-beam report, matched to the bound in step14_complete.py.

The report is the strongest noise-free beam and three differentials. Beam indices
are fixed from the noise-free mean at the true wall, as in that bound. Noise is
Gaussian with the Sheppard variances used there: sigma_e^2 + 1/12 on the
strongest beam and sigma_e^2 + 4/12 on each differential. sigma_e = 1 dB.

The truth is moved from the map by U[-6, 6] m and U[-2, 2] deg. Starts sit on
the map, at d offsets {0, +-3, +-6} m and eps_r in {3, 8}. No start uses the truth.
1000 trials per class (50 sets x 20 draws), facade B, N = 10, sigma_u = 1 m.

Usage:
  python3 step17_fourbeam_map.py check    # recompute the published 4.73 / 3.10 medians
  python3 step17_fourbeam_map.py bench    # one noise-free recovery, then stop
  python3 step17_fourbeam_map.py full     # 1000 trials per class; resumes a partial file
"""
import math
import os
import sys
import time
import numpy as np
import torch
import rsrp_model as rm
from step4_mle import setup
from step14_complete import schur, crb_stat

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SIGMA_U, N_USERS = 1.0, 10
SETS, DRAWS = 50, 20
SIGMA_E = 1.0
D_STARTS, EPS_STARTS = [0.0, -3.0, 3.0, -6.0, 6.0], [3.0, 8.0]
WORKERS = int(os.environ.get("WORKERS", "12"))
PARTIAL = os.path.join(ART, "step17_fourbeam_partial.npz")


def reduce_mu(mu, pick):
    n = torch.arange(mu.shape[0])
    strong = mu[n, pick[:, 0]]
    cols = [strong]
    for i in range(1, 4):
        cols.append(mu[n, pick[:, i]] - strong)
    return torch.stack(cols, 1)


def report_std(se):
    return torch.tensor([
        math.sqrt(se ** 2 + 1.0 / 12),
        math.sqrt(se ** 2 + 4.0 / 12),
        math.sqrt(se ** 2 + 4.0 / 12),
        math.sqrt(se ** 2 + 4.0 / 12),
    ])


def information(cfg, th, users, h0, pick=None):
    """Per-user robust information of the four-beam report. pick fixed, or chosen here."""
    def mu(th_, u, hr, dxy, psi):
        uu = (u + torch.cat([dxy, torch.zeros(1)]))[None]
        return rm.rsrp_mean_phase(cfg, th_, uu, torch.view_as_complex(hr)[None], psi[:1])[0]

    n = users.shape[0]
    jac = torch.func.vmap(torch.func.jacfwd(mu, argnums=(0, 3, 4)), in_dims=(None, 0, 0, 0, 0))
    Jt, Jd, Jp = jac(th, users, torch.view_as_real(h0), torch.zeros(n, 2), torch.zeros(n, 1))
    if pick is None:
        with torch.no_grad():
            mu0 = rm.rsrp_mean_phase(cfg, th, users, h0, torch.zeros(n))
            pick = mu0.argsort(dim=1, descending=True)[:, :4]

    def take(J):
        js = J[torch.arange(n), pick[:, 0]]
        rows = [js]
        for i in range(1, 4):
            rows.append(J[torch.arange(n), pick[:, i]] - js)
        return torch.stack(rows, 1)

    Jt, Jd, Jp = take(Jt), take(Jd), take(Jp)
    sig = report_std(cfg.sigma_e).repeat(n, 1)
    return schur(Jt, torch.cat([Jd, Jp], -1), sig), pick


def nll(r, y, std):
    return ((r - y) ** 2 / std ** 2).sum() / 2


def fit_top4(cfg, r, users_hat, h0, pick, th_init, free=(1.0, 1.0, 1.0, 1.0), nuis=None, ret_nuis=False, nphase=16):
    """L-BFGS on (theta, offsets, wall phases) for a fixed four-beam report."""
    n = len(users_hat)
    std = report_std(cfg.sigma_e)
    cap = torch.tensor([5.0, math.radians(2.0), 3.0, 1.0])
    S = torch.tensor(free) * cap
    zt = torch.zeros(4, requires_grad=True)
    zd = torch.zeros(n, 2, requires_grad=True)
    with torch.no_grad():
        if nuis is not None:
            zd.data = nuis[0].clone() / SIGMA_U
            psi0 = nuis[1].clone()
        else:
            grid = torch.arange(nphase) * (2 * math.pi / nphase)
            cols = []
            for g in grid:
                mu = rm.rsrp_mean_phase(cfg, th_init, users_hat, h0, g.repeat(n))
                y = reduce_mu(mu, pick)
                cols.append(((r - y) ** 2 / std ** 2).sum(1))
            psi0 = grid[torch.stack(cols, 1).argmin(1)]
    psi = psi0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([zt, zd, psi], lr=1, max_iter=600, tolerance_grad=1e-9,
                            tolerance_change=1e-14, history_size=50, line_search_fn="strong_wolfe")

    def cost():
        opt.zero_grad()
        th = th_init + S * zt
        dxy = SIGMA_U * zd
        u = users_hat + torch.cat([dxy, torch.zeros(n, 1)], 1)
        mu = rm.rsrp_mean_phase(cfg, th, u, h0, psi)
        c = nll(r, reduce_mu(mu, pick), std) + (zd ** 2).sum() / 2
        c.backward()
        return c

    try:
        opt.step(cost)
        with torch.no_grad():
            th_now = th_init + S * zt
            u = users_hat + torch.cat([SIGMA_U * zd, torch.zeros(n, 1)], 1)
            mu = rm.rsrp_mean_phase(cfg, th_now, u, h0, psi)
            c = (nll(r, reduce_mu(mu, pick), std) + (zd ** 2).sum() / 2).item()
    except RuntimeError:
        c = float("inf")
    th = (th_init + S * zt).detach()
    if ret_nuis:
        return th, c, (SIGMA_U * zd.detach(), psi.detach())
    return th, c


def best_fit(cfg, r, u_hat, h0, pick, th_map):
    best = (float("inf"), None)
    for dd in D_STARTS:
        for eps in EPS_STARTS:
            init = th_map.clone()
            init[0] += dd
            init[2], init[3] = eps, -1.5
            th, c = fit_top4(cfg, r, u_hat, h0, pick, init)
            if c < best[0]:
                best = (c, th)
    return best


def make_reports(cfg, th_true, u, h0, gen):
    mu = rm.rsrp_mean_phase(cfg, th_true, u, h0, torch.zeros(len(u)))
    pick = mu.argsort(dim=1, descending=True)[:, :4]
    y = reduce_mu(mu, pick)
    r = y + report_std(cfg.sigma_e) * torch.randn(y.shape, generator=gen)
    return r, pick


def trial(task):
    torch.set_num_threads(1)
    name, sid, idx, seed = task
    cfg, th0, U, H0, _ = setup("B")
    cfg.sigma_e = SIGMA_E
    u, h0 = U[idx], H0[idx]
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    th_true = th0.clone()
    th_true[0] += rng.uniform(-6, 6)
    th_true[1] += math.radians(rng.uniform(-2, 2))
    r, pick = make_reports(cfg, th_true, u, h0, gen)
    u_hat = u + torch.cat([SIGMA_U * torch.randn(len(idx), 2, generator=gen), torch.zeros(len(idx), 1)], 1)
    J, _ = information(cfg, th_true, u, h0, pick=pick)
    M = J.sum(0).detach().numpy()
    crb = np.full(4, np.nan)
    ev = np.linalg.eigvalsh(M)
    if ev[0] > 1e-8:
        crb = np.diag(np.linalg.inv(M))
    _c, th_hat = best_fit(cfg, r, u_hat, h0, pick, th0)
    err = (th_hat - th_true).numpy() if th_hat is not None else np.full(4, np.nan)
    return name, sid, seed, err, crb


def check():
    """The published medians are 4.73 m LOS and 3.10 m NLOS on seed-14 sets."""
    cfg, th0, U, H0, los = setup("B")
    cfg.sigma_e = SIGMA_E
    rng = np.random.default_rng(14)
    pools = {"LOS": np.where(los)[0], "NLOS": np.where(~los)[0]}
    sets = {k: [rng.choice(v, N_USERS, replace=False) for _ in range(300)] for k, v in pools.items()}
    expect = {"LOS": 4.73, "NLOS": 3.10}
    t0 = time.time()
    for name in ("LOS", "NLOS"):
        Js = []
        for s in sets[name]:
            J, _ = information(cfg, th0, U[s], H0[s])
            Js.append(J.detach())
        med, lo, hi, nbad = crb_stat(torch.stack(Js).numpy(), range(300))
        print(f"check {name}: median {med:.2f} [{lo:.2f}, {hi:.2f}] singular {nbad}  ({time.time() - t0:.0f}s)", flush=True)
        if abs(med - expect[name]) > 0.02 or nbad:
            raise SystemExit(f"four-beam information does not match the published {expect[name]:.2f} m median")
    print("check ok", flush=True)


def bench():
    cfg, th0, U, H0, los = setup("B")
    cfg.sigma_e = SIGMA_E
    idx = np.where(los)[0][:N_USERS]
    u, h0 = U[idx], H0[idx]
    gen = torch.Generator().manual_seed(1)
    mu = rm.rsrp_mean_phase(cfg, th0, u, h0, torch.zeros(N_USERS))
    pick = mu.argsort(dim=1, descending=True)[:, :4]
    r = reduce_mu(mu, pick)
    t0 = time.time()
    c, th = best_fit(cfg, r, u, h0, pick, th0)
    err = (th - th0).detach().numpy()
    print(
        f"noise-free recovery d {err[0]:.4f} m, phi {math.degrees(err[1]):.4f} deg, "
        f"eps {err[2]:.4f}, cost {c:.4f}, {time.time() - t0:.0f}s",
        flush=True,
    )
    if abs(float(err[0])) > 0.3:
        raise SystemExit("noise-free four-beam fit did not return to the true wall")


def save_rows(rows):
    if not rows:
        return
    np.savez_compressed(
        PARTIAL,
        name=np.array([r[0] for r in rows]),
        sid=np.array([r[1] for r in rows]),
        seed=np.array([r[2] for r in rows]),
        err=np.array([r[3] for r in rows]),
        crb=np.array([r[4] for r in rows]),
    )


def summarise(rows):
    lines = [f"four-beam MAP, sigma_e={SIGMA_E} dB, Sheppard variances, {SETS} sets x {DRAWS} draws, N={N_USERS}"]
    brng = np.random.default_rng(1)
    for name in ("LOS", "NLOS"):
        sub = [r for r in rows if r[0] == name]
        err = np.array([r[3] for r in sub])
        crb = np.array([r[4] for r in sub])
        sid = np.array([r[1] for r in sub])
        ok = np.isfinite(err[:, 0])
        rmse = np.sqrt(np.nanmean(err ** 2, 0))
        bound = np.sqrt(np.nanmean(crb, 0))
        sets = np.unique(sid)
        boot = []
        for _ in range(2000):
            pick = brng.choice(sets, len(sets), replace=True)
            m = np.concatenate([np.where(sid == s)[0] for s in pick])
            boot.append(np.sqrt(np.nanmean(err[m, 0] ** 2)))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        far = np.mean(np.abs(err[ok, 0]) > 3 * np.sqrt(crb[ok, 0]))
        per = np.sqrt(crb[:, 0])
        lines.append(
            f"{name}: d RMSE {rmse[0]:.2f} m [{lo:.2f}, {hi:.2f}] / sqrt(mean CRB) {bound[0]:.2f}; "
            f"median per-set bound {np.median(per):.2f}; "
            f"|d err| > 3 sqrt(CRB) in {100 * far:.0f}%  [{ok.sum()} fits, {(~ok).sum()} failed, {len(sets)} sets]"
        )
    return lines


def full():
    from multiprocessing import get_context
    _, _, _, _, los = setup("B")
    pools = {"LOS": np.where(los)[0], "NLOS": np.where(~los)[0]}
    rng = np.random.default_rng(17)
    tasks, seed = [], 170000
    for name, pool in pools.items():
        sets = [rng.choice(pool, N_USERS, replace=False) for _ in range(SETS)]
        for sid, idx in enumerate(sets):
            for _ in range(DRAWS):
                seed += 1
                tasks.append((name, sid, idx, seed))
    rows = []
    if os.path.exists(PARTIAL):
        z = np.load(PARTIAL, allow_pickle=True)
        rows = list(zip(z["name"], z["sid"], z["seed"], z["err"], z["crb"]))
        have = set(int(s) for s in z["seed"])
        tasks = [t for t in tasks if t[3] not in have]
        print(f"resume {len(rows)} done, {len(tasks)} left", flush=True)
    t0 = time.time()
    if tasks:
        with get_context("fork").Pool(WORKERS) as p:
            for k, row in enumerate(p.imap_unordered(trial, tasks, chunksize=1), 1):
                rows.append(row)
                if k % 20 == 0 or k == len(tasks):
                    save_rows(rows)
                    print(f"progress {k}/{len(tasks)} new, {len(rows)} stored, {time.time() - t0:.0f}s", flush=True)
    lines = summarise(rows)
    lines.append(f"total {time.time() - t0:.0f}s")
    text = "\n".join(lines)
    print(text, flush=True)
    open(os.path.join(ART, "step17_fourbeam.txt"), "w").write(text + "\n")
    save_rows(rows)
    np.savez_compressed(os.path.join(ART, "step17_fourbeam.npz"), **dict(np.load(PARTIAL)))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "check":
        check()
    elif mode == "bench":
        bench()
    elif mode == "full":
        full()
    else:
        raise SystemExit("usage: step17_fourbeam_map.py check|bench|full")
