"""Non-wall amplitude as a nuisance, facade B, 3.5 GHz, N = 10, sigma_u = 1 m.

crb  : 300 sets (seed 14, the sets of step14).
       split  — phases of the direct path and of the residual, amplitudes fixed.
       amp    — those phases, plus a log10 amplitude on each of those two paths.
       scale  — one log10 amplitude on the stored h0, plus the wall-path phase.
       No prior on phases or amplitudes. Position offsets keep the 1 m prior.
map  : MAP at sigma_e = 1 dB with one log10 amplitude per user on the stored h0,
       bounded by a factor of 100 (a = 2 tanh z). 30 sets x 4 draws, both classes.
       Starts match step4b. The matched bound is the scale row above, evaluated
       at each trial's true wall.

Usage: python3 step19_amp.py crb|map
"""
import math
import os
import sys
import time
import numpy as np
import torch
import rsrp_model as rm
from step4_mle import setup
from step14_complete import schur, direct, crb_stat

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SIGMA_U, N, SETS = 1.0, 10, 300
WORKERS = int(os.environ.get("WORKERS", "12"))


def batches(cfg, th, users, h0, los_row, kind):
    hd = direct(cfg, users)
    use = torch.tensor(los_row)
    hd = torch.where(use[:, None], hd, torch.zeros_like(hd))
    hr = torch.view_as_real(h0 - hd)
    n = len(users)

    def mu_split(th_, u, hd_r, hr_, dxy, psi):
        uu = (u + torch.cat([dxy, torch.zeros(1)]))[None]
        hd_u = direct(cfg, uu)[0]
        has = (torch.view_as_complex(hd_r).abs().sum() > 0).to(hd_u.real.dtype)
        h = torch.exp(1j * psi[2]) * torch.view_as_complex(hr_) + has * torch.exp(1j * psi[1]) * hd_u
        return rm.rsrp_mean_phase(cfg, th_, uu, h[None], psi[:1])[0]

    def mu_amp(th_, u, hd_r, hr_, dxy, psi):
        uu = (u + torch.cat([dxy, torch.zeros(1)]))[None]
        hd_u = direct(cfg, uu)[0]
        has = (torch.view_as_complex(hd_r).abs().sum() > 0).to(hd_u.real.dtype)
        h = (10 ** psi[3]) * torch.exp(1j * psi[2]) * torch.view_as_complex(hr_)
        h = h + has * (10 ** psi[4]) * torch.exp(1j * psi[1]) * hd_u
        return rm.rsrp_mean_phase(cfg, th_, uu, h[None], psi[:1])[0]

    def mu_scale(th_, u, hr_, dxy, psi):
        uu = (u + torch.cat([dxy, torch.zeros(1)]))[None]
        h = (10 ** psi[1]) * torch.view_as_complex(hr_)
        return rm.rsrp_mean_phase(cfg, th_, uu, h[None], psi[:1])[0]

    if kind == "scale":
        f, arg = mu_scale, (th, users, torch.view_as_real(h0), torch.zeros(n, 2), torch.zeros(n, 2))
        jac = torch.func.vmap(torch.func.jacfwd(f, argnums=(0, 3, 4)), in_dims=(None, 0, 0, 0, 0))
        Jt, Jd, Jp = jac(*arg)
    elif kind == "split":
        f = mu_split
        arg = (th, users, torch.view_as_real(hd), hr, torch.zeros(n, 2), torch.zeros(n, 3))
        jac = torch.func.vmap(torch.func.jacfwd(f, argnums=(0, 4, 5)), in_dims=(None, 0, 0, 0, 0, 0))
        Jt, Jd, Jp = jac(*arg)
    else:
        f = mu_amp
        arg = (th, users, torch.view_as_real(hd), hr, torch.zeros(n, 2), torch.zeros(n, 5))
        jac = torch.func.vmap(torch.func.jacfwd(f, argnums=(0, 4, 5)), in_dims=(None, 0, 0, 0, 0, 0))
        Jt, Jd, Jp = jac(*arg)
    sig = torch.full(Jt.shape[:2], cfg.sigma_e)
    return schur(Jt, torch.cat([Jd, Jp], -1), sig)


def crb():
    cfg, th0, U, H0, los = setup("B")
    rng = np.random.default_rng(14)
    pools = {"LOS": np.where(los)[0], "NLOS": np.where(~los)[0]}
    sets = {k: [rng.choice(v, N, replace=False) for _ in range(SETS)] for k, v in pools.items()}
    lines = []
    t0 = time.time()
    for kind in ("split", "amp", "scale"):
        for name in ("LOS", "NLOS"):
            Js = [batches(cfg, th0, U[s], H0[s], los[s], kind).detach() for s in sets[name]]
            med, lo, hi, nbad = crb_stat(torch.stack(Js).numpy(), range(SETS))
            line = f"{kind:5s} {name}: median sqrt CRB d {med:.2f} m [{lo:.2f}, {hi:.2f}]  singular {nbad}/{SETS}"
            print(line, flush=True)
            lines.append(line)
            if kind == "split" and name == "LOS" and abs(med - 2.05) > 0.03:
                raise SystemExit("split-phase bound does not match the published 2.05 m")
    lines.append(f"crb wall {time.time() - t0:.0f}s")
    open(os.path.join(ART, "step19_amp_crb.txt"), "w").write("\n".join(lines) + "\n")


def fit_scale(cfg, r, users_hat, h0, th_init):
    """MAP with one log10 amplitude per user on the stored h0. a = 2 tanh(z), so |a| < 2."""
    n = len(users_hat)
    with torch.no_grad():
        Jr = rm.fisher_robust(cfg, th_init, users_hat, h0, sigma_u=SIGMA_U).sum(0)
        cap = torch.tensor([5.0, math.radians(2.0), 3.0, 1.0])
        S = torch.minimum(1 / torch.sqrt(torch.diagonal(Jr).clamp_min(1e-12)), cap)
    zt = torch.zeros(4, requires_grad=True)
    zd = torch.zeros(n, 2, requires_grad=True)
    za = torch.zeros(n, requires_grad=True)
    with torch.no_grad():
        grid = torch.arange(16) * (2 * math.pi / 16)
        cost_g = torch.stack([
            ((r - rm.rsrp_mean_phase(cfg, th_init, users_hat, h0, g.repeat(n))) ** 2).sum(1)
            for g in grid
        ], 1)
        psi0 = grid[cost_g.argmin(1)]
    psi = psi0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([zt, zd, psi, za], lr=1, max_iter=600, tolerance_grad=1e-9,
                            tolerance_change=1e-14, history_size=50, line_search_fn="strong_wolfe")
    sig2 = cfg.sigma_e ** 2

    def cost():
        opt.zero_grad()
        th = th_init + S * zt
        dxy = SIGMA_U * zd
        amp = 2 * torch.tanh(za)
        u = users_hat + torch.cat([dxy, torch.zeros(n, 1)], 1)
        h = (10 ** amp)[:, None] * h0
        mu = rm.rsrp_mean_phase(cfg, th, u, h, psi)
        c = ((r - mu) ** 2).sum() / (2 * sig2) + (zd ** 2).sum() / 2
        c.backward()
        return c

    try:
        opt.step(cost)
    except RuntimeError:
        return float("inf"), None
    with torch.no_grad():
        th = th_init + S * zt
        dxy = SIGMA_U * zd
        amp = 2 * torch.tanh(za)
        u = users_hat + torch.cat([dxy, torch.zeros(n, 1)], 1)
        h = (10 ** amp)[:, None] * h0
        mu = rm.rsrp_mean_phase(cfg, th, u, h, psi)
        c = (((r - mu) ** 2).sum() / (2 * sig2) + (zd ** 2).sum() / 2).item()
    return c, th.detach()


def scale_crb_at(cfg, th, users, h0):
    J = batches(cfg, th, users, h0, np.ones(len(users), dtype=bool), "scale")
    M = J.sum(0).detach().numpy()
    ev = np.linalg.eigvalsh(M)
    if ev[0] <= 1e-8:
        return np.nan
    return float(np.linalg.inv(M)[0, 0])


def map_trial(task):
    torch.set_num_threads(1)
    name, sid, idx, seed = task
    cfg, th0, U, H0, _ = setup("B")
    cfg.sigma_e = 1.0
    u, h0 = U[idx], H0[idx]
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    th_true = th0.clone()
    th_true[0] += rng.uniform(-6, 6)
    th_true[1] += math.radians(rng.uniform(-2, 2))
    mu = rm.rsrp_mean_phase(cfg, th_true, u, h0, torch.zeros(len(idx)))
    r = mu + cfg.sigma_e * torch.randn(mu.shape, generator=gen)
    u_hat = u + torch.cat([SIGMA_U * torch.randn(len(idx), 2, generator=gen), torch.zeros(len(idx), 1)], 1)
    best = (float("inf"), None)
    for dd in (0.0, -3.0, 3.0, -6.0, 6.0):
        for eps in (3.0, 8.0):
            init = th0.clone()
            init[0] += dd
            init[2], init[3] = eps, -1.5
            c, th = fit_scale(cfg, r, u_hat, h0, init)
            if c < best[0]:
                best = (c, th)
    err = (best[1] - th_true).numpy() if best[1] is not None else np.full(4, np.nan)
    return name, sid, seed, err, scale_crb_at(cfg, th_true, u, h0)


def map_run():
    from multiprocessing import get_context
    _, _, _, _, los = setup("B")
    pools = {"LOS": np.where(los)[0], "NLOS": np.where(~los)[0]}
    rng = np.random.default_rng(19)
    tasks, seed = [], 190000
    for name, pool in pools.items():
        sets = [rng.choice(pool, N, replace=False) for _ in range(30)]
        for sid, idx in enumerate(sets):
            for _ in range(4):
                seed += 1
                tasks.append((name, sid, idx, seed))
    t0 = time.time()
    with get_context("fork").Pool(WORKERS) as p:
        out = p.map(map_trial, tasks, chunksize=1)
    lines = [f"scale-amplitude MAP, sigma_e=1 dB, 30 sets x 4 draws, |log10 amp| < 2, wall {time.time() - t0:.0f}s"]
    for name in ("LOS", "NLOS"):
        sub = [r for r in out if r[0] == name]
        err = np.array([r[3] for r in sub])
        crb = np.array([r[4] for r in sub])
        ok = np.isfinite(err[:, 0]) & np.isfinite(crb)
        rmse = np.sqrt(np.nanmean(err[:, 0] ** 2))
        bound = np.sqrt(np.nanmean(crb[np.isfinite(crb)]))
        far = np.mean(np.abs(err[ok, 0]) > 3 * np.sqrt(crb[ok])) if ok.any() else float("nan")
        lines.append(
            f"{name}: d RMSE {rmse:.2f} m / sqrt(mean scale-CRB) {bound:.2f}; "
            f"singular bounds {int((~np.isfinite(crb)).sum())}; "
            f"|d err| > 3 sqrt(CRB) in {100 * far:.0f}%  [{ok.sum()} fits]"
        )
    text = "\n".join(lines)
    print(text, flush=True)
    open(os.path.join(ART, "step19_amp_map.txt"), "w").write(text + "\n")
    np.savez_compressed(
        os.path.join(ART, "step19_amp_map.npz"),
        name=np.array([r[0] for r in out]),
        err=np.array([r[3] for r in out]),
        crb=np.array([r[4] for r in out]),
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "crb"
    if mode == "crb":
        crb()
    elif mode == "map":
        map_run()
    else:
        raise SystemExit("usage: step19_amp.py crb|map")
