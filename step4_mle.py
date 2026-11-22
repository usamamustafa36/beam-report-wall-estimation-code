"""Step 4: does a maximum-a-posteriori fit reach the robust bound?

Facade B, 3.5 GHz, N users from one LOS class. Truth: the scene's material, true user positions.
Reports: model channel (h0 + wall path) at the true positions, Gaussian noise sigma_e in dB,
optionally 1 dB quantisation (and clipping). The estimator sees positions with N(0, sigma_u^2)
errors and fits theta = (d, phi, eps_r, log10 sigma) jointly with a per-user position offset
(Gaussian prior sigma_u) and a per-user wall-path phase, by L-BFGS from several starts drawn
around the map value (d +- 2 m, phi +- 1 deg) with eps_r in {3, 5, 7, 9}, log10 sigma = -1.5.
The fit runs in coordinates whitened by the Fisher information at the start. The start with the
smallest cost wins. RMSE is compared with sqrt(mean CRB) of the robust Fisher information over
the same user sets.
Usage: python3 step4_mle.py [quick]   -> artifacts/step4_mle.npz, artifacts/step4_mle.txt
"""
import math, os, sys, time
import numpy as np
import torch
import rsrp_model as rm
import scene_data as sd

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
QUICK = len(sys.argv) > 1
SIGMA_U, N_USERS = 1.0, 10
SETS, DRAWS, STARTS = (2, 2, 4) if QUICK else (10, 6, 4)
SIGMA_E = [1.0] if QUICK else [0.25, 0.5, 1.0, 2.0, 4.0]
lines = []


def log(s):
    print(s, flush=True)
    lines.append(s)


def setup(F="B", f="3.5"):
    _, cfg, th0, U, H0, los = sd.load(F, f)
    return cfg, th0, U, H0, los


def mu_full(cfg, th, users, h0, dxy, psi):
    u = users + torch.cat([dxy, torch.zeros(len(users), 1)], 1)
    return rm.rsrp_mean_phase(cfg, th, u, h0, psi)


def fit(cfg, r, users_hat, h0, sigma_u, th_init, quantised, free=(1.0, 1.0, 1.0, 1.0), nuis=None, ret_nuis=False, nphase=16):
    """MAP over (theta, offsets, phases) by L-BFGS in whitened coordinates.

    theta = th_init + S z with S = 1/sqrt(diag J) of the robust information at the start (capped),
    so that every coordinate has about unit curvature; offsets are scaled by sigma_u. free masks parameters
    (0 = held at th_init, e.g. a material class under test). nuis = (offsets [m], phases) starts the
    nuisances there instead of at zero offsets and the phase grid. Returns theta_hat, cost
    (and the fitted nuisances if ret_nuis)."""
    N = len(users_hat)
    # Whiten by the robust information (position offsets and wall-path phases eliminated), not the
    # coherent one: the coherent bound on d is ~mm, so a coherent-whitened fit could not move d by
    # metres and stayed at its start (bug found 2026-10-29). Caps keep the scales physical.
    with torch.no_grad():
        Jr = rm.fisher_robust(cfg, th_init, users_hat, h0, sigma_u=sigma_u).sum(0)
        cap = torch.tensor([5.0, math.radians(2.0), 3.0, 1.0])
        S = torch.tensor(free) * torch.minimum(1 / torch.sqrt(torch.diagonal(Jr).clamp_min(1e-12)), cap)
    zt = torch.zeros(4, requires_grad=True)
    zd = torch.zeros(N, 2, requires_grad=True)
    # the wall-path phase absorbs kappa * (path-length error) and can sit anywhere in [-pi, pi):
    # start each user at the best of a 16-point grid (users are independent given theta)
    with torch.no_grad():
        if nuis is not None:
            zd.data = nuis[0].clone() / sigma_u
        grid = torch.arange(nphase) * (2 * math.pi / nphase)
        cost_g = torch.stack([((r - rm.rsrp_mean_phase(cfg, th_init, users_hat, h0, g.repeat(N))) ** 2).sum(1)
                              for g in grid], 1)                              # (N, 16)
        psi0 = grid[cost_g.argmin(1)] if nuis is None else nuis[1].clone()
    psi = psi0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([zt, zd, psi], lr=1, max_iter=600, tolerance_grad=1e-9,
                            tolerance_change=1e-14, history_size=50, line_search_fn="strong_wolfe")
    th = lambda: th_init + S * zt
    dxy = lambda: sigma_u * zd
    sig2 = cfg.sigma_e ** 2 + (1 / 12 if quantised else 0.0)        # quantisation as extra noise

    def cost():
        opt.zero_grad()
        mu = mu_full(cfg, th(), users_hat, h0, dxy(), psi)
        if quantised:
            mu = torch.clamp(mu, -156.0, -31.0)
        c = ((r - mu) ** 2).sum() / (2 * sig2) + (zd ** 2).sum() / 2
        c.backward()
        return c

    try:
        opt.step(cost)
        with torch.no_grad():
            mu = mu_full(cfg, th(), users_hat, h0, dxy(), psi)
            c = (((r - mu) ** 2).sum() / (2 * sig2) + (zd ** 2).sum() / 2).item()
    except RuntimeError:
        c = float("inf")
    if ret_nuis:
        return th().detach(), c, (dxy().detach(), psi.detach())
    return th().detach(), c


def trial(task):
    """One Monte-Carlo trial: (class, sigma_e, quantised, set seed, draw seed) -> error, CRB diag."""
    torch.set_num_threads(1)
    name, se, quantised, idx, seed = task
    cfg, th0, U, H0, _ = setup()
    cfg.sigma_e = se
    u, h0 = U[idx], H0[idx]
    J = rm.fisher_robust(cfg, th0, u, h0, sigma_u=SIGMA_U).sum(0)
    if quantised:
        J = J * se ** 2 / (se ** 2 + 1 / 12)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu0 = rm.rsrp_mean_phase(cfg, th0, u, h0, torch.zeros(len(idx)))
    r = rm.measure(cfg, mu0, gen, quantise=quantised)
    u_hat = u + torch.cat([SIGMA_U * torch.randn(len(idx), 2, generator=gen), torch.zeros(len(idx), 1)], 1)
    best = (float("inf"), None)
    for k in range(STARTS):
        init = th0 + torch.tensor([2.0 * rng.standard_normal(), math.radians(1.0) * rng.standard_normal(), 0, 0])
        init[2], init[3] = [3.0, 5.0, 7.0, 9.0][k % 4], -1.5        # spread the material starts
        th, c = fit(cfg, r, u_hat, h0, SIGMA_U, init, quantised)
        if c < best[0]:
            best = (c, th)
    err = (best[1] - th0).numpy() if best[1] is not None else np.full(4, np.nan)
    return task[:3], err, np.diag(np.linalg.inv(J.numpy()))


def main():
    from multiprocessing import Pool
    _, _, _, _, los = setup()
    pools = {"LOS": np.where(los)[0], "NLOS": np.where(~los)[0]}
    rng = np.random.default_rng(0)
    tasks, seed = [], 1000
    for name, pool in pools.items():
        sets = [rng.choice(pool, N_USERS, replace=False) for _ in range(SETS)]
        for se in SIGMA_E:
            for quantised in ([False] if se != 1.0 else [False, True]):
                for idx in sets:
                    for _ in range(DRAWS):
                        seed += 1
                        tasks.append((name, se, quantised, idx, seed))
    t0 = time.time()
    with Pool(18 if not QUICK else 4) as p:
        out = p.map(trial, tasks, chunksize=1)
    res = {}
    for key, err, crb in out:
        res.setdefault(key, ([], []))
        res[key][0].append(err)
        res[key][1].append(crb)
    save = {}
    for (name, se, q), (err, crb) in res.items():
        err, crb = np.array(err), np.array(crb)
        ok = np.isfinite(err[:, 0])
        e = err[ok]
        rmse = np.sqrt((e ** 2).mean(0))
        med = np.median(np.abs(e), 0)
        bound = np.sqrt(crb.mean(0))
        outl = np.mean(np.abs(e[:, 2]) > 3 * bound[2])
        key = f"{name}_se{se}" + ("_q" if q else "")
        save[key + "_err"], save[key + "_crb"] = err, crb
        log(f"{name:4s} sigma_e={se:4.2f} dB{' quantised' if q else '          '}: "
            f"d RMSE {rmse[0]:.3g} m / bound {bound[0]:.3g} (median |err| {med[0]:.3g}); "
            f"phi RMSE {math.degrees(rmse[1]):.3g} / {math.degrees(bound[1]):.3g} deg; "
            f"eps_r RMSE {rmse[2]:.3g} / {bound[2]:.3g}, |err|>3 bound in {100 * outl:.0f}%  "
            f"[{ok.sum()} fits, {(~ok).sum()} failed]")
    log(f"total {time.time() - t0:.0f}s")
    np.savez_compressed(os.path.join(ART, "step4_mle" + ("_quick" if QUICK else "") + ".npz"), **save)
    with open(os.path.join(ART, "step4_mle" + ("_quick" if QUICK else "") + ".txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
