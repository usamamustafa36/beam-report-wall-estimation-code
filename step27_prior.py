"""Prior-aware four-beam and eight-beam bounds and fits.

The offset prior is normal with variance A**2/3, the variance of U[-A, A].
The angle prior is normal with the variance of U[-2, 2] deg. A uniform prior
has no Van Trees term, so the bound adds the information of this normal prior
to the average robust information of the report.

The report is the strongest noise-free beam and the other selected beams as
differentials. Beam noise is independent N(0, sigma_e**2) on each dB power.
The differential then has variance 2 sigma_e**2 and covariance -sigma_e**2
with the strongest beam. Sheppard's term adds 1/12 on the strongest beam and
4/12 on each 2 dB differential. Differentials are clipped to [-30, 0] dB in
the simulated report. The fit is MAP under this likelihood and the normal prior.

Modes:
  python3 step27_prior.py check
  python3 step27_prior.py bound
  python3 step27_prior.py fit
"""
import math
import os
import sys
import time
import numpy as np
import torch
import rsrp_model as rm
import wideband as wb
from step4_mle import setup
from step14_complete import schur
from step17_fourbeam_map import information as information_old

RSRP_NB = rm.rsrp_mean_phase
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SIGMA_U = 1.0
SIGMA_E = 1.0
PHI_HALF = math.radians(2.0)
VAR_PHI = PHI_HALF ** 2 / 3.0
WORKERS = int(os.environ.get("WORKERS", "14"))
BOUND_TXT = os.path.join(ART, "step27_prior_bound.txt")
FIT_PARTIAL = os.path.join(ART, "step27_prior_fit_partial.npz")
FIT_TXT = os.path.join(ART, "step27_prior_fit.txt")


def prior_rmse(half):
    return half / math.sqrt(3.0)


def report_precision(se, n_beams, dtype):
    """Precision of (strongest, differentials) after independent beam noise and Sheppard."""
    eye = torch.eye(n_beams, dtype=dtype)
    mix = eye.clone()
    mix[1:, 0] = -1.0
    cov = (se ** 2) * (mix @ mix.T)
    q = torch.full((n_beams,), 4.0 / 12.0, dtype=dtype)
    q[0] = 1.0 / 12.0
    cov = cov + torch.diag(q)
    chol = torch.linalg.cholesky(cov)
    prec = torch.cholesky_inverse(chol)
    # Sigma^{-1/2} via the inverse Cholesky factor, so whitened noise has identity covariance.
    half = torch.linalg.solve_triangular(chol, torch.eye(n_beams, dtype=dtype), upper=False)
    return prec, half, cov


def prior_info(var_d, dtype):
    jp = torch.zeros(4, 4, dtype=dtype)
    jp[0, 0] = 1.0 / var_d
    jp[1, 1] = 1.0 / VAR_PHI
    return jp


def jacobians(cfg, th, users, h0, n_beams):
    """Unwhitened derivatives of the reduced report. Beams fixed from the noise-free mean."""
    def mu(th_, u, hr, dxy, psi):
        uu = (u + torch.cat([dxy, torch.zeros(1)]))[None]
        return rm.rsrp_mean_phase(cfg, th_, uu, torch.view_as_complex(hr)[None], psi[:1])[0]

    n = users.shape[0]
    jac = torch.func.vmap(torch.func.jacfwd(mu, argnums=(0, 3, 4)), in_dims=(None, 0, 0, 0, 0))
    Jt, Jd, Jp = jac(th, users, torch.view_as_real(h0), torch.zeros(n, 2), torch.zeros(n, 1))
    with torch.no_grad():
        mu0 = rm.rsrp_mean_phase(cfg, th, users, h0, torch.zeros(n))
        pick = mu0.argsort(dim=1, descending=True)[:, :n_beams]

    def take(J):
        js = J[torch.arange(n), pick[:, 0]]
        rows = [js]
        for i in range(1, n_beams):
            rows.append(J[torch.arange(n), pick[:, i]] - js)
        return torch.stack(rows, 1)

    return take(Jt), take(Jd), take(Jp), pick


def total_information(cfg, th, users, h0, n_beams):
    Jt, Jd, Jp, pick = jacobians(cfg, th, users, h0, n_beams)
    _, white, _ = report_precision(cfg.sigma_e, n_beams, Jt.dtype)
    Jt = torch.einsum("ab,nbp->nap", white, Jt)
    Jn = torch.einsum("ab,nbp->nap", white, torch.cat([Jd, Jp], -1))
    sig = torch.ones(Jt.shape[:2], dtype=Jt.dtype)
    return schur(Jt, Jn, sig).sum(0), pick


def dd_variance(M):
    ev = np.linalg.eigvalsh(M)
    if ev[0] <= 1e-8:
        return np.nan
    return float(np.linalg.inv(M)[0, 0])


def quantise_report(y):
    """1 dB on the strongest beam, 2 dB differentials clipped to [-30, 0]."""
    strong = torch.round(y[:, :1]).clamp(-140.0, -44.0)
    diff = (torch.round(y[:, 1:] / 2.0) * 2.0).clamp(-30.0, 0.0)
    return torch.cat([strong, diff], 1)


def make_report(cfg, th, users, h0, n_beams, gen):
    mu = rm.rsrp_mean_phase(cfg, th, users, h0, torch.zeros(len(users)))
    pick = mu.argsort(dim=1, descending=True)[:, :n_beams]
    noisy = mu + cfg.sigma_e * torch.randn(mu.shape, generator=gen)
    n = len(users)
    strong = noisy[torch.arange(n), pick[:, 0]]
    cols = [strong]
    for i in range(1, n_beams):
        cols.append(noisy[torch.arange(n), pick[:, i]] - strong)
    raw = torch.stack(cols, 1)
    clipped = (raw[:, 1:] <= -30.0).sum().item()
    return quantise_report(raw), pick, int(clipped), int(raw[:, 1:].numel())


def reduced_mean(cfg, th, users, h0, psi, pick):
    mu = rm.rsrp_mean_phase(cfg, th, users, h0, psi)
    n = len(users)
    strong = mu[torch.arange(n), pick[:, 0]]
    cols = [strong]
    for i in range(1, pick.shape[1]):
        cols.append(mu[torch.arange(n), pick[:, i]] - strong)
    return torch.stack(cols, 1)


def fit_prior(cfg, r, users_hat, h0, pick, th_start, d_map, phi_map, var_d, prec, nphase=16):
    n = len(users_hat)
    J, _ = total_information(cfg, th_start, users_hat, h0, pick.shape[1])
    J = J.detach()
    cap = torch.tensor([5.0, PHI_HALF, 3.0, 1.0], dtype=J.dtype)
    scale = torch.minimum(1 / torch.sqrt(torch.diagonal(J).clamp_min(1e-12)), cap)
    zt = torch.zeros(4, dtype=scale.dtype, requires_grad=True)
    zd = torch.zeros(n, 2, dtype=scale.dtype, requires_grad=True)
    with torch.no_grad():
        grid = torch.arange(nphase) * (2 * math.pi / nphase)
        cols = []
        for g in grid:
            y = reduced_mean(cfg, th_start, users_hat, h0, g.repeat(n), pick)
            e = r - y
            cols.append(torch.einsum("nk,kl,nl->n", e, prec, e))
        psi0 = grid[torch.stack(cols, 1).argmin(1)]
    psi = psi0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([zt, zd, psi], lr=1, max_iter=400, tolerance_grad=1e-8,
                            tolerance_change=1e-12, history_size=40, line_search_fn="strong_wolfe")

    def theta():
        return th_start + scale * zt

    def cost():
        opt.zero_grad()
        th = theta()
        dxy = SIGMA_U * zd
        u = users_hat + torch.cat([dxy, torch.zeros(n, 1)], 1)
        y = reduced_mean(cfg, th, u, h0, psi, pick)
        e = r - y
        nll = 0.5 * torch.einsum("nk,kl,nl->", e, prec, e)
        pen = (th[0] - d_map) ** 2 / (2 * var_d) + (th[1] - phi_map) ** 2 / (2 * VAR_PHI)
        c = nll + (zd ** 2).sum() / 2 + pen
        c.backward()
        return c

    try:
        opt.step(cost)
    except RuntimeError:
        return None, float("inf")
    th = theta().detach()
    with torch.no_grad():
        dxy = SIGMA_U * zd.detach()
        u = users_hat + torch.cat([dxy, torch.zeros(n, 1)], 1)
        y = reduced_mean(cfg, th, u, h0, psi.detach(), pick)
        e = r - y
        nll = 0.5 * torch.einsum("nk,kl,nl->", e, prec, e)
        pen = (th[0] - d_map) ** 2 / (2 * var_d) + (th[1] - phi_map) ** 2 / (2 * VAR_PHI)
        c = float(nll + (zd.detach() ** 2).sum() / 2 + pen)
    return th, c


def prepare(kind):
    """kind is 'nb' (stored h0) or 'wb' (20 MHz, direct path plus wall, h0 ignored)."""
    cfg, th0, U, H0, los = setup("B")
    cfg.sigma_e = SIGMA_E
    if kind == "wb":
        wb.install(20.0)
    else:
        rm.rsrp_mean_phase = RSRP_NB
    return cfg, th0, U, H0, los


def bound_cell(kind, n_beams, n_users, half, n_sets=24, n_draw=4, seed=27000):
    torch.set_num_threads(1)
    cfg, th0, U, H0, los = prepare(kind)
    var_d = half ** 2 / 3.0
    rng = np.random.default_rng(seed + int(half * 10) + n_beams * 100 + n_users)
    lines = []
    for name, mask in (("LOS", los), ("NLOS", ~los)):
        if kind == "wb":
            cfg.use_direct = name == "LOS"
        pool = np.where(mask)[0]
        Js, cond = [], []
        for s in range(n_sets):
            idx = rng.choice(pool, n_users, replace=False)
            users, h0 = U[idx], H0[idx]
            if kind == "wb":
                h0 = torch.zeros_like(h0)
            for _ in range(n_draw):
                th = th0.clone()
                th[0] = th0[0] + rng.normal(0.0, math.sqrt(var_d))
                th[1] = th0[1] + rng.normal(0.0, math.sqrt(VAR_PHI))
                J, _ = total_information(cfg, th, users, h0, n_beams)
                M = (J + prior_info(var_d, J.dtype)).detach().numpy()
                Js.append(J.detach().numpy())
                cond.append(dd_variance(M))
        mean_J = np.mean(Js, 0) + prior_info(var_d, torch.float64).numpy()
        vt = dd_variance(mean_J)
        c = np.array(cond, dtype=float)
        ok = c[np.isfinite(c)]
        std = np.sqrt(ok)
        vt_m = math.sqrt(vt) if np.isfinite(vt) and vt > 0 else float("nan")
        lines.append(
            f"bound {kind} beams={n_beams} N={n_users} A={half:g} {name} "
            f"prior={prior_rmse(half):.3f} VT={vt_m:.3f} "
            f"med={np.median(std):.3f} rms={math.sqrt(np.mean(ok)):.3f} "
            f"p10={np.percentile(std, 10):.3f} p90={np.percentile(std, 90):.3f} "
            f"n={len(ok)} singular={int(np.isnan(c).sum())}"
        )
    return lines


def bound():
    jobs = []
    for n_users in (10, 20, 50, 100, 200):
        for n_beams in (4, 8):
            for kind in ("nb", "wb"):
                jobs.append((kind, n_beams, n_users, 6.0))
    for half in (3.0, 10.0):
        for n_beams in (4, 8):
            for kind in ("nb", "wb"):
                jobs.append((kind, n_beams, 10, half))
    from multiprocessing import get_context
    t0 = time.time()
    lines = [f"Van Trees cells, sigma_e={SIGMA_E} dB, angle prior std {math.degrees(math.sqrt(VAR_PHI)):.3f} deg"]
    with get_context("fork").Pool(WORKERS) as pool:
        for block in pool.starmap(bound_cell, jobs, chunksize=1):
            lines.extend(block)
            print(block[-1], flush=True)
    lines.append(f"bound seconds {time.time() - t0:.0f}")
    text = "\n".join(lines) + "\n"
    open(BOUND_TXT, "w").write(text)
    print(text, flush=True)


def fit_trial(task):
    torch.set_num_threads(1)
    kind, n_beams, n_users, half, name, sid, idx, seed = task
    cfg, th0, U, H0, _ = prepare(kind)
    if kind == "wb":
        cfg.use_direct = name == "LOS"
    var_d = half ** 2 / 3.0
    users, h0 = U[idx], H0[idx]
    if kind == "wb":
        h0 = torch.zeros_like(h0)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(int(seed))
    th_true = th0.clone()
    th_true[0] = th0[0] + rng.normal(0.0, math.sqrt(var_d))
    th_true[1] = th0[1] + rng.normal(0.0, math.sqrt(VAR_PHI))
    r, pick, n_clip, n_diff = make_report(cfg, th_true, users, h0, n_beams, gen)
    u_hat = users + torch.cat([
        SIGMA_U * torch.randn(len(idx), 2, generator=gen), torch.zeros(len(idx), 1)], 1)
    J, _ = total_information(cfg, th_true, users, h0, n_beams)
    bcrb = dd_variance((J + prior_info(var_d, J.dtype)).detach().numpy())
    prec, _, _ = report_precision(cfg.sigma_e, n_beams, r.dtype)
    best_c, best_th = float("inf"), None
    for dd in (0.0, -0.5 * half, 0.5 * half, -0.9 * half, 0.9 * half):
        for eps in (3.0, 8.0):
            init = th0.clone()
            init[0] = th0[0] + dd
            init[2], init[3] = eps, -1.5
            th, c = fit_prior(cfg, r, u_hat, h0, pick, init, th0[0], th0[1], var_d, prec)
            if th is not None and c < best_c:
                best_c, best_th = c, th
    err = (best_th - th_true).numpy() if best_th is not None else np.full(4, np.nan)
    return (kind, n_beams, n_users, half, name, sid, seed, err, bcrb, n_clip, n_diff)


def fit_tasks():
    """120 trials per class for the headline cells, 48 for the N=50 cell."""
    cfg, th0, U, H0, los = prepare("nb")
    del cfg, th0, U, H0
    pools = {"LOS": np.where(los)[0], "NLOS": np.where(~los)[0]}
    specs = []
    for half in (3.0, 6.0, 10.0):
        specs.append(("nb", 4, 10, half, 24, 5))
    specs.append(("nb", 8, 10, 6.0, 24, 5))
    specs.append(("wb", 4, 10, 6.0, 24, 5))
    specs.append(("nb", 4, 50, 6.0, 16, 3))
    specs.append(("nb", 4, 100, 6.0, 16, 3))
    specs.append(("nb", 4, 200, 6.0, 16, 3))
    tasks, seed = [], 271000
    rng = np.random.default_rng(27)
    for kind, n_beams, n_users, half, n_sets, n_draws in specs:
        for name, pool in pools.items():
            sets = [rng.choice(pool, n_users, replace=False) for _ in range(n_sets)]
            for sid, idx in enumerate(sets):
                for _ in range(n_draws):
                    seed += 1
                    tasks.append((kind, n_beams, n_users, half, name, sid, idx, seed))
    return tasks


def save_fit(rows):
    if not rows:
        return
    np.savez_compressed(
        FIT_PARTIAL,
        kind=np.array([r[0] for r in rows]),
        beams=np.array([r[1] for r in rows]),
        users=np.array([r[2] for r in rows]),
        half=np.array([r[3] for r in rows], dtype=float),
        name=np.array([r[4] for r in rows]),
        sid=np.array([r[5] for r in rows]),
        seed=np.array([r[6] for r in rows]),
        err=np.stack([r[7] for r in rows]),
        bcrb=np.array([r[8] for r in rows], dtype=float),
        nclip=np.array([r[9] for r in rows]),
        ndiff=np.array([r[10] for r in rows]),
    )


def summarise_fit(rows):
    lines = []
    keys = sorted(set((r[0], r[1], r[2], r[3], r[4]) for r in rows))
    rng = np.random.default_rng(1)
    for key in keys:
        sub = [r for r in rows if (r[0], r[1], r[2], r[3], r[4]) == key]
        err = np.stack([r[7][0] for r in sub])
        bcrb = np.array([r[8] for r in sub], dtype=float)
        sid = np.array([r[5] for r in sub])
        ok = np.isfinite(err)
        rmse = math.sqrt(np.mean(err[ok] ** 2))
        sets = np.unique(sid[ok])
        boot = []
        for _ in range(2000):
            pick = rng.choice(sets, len(sets), replace=True)
            m = np.concatenate([np.where(sid == s)[0] for s in pick])
            boot.append(math.sqrt(np.mean(err[m] ** 2)))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        prior = prior_rmse(key[3])
        clip = sum(r[9] for r in sub) / max(1, sum(r[10] for r in sub))
        far = np.mean(np.abs(err[ok]) > 3 * np.sqrt(bcrb[ok])) if np.isfinite(bcrb[ok]).all() else float("nan")
        lines.append(
            f"fit {key[0]} beams={key[1]} N={key[2]} A={key[3]:g} {key[4]} "
            f"prior={prior:.3f} RMSE={rmse:.3f} [{lo:.3f}, {hi:.3f}] "
            f"gain={prior - rmse:.3f} max={np.max(np.abs(err[ok])):.2f} "
            f"clip={clip:.4f} far={far:.3f} n={int(ok.sum())} fail={int((~ok).sum())}"
        )
    return lines


def fit():
    from multiprocessing import get_context
    tasks = fit_tasks()
    rows = []
    if os.path.exists(FIT_PARTIAL):
        z = np.load(FIT_PARTIAL, allow_pickle=True)
        rows = list(zip(
            z["kind"], z["beams"], z["users"], z["half"], z["name"], z["sid"], z["seed"],
            z["err"], z["bcrb"], z["nclip"], z["ndiff"]))
        # npz strings need normalising
        rows = [(str(a), int(b), int(c), float(d), str(e), int(f), int(g), h, float(i), int(j), int(k))
                for a, b, c, d, e, f, g, h, i, j, k in rows]
        have = set(int(r[6]) for r in rows)
        tasks = [t for t in tasks if t[-1] not in have]
        print(f"resume {len(rows)} done, {len(tasks)} left", flush=True)
    t0 = time.time()
    if tasks:
        with get_context("fork").Pool(WORKERS) as pool:
            for k, row in enumerate(pool.imap_unordered(fit_trial, tasks, chunksize=1), 1):
                rows.append(row)
                if k % 12 == 0 or k == len(tasks):
                    save_fit(rows)
                    print(f"fit {k}/{len(tasks)} new, {len(rows)} stored, {time.time() - t0:.0f}s", flush=True)
    lines = summarise_fit(rows)
    lines.append(f"fit seconds {time.time() - t0:.0f}")
    text = "\n".join(lines) + "\n"
    open(FIT_TXT, "w").write(text)
    save_fit(rows)
    print(text, flush=True)


def check():
    cfg, th0, U, H0, los = setup("B")
    cfg.sigma_e = SIGMA_E
    idx = np.where(los)[0][:10]
    users, h0 = U[idx], H0[idx]
    J_old, pick = information_old(cfg, th0, users, h0)
    J_new, pick2 = total_information(cfg, th0, users, h0, 4)
    # Rebuild the published diagonal-Sheppard information for one set and compare to step17.
    Jt, Jd, Jp, _ = jacobians(cfg, th0, users, h0, 4)
    std = torch.tensor([
        math.sqrt(SIGMA_E ** 2 + 1 / 12),
        math.sqrt(SIGMA_E ** 2 + 4 / 12),
        math.sqrt(SIGMA_E ** 2 + 4 / 12),
        math.sqrt(SIGMA_E ** 2 + 4 / 12),
    ], dtype=Jt.dtype)
    sig = std.repeat(len(idx), 1)
    J_diag = schur(Jt, torch.cat([Jd, Jp], -1), sig).sum(0)
    d_old = dd_variance(J_old.detach().sum(0).numpy())
    d_diag = dd_variance(J_diag.detach().numpy())
    d_new = dd_variance(J_new.detach().numpy())
    print(f"one LOS set sqrt CRB old {math.sqrt(d_old):.4f} diag-copy {math.sqrt(d_diag):.4f} correlated {math.sqrt(d_new):.4f}")
    if abs(d_old - d_diag) / d_old > 1e-4:
        raise SystemExit("diagonal copy does not match step17")
    # covariance Monte Carlo
    prec, white, cov = report_precision(SIGMA_E, 4, torch.float64)
    gen = torch.Generator().manual_seed(0)
    e = torch.randn(20000, 4, generator=gen, dtype=torch.float64)
    mix = torch.eye(4, dtype=torch.float64)
    mix[1:, 0] = -1
    z = SIGMA_E * e @ mix.T
    emp = torch.cov(z.T)
    # emp excludes Sheppard; cov includes it
    q = torch.diag(torch.tensor([1 / 12, 4 / 12, 4 / 12, 4 / 12], dtype=torch.float64))
    print("cov latent max abs", float((emp - (cov - q)).abs().max()))
    print("pick match", bool(torch.equal(pick, pick2)))
    print("check ok")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if mode == "check":
        check()
    elif mode == "bound":
        bound()
    elif mode == "fit":
        fit()
    else:
        raise SystemExit("usage: step27_prior.py check|bound|fit")
