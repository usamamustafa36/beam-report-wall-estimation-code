"""Step 4b: does the MAP fit reach the robust bound when its starts carry no information on the truth?

Reviewer M1: in step4_mle.py the true wall is the map wall and the starts lie +-2 m around it, so the
starts act as a prior and the RMSE can fall below the CRB. Here the true wall is moved from the map by
(U[-6, 6] m, U[-2, 2] deg) (as in step10_learned.py), the material is the scene's (marble), and the fit
starts at the map with d offsets in {0, +-3, +-6} m and eps_r in {3, 8}; no start uses the truth.
The robust CRB is evaluated at the true wall. Quantised and unquantised reports share the same noise
draw (paired comparison, reviewer m3). 95% CIs: cluster bootstrap over user sets.
Facade B (default) or E (argv), 3.5 GHz, N = 10, sigma_u = 1 m. Reuses step4_mle.fit/setup unchanged.
Output: artifacts/step4b_mle_far.npz, artifacts/step4b_mle_far.txt
"""
import math, os, sys, time
import numpy as np
import torch
import rsrp_model as rm
from step4_mle import fit, setup

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
FACADE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("B", "E") else "B"
QUICK = "quick" in sys.argv[1:]
SIGMA_U, N_USERS = 1.0, 10
SETS, DRAWS = (2, 2) if QUICK else (30, 4)
SIGMA_E = [1.0] if QUICK else [0.25, 1.0, 4.0]
D_STARTS, EPS_STARTS = [0.0, -3.0, 3.0, -6.0, 6.0], [3.0, 8.0]
B_BOOT = 2000


def best_fit(cfg, r, u_hat, h0, th_map, quantised):
    best = (float("inf"), None)
    for dd in D_STARTS:
        for eps in EPS_STARTS:
            init = th_map.clone()
            init[0] += dd
            init[2], init[3] = eps, -1.5
            th, c = fit(cfg, r, u_hat, h0, SIGMA_U, init, quantised)
            if c < best[0]:
                best = (c, th)
    return best[1]


def trial(task):
    """(class, sigma_e, set id, user idx, seed) -> errors (unquantised[, quantised]) and CRB diag at the truth."""
    torch.set_num_threads(1)
    name, se, sid, idx, seed = task
    cfg, th0, U, H0, _ = setup(FACADE)
    cfg.sigma_e = se
    u, h0 = U[idx], H0[idx]
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    th_true = th0.clone()
    th_true[0] += rng.uniform(-6, 6)
    th_true[1] += math.radians(rng.uniform(-2, 2))
    J = rm.fisher_robust(cfg, th_true, u, h0, sigma_u=SIGMA_U).sum(0).numpy()
    mu = rm.rsrp_mean_phase(cfg, th_true, u, h0, torch.zeros(len(idx)))
    r = mu + se * torch.randn(mu.shape, generator=gen)
    u_hat = u + torch.cat([SIGMA_U * torch.randn(len(idx), 2, generator=gen), torch.zeros(len(idx), 1)], 1)
    out = {"crb": np.diag(np.linalg.inv(J))}
    th = best_fit(cfg, r, u_hat, h0, th0, False)
    out["err"] = (th - th_true).numpy() if th is not None else np.full(4, np.nan)
    if se == 1.0:                                   # paired: same noise draw, 1 dB rounding and clipping
        rq = torch.clamp(torch.round(r), -156.0, -31.0)
        thq = best_fit(cfg, rq, u_hat, h0, th0, True)
        out["err_q"] = (thq - th_true).numpy() if thq is not None else np.full(4, np.nan)
        out["crb_q"] = out["crb"] * (se ** 2 + 1 / 12) / se ** 2
    return task[:3], out


def boot_ci(err, crb, sid, k, rng):
    """Cluster bootstrap over user sets: CI of RMSE and of sqrt(mean CRB) for parameter k."""
    sets = np.unique(sid)
    rm_, bd_ = [], []
    for _ in range(B_BOOT):
        pick = rng.choice(sets, len(sets), replace=True)
        m = np.concatenate([np.where(sid == s)[0] for s in pick])
        rm_.append(np.sqrt(np.nanmean(err[m, k] ** 2)))
        bd_.append(np.sqrt(crb[m, k].mean()))
    return np.percentile(rm_, [2.5, 97.5]), np.percentile(bd_, [2.5, 97.5])


def main():
    from multiprocessing import Pool
    _, _, _, _, los = setup(FACADE)
    pools = {"LOS": np.where(los)[0], "NLOS": np.where(~los)[0]}
    rng = np.random.default_rng(7)
    tasks, seed = [], 40000
    for name, pool in pools.items():
        sets = [rng.choice(pool, N_USERS, replace=False) for _ in range(SETS)]
        for se in SIGMA_E:
            for sid, idx in enumerate(sets):
                for _ in range(DRAWS):
                    seed += 1
                    tasks.append((name, se, sid, idx, seed))
    t0 = time.time()
    with Pool(18 if not QUICK else 4) as p:
        out = p.map(trial, tasks, chunksize=1)
    res = {}
    for (name, se, sid), o in out:
        variants = [("", o["err"], o["crb"])] + ([("_q", o["err_q"], o["crb_q"])] if "err_q" in o else [])
        for tag, e, c in variants:
            d = res.setdefault((name, se, tag), {"err": [], "crb": [], "sid": []})
            d["err"].append(e)
            d["crb"].append(c)
            d["sid"].append(sid)
    lines, save = [], {}
    brng = np.random.default_rng(1)
    for (name, se, tag), d in sorted(res.items()):
        err, crb, sid = np.array(d["err"]), np.array(d["crb"]), np.array(d["sid"])
        ok = np.isfinite(err[:, 0])
        rmse = np.sqrt(np.nanmean(err ** 2, 0))
        bound = np.sqrt(crb.mean(0))
        (rl, rh), (bl, bh) = boot_ci(err, crb, sid, 0, brng)
        out3 = np.mean(np.abs(err[ok, 0]) > 3 * np.sqrt(crb[ok, 0]))
        key = f"{name}_se{se}{tag}"
        save[key + "_err"], save[key + "_crb"], save[key + "_sid"] = err, crb, sid
        lines.append(f"{name:4s} sigma_e={se:4.2f} dB{' quantised' if tag else '          '}: d RMSE {rmse[0]:.2f} m "
                     f"[{rl:.2f}, {rh:.2f}] / bound {bound[0]:.2f} [{bl:.2f}, {bh:.2f}]; phi RMSE "
                     f"{math.degrees(rmse[1]):.2f} / {math.degrees(bound[1]):.2f} deg; |d err| > 3 sqrt(CRB) in "
                     f"{100 * out3:.0f}%  [{ok.sum()} fits, {(~ok).sum()} failed, {len(np.unique(sid))} sets]")
    lines.append(f"total {time.time() - t0:.0f}s")
    print("\n".join(lines))
    suf = ("" if FACADE == "B" else f"_{FACADE}") + ("_quick" if QUICK else "")
    np.savez_compressed(os.path.join(ART, f"step4b_mle_far{suf}.npz"), **save)
    open(os.path.join(ART, f"step4b_mle_far{suf}.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
