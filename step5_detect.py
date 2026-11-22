"""Step 5: detecting a stale twin from wide-beam reports.

Facade B, 3.5 GHz, material known (the scene's, marble). The twin holds the wall at (d0, phi0). Truth: wall
moved by Delta along its normal, towards the BS (d -> d + Delta). Test statistic
    T = 2 [ min_{nuisances} C(d0, phi0) - min_{d, phi, nuisances} C(d, phi) ],
nuisances = per-user position offsets (prior 1 m) and wall-path phases; sigma_e = 1 dB.
Nuisances are searched globally per user at a fixed wall (see statistic()); H1 starts at the map
and at d +- 3, 6 m.
  * threshold: empirical 99th percentile of T under no change (P_FA = 1%), compared with chi2_2;
  * P_D vs Delta for N in {5, 10, 20}, LOS and NLOS users;
  * realism: T under no change when the reports come from the full Sionna channel (all paths,
    incl. two-bounce facade paths, full polarisation; e2e users; the twin's known paths come from
    the same Sionna run), with the model-calibrated
    threshold -> the false-alarm rate that model mismatch alone causes.
Output: artifacts/step5_detect.npz, artifacts/step5_detect.txt
"""
import math, os, sys, time
import numpy as np
import torch
import rsrp_model as rm
import scene_data as sd
from step4_mle import fit, setup

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SIGMA_U, NS, DELTAS = 1.0, [5, 10, 20], [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
FACADE = "B"                                    # set by callers (step5b) before the pool forks
D_STARTS = [0.0, -3.0, 3.0, -6.0, 6.0]           # H1 starts: wall offset from the map [m]
H0_TRIALS, H1_TRIALS = 300, 60
N_SIONNA = 10                                   # full-Sionna false-alarm tests at this N
QUICK = len(sys.argv) > 1
if QUICK:
    NS, DELTAS, H0_TRIALS, H1_TRIALS = [10], [0.0, 2.0], 6, 4


def nuisance_grid(cfg, th, r, u_hat, h0, step=0.2, half=2.5):
    """Per-user global search of (offset, phase) at a fixed wall. Given the wall, users are
    independent, and with its phase free the per-user cost is smooth in the offset on the
    beam-width scale (the phase absorbs kappa * L_r, Lemma 1), so a 20 cm grid over +-2.5 sigma_u
    and 16 phases finds the basin; fit() then polishes."""
    g = torch.arange(-half, half + 1e-9, step)
    gx, gy = [a.reshape(-1) for a in torch.meshgrid(g, g, indexing="ij")]
    m, ps = len(gx), torch.arange(16) * (2 * math.pi / 16)
    prior = (gx ** 2 + gy ** 2) / (2 * SIGMA_U ** 2)
    dxy, psi = torch.zeros(len(u_hat), 2), torch.zeros(len(u_hat))
    with torch.no_grad():
        for n in range(len(u_hat)):
            u = u_hat[n].repeat(m, 1)
            u[:, 0] += gx
            u[:, 1] += gy
            c = torch.stack([((r[n] - rm.rsrp_mean_phase(cfg, th, u, h0[n].repeat(m, 1), p.repeat(m))) ** 2).sum(1)
                             / (2 * cfg.sigma_e ** 2) + prior for p in ps], 1)        # (m, 16)
            k = int(c.argmin())
            dxy[n] = torch.stack([gx[k // 16], gy[k // 16]])
            psi[n] = ps[k % 16]
    return dxy, psi


def statistic(cfg, th_map, r, u_hat, h0, rng):
    """T = 2 (C0 - C1); see costs()."""
    c0, c1 = costs(cfg, th_map, r, u_hat, h0)
    return 2 * (c0 - c1)


def costs(cfg, th_map, r, u_hat, h0, ret_th=False):
    """(C0, C1) with C1 <= C0 (the H0 set lies inside H1); the material is held at th_map's.
    ret_th: also return the H1 estimate (th_map if no free fit beats the map).

    Position offsets and wall-path phases are multimodal. A local fit at the map wall can miss the
    H0 minimum while a fit with the wall free reaches an equally good basin elsewhere, which inflates T under no change. So H0 starts from a per-user global nuisance search
    and is refitted from the H1 nuisances, H1 starts at the map and at offsets D_STARTS of the wall
    and is polished from a global nuisance search at its estimate."""
    held, freed = (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)
    _, c0, nu0 = fit(cfg, r, u_hat, h0, SIGMA_U, th_map.clone(), False, free=held, ret_nuis=True)
    _, c0g, nu0g = fit(cfg, r, u_hat, h0, SIGMA_U, th_map.clone(), False, free=held,
                       nuis=nuisance_grid(cfg, th_map, r, u_hat, h0), ret_nuis=True)
    if c0g < c0:
        c0, nu0 = c0g, nu0g
    c1, th1, nu1 = c0, th_map.clone(), nu0
    for dd in D_STARTS:
        init = th_map.clone()
        init[0] += dd
        th, c, nu = fit(cfg, r, u_hat, h0, SIGMA_U, init, False, free=freed, nuis=nu0, ret_nuis=True)
        if c < c1:
            c1, th1, nu1 = c, th, nu
    if c1 < c0:
        th, c, nu = fit(cfg, r, u_hat, h0, SIGMA_U, th1.clone(), False, free=freed,
                        nuis=nuisance_grid(cfg, th1, r, u_hat, h0), ret_nuis=True)
        if c < c1:
            c1, th1, nu1 = c, th, nu
        _, c0b = fit(cfg, r, u_hat, h0, SIGMA_U, th_map.clone(), False, free=held, nuis=nu1)
        c0 = min(c0, c0b)
    if ret_th:
        return c0, min(c1, c0), (th1 if c1 < c0 else th_map.clone())
    return c0, min(c1, c0)                          # the H0 set lies inside H1


def trial(task):
    torch.set_num_threads(1)
    kind, cls_name, N, delta, seed = task
    cfg, th0, U, H0, los = setup(FACADE, "3.5")
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    if kind in ("sionna", "sionna1b"):
        pos, h0s, _, hall, hrest = sd.e2e(FACADE, "3.5")                  # same-run Sionna channels
        keep = los[pos] if cls_name == "LOS" else ~los[pos]
        j = rng.choice(np.where(keep)[0], N, replace=False)
        idx = pos[j]
        # complete twin: knows every path but the single-bounce facade path (incl. two-bounce paths
        # via the facade, at the map position); single-bounce twin: knows only paths avoiding the facade
        h_known = (hrest if kind == "sionna" else h0s)[torch.tensor(j)]
        mu = 10 * torch.log10(cfg.ptx * torch.abs(hall[torch.tensor(j)] @ cfg.C.conj().T) ** 2 + cfg.noise)
    else:
        pool = np.where(los if cls_name == "LOS" else ~los)[0]
        idx = rng.choice(pool, N, replace=False)
        th_true = th0.clone()
        th_true[0] += delta                     # n points to the BS side: d + delta moves the wall towards it
        h_known = H0[idx]
        mu = rm.rsrp_mean_phase(cfg, th_true, U[idx], h_known, torch.zeros(N))
    r = rm.measure(cfg, mu, gen)
    u_hat = U[idx] + torch.cat([SIGMA_U * torch.randn(N, 2, generator=gen), torch.zeros(N, 1)], 1)
    return task[:4], statistic(cfg, th0, r, u_hat, h_known, rng)


def main():
    from multiprocessing import Pool
    tasks, seed = [], 9000
    for cls_name in ["LOS", "NLOS"]:
        for N in NS:
            for delta in DELTAS:
                for _ in range(H0_TRIALS if delta == 0 else H1_TRIALS):
                    seed += 1
                    tasks.append(("model", cls_name, N, delta, seed))
            for kind in ("sionna", "sionna1b"):
                for _ in range(H0_TRIALS if N == N_SIONNA else 0):
                    seed += 1
                    tasks.append((kind, cls_name, N, 0.0, seed))
    t0 = time.time()
    with Pool(18) as p:
        out = p.map(trial, tasks, chunksize=1)
    T = {}
    for key, t in out:
        T.setdefault(key, []).append(t)
    lines = [f"chi2_2 99% = 9.21"]
    save = {}
    for cls_name in ["LOS", "NLOS"]:
        for N in NS:
            t0_ = np.array(T[("model", cls_name, N, 0.0)])
            thr = np.percentile(t0_, 99)
            pd = [np.mean(np.array(T[("model", cls_name, N, dl)]) > thr) for dl in DELTAS[1:]]
            line = (f"{cls_name:4s} N={N:2d}: threshold {thr:6.2f} (median T|H0 {np.median(t0_):.2f}); "
                    f"P_D " + " ".join(f"{dl:g}m:{p:.2f}" for dl, p in zip(DELTAS[1:], pd)))
            if N == N_SIONNA:
                ts = np.array(T[("sionna", cls_name, N, 0.0)])
                t1b = np.array(T[("sionna1b", cls_name, N, 0.0)])
                line += (f" | full Sionna, no change: P_FA {np.mean(ts > thr):.2f} with a complete twin (median T {np.median(ts):.1f}), "
                         f"{np.mean(t1b > thr):.2f} with a single-bounce twin (median T {np.median(t1b):.1f})")
                save[f"{cls_name}_{N}_Tsionna"] = ts
                save[f"{cls_name}_{N}_Tsionna1b"] = t1b
            lines.append(line)
            save[f"{cls_name}_{N}_thr"] = thr
            save[f"{cls_name}_{N}_pd"] = np.array(pd)
            save[f"{cls_name}_{N}_T0"] = t0_
    lines.append(f"total {time.time() - t0:.0f}s")
    print("\n".join(lines))
    np.savez_compressed(os.path.join(ART, "step5_detect" + ("_quick" if QUICK else "") + ".npz"),
                        deltas=np.array(DELTAS[1:]), Ns=np.array(NS), **save)
    open(os.path.join(ART, "step5_detect" + ("_quick" if QUICK else "") + ".txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
