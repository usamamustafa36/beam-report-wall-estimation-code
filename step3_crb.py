"""Step 3d: Cramer-Rao bounds on the wall (d, phi) and its material (eps_r, sigma) in Munich.

Physical model (P2): ITU slab of the scene material (marble for A, B, E), 0.1 m, TE; theta = (d, phi, eps_r, log10 sigma).
Every other path is known to the twin (h0 from step3_trace.py). sigma_e = 1 dB (the bounds scale
linearly with sigma_e). Pools: users served by the facade, split by LOS to the BS.
Three information models:
  coherent : user positions known exactly (ideal; LOS bounds are then carrier-phase limited)
  robust1  : per-user horizontal position offset with a 1 m prior + unknown wall-path phase
  robust3  : the same with a 3 m prior (GNSS-grade positions)
The robust bounds are the ones to quote; coherent is the ideal limit.
  * CRB vs N: 300 random subsets of N users per pool (median, 10th and 90th percentile).
  * local map: each user pooled with its 4 nearest facade users (N = 5).
  * K wide beams in {8, 16, 32} at N = 10.
  * single NLOS user with a map prior on (d, phi): is the material then identifiable?
Writes artifacts/step3_crb_<F>_<f>GHz.npz and artifacts/step3_crb.txt.
"""
import math, os, sys
import numpy as np
import torch
import rsrp_model as rm
import scene_data as sd

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
NS, DRAWS, SING = [2, 3, 5, 10, 20, 50], 300, 1e12
lines = []


def log(s):
    print(s, flush=True)
    lines.append(s)


def load(F, f, K=16):
    z, cfg, theta, _, _, _ = sd.load(F, f, K)
    return z, z["fval"], cfg, theta


KINDS = {"coherent": "c", "robust1": 1.0, "robust3": 3.0}


def per_user_fisher(cfg, theta, z, v, kind="robust1"):
    U, h0 = torch.tensor(z["users"][v]), torch.tensor(z["h0"][v])
    if KINDS[kind] == "c":
        Jac = rm.jacobian(cfg, theta, U, None, mode="material", h0=h0)           # (N, K, 4)
        return torch.einsum("nkp,nkq->npq", Jac, Jac).numpy() / cfg.sigma_e ** 2
    return rm.fisher_robust(cfg, theta, U, h0, sigma_u=KINDS[kind]).numpy()


def crb_std(J):
    """sqrt(diag(J^-1)) in reporting units (d [m], phi [deg], eps_r, sigma factor), inf if singular."""
    w = np.linalg.eigvalsh(J)
    if w[0] <= w[-1] / SING:
        return np.full(4, np.inf)
    s = np.sqrt(np.diag(np.linalg.inv(J)))
    return np.array([s[0], np.degrees(s[1]), s[2], 10 ** min(s[3], 300.0)])


def subsets(Jn, pool, N, rng):
    out = []
    for _ in range(DRAWS):
        idx = rng.choice(pool, N, replace=False)
        out.append(crb_std(Jn[idx].sum(0)))
    return np.array(out)


def summary(x):
    finite = np.isfinite(x[:, 0])
    if finite.sum() == 0:
        return "singular in every draw"
    q = np.percentile(x[finite], [10, 50, 90], axis=0)
    return (f"d {q[1, 0]:.3g} m [{q[0, 0]:.2g}, {q[2, 0]:.2g}]  phi {q[1, 1]:.3g} deg  "
            f"eps_r {q[1, 2]:.3g} [{q[0, 2]:.2g}, {q[2, 2]:.2g}]  sigma x{q[1, 3]:.3g}  "
            f"(singular {100 * (1 - finite.mean()):.0f}%)")


def main():
    rng = np.random.default_rng(42)
    for F, f in [("B", "3.5"), ("B", "28"), ("A", "3.5"), ("A", "28"), ("E", "3.5"), ("E", "28")]:
        z, v, cfg, theta = load(F, f)
        los = z["los"][v]
        pools = {"LOS": np.where(los)[0], "NLOS": np.where(~los)[0]}
        out = dict(users=z["users"][v], los=los, theta=theta.numpy(), Ns=np.array(NS), bs=z["bs"],
                   facade_tri=z["facade_tri"])
        for kind in KINDS:
            Jn = per_user_fisher(cfg, theta, z, v, kind)
            out[f"Jn_{kind}"] = Jn
            log(f"\n=== facade {F}, {f} GHz, K=16, sigma_e={cfg.sigma_e} dB, {kind}: {v.sum()} users "
                f"({los.sum()} LOS, {(~los).sum()} NLOS) ===")
            for name, pool in pools.items():
                if len(pool) < 2:
                    continue
                for N in NS:
                    if N > len(pool):
                        continue
                    out[f"crb_{kind}_{name}_{N}"] = x = subsets(Jn, pool, N, np.random.default_rng(N))
                    log(f"  {name:4s} N={N:2d}: {summary(x)}")
            rk = np.array([np.linalg.matrix_rank(J, tol=np.abs(J).max() * 1e-10) for J in Jn])
            log(f"  single-user rank: LOS {np.bincount(rk[los], minlength=5)[1:]}, "
                f"NLOS {np.bincount(rk[~los], minlength=5)[1:]} (counts of rank 1..4)")
            if kind != "robust1":
                continue
            Jp = np.diag([1 / 0.5 ** 2, 1 / np.radians(1.0) ** 2, 0, 0])                 # map prior
            if len(pools["NLOS"]):
                sing = np.mean([np.isinf(crb_std(Jn[i] + Jp)[2]) for i in pools["NLOS"]])
                log(f"  one NLOS user + map prior (0.5 m, 1 deg): material singular for {100 * sing:.0f}% of users")
            P = z["users"][v][:, :2]
            nn = np.argsort(((P[:, None] - P[None]) ** 2).sum(-1), 1)[:, :5]
            out["local"] = local = np.array([crb_std(Jn[i].sum(0)) for i in nn])
            log(f"  local N=5 (nearest neighbours): median d {np.median(local[:, 0]):.3g} m, "
                f"eps_r {np.median(local[:, 2]):.3g}; singular {100 * np.mean(~np.isfinite(local[:, 0])):.0f}%")
            if F in "BE":                                               # number of wide beams
                for K in [8, 16, 32]:
                    _, _, cK, _ = load(F, f, K)
                    JK = Jn if K == 16 else per_user_fisher(cK, theta, z, v, kind)
                    for name, pool in pools.items():
                        out[f"crbK{K}_{name}_10"] = x = subsets(JK, pool, 10, np.random.default_rng(7))
                        log(f"  K={K:2d} {name:4s} N=10: {summary(x)}")
        np.savez_compressed(os.path.join(ART, f"step3_crb_{F}_{f}GHz.npz"), **out)
    with open(os.path.join(ART, "step3_crb.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
