"""Step 9 (A2): can wide-beam reports tell the facade's material class?

Facade B. For each true ITU class, reports are generated from the model (h0 + wall path, true
positions, sigma_e = 1 dB). The estimator sees positions with N(0, 1 m^2) errors and, for every
candidate class, fits the wall geometry, position offsets and wall-path phases with the material
held at that class (step4_mle.fit with the material masked). The class with the smallest cost wins.
Confusion matrices for N users in {2, 5, 10, 20}, LOS / NLOS pools, 3.5 and 28 GHz.
Output: artifacts/step9_material.npz, artifacts/step9_material.txt
"""
import math, os, sys, time
import numpy as np
import torch
import rsrp_model as rm
from step4_mle import fit, setup

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
CLASSES = ["concrete", "brick", "plasterboard", "wood", "glass", "marble", "metal"]
NS, TRIALS, SIGMA_U = [2, 5, 10, 20], 20, 1.0
QUICK = len(sys.argv) > 1


def material_theta(th_geo, name, freq):
    a, b, c, d = rm.ITU[name]
    f = freq / 1e9
    return torch.tensor([th_geo[0], th_geo[1], a * f ** b, math.log10(c * f ** d)])


def trial(task):
    torch.set_num_threads(1)
    f, cls_name, truth, N, seed = task
    cfg, th0, U, H0, los = setup("B", f)
    pool = np.where(los if cls_name == "LOS" else ~los)[0]
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    idx = rng.choice(pool, N, replace=False)
    u, h0 = U[idx], H0[idx]
    th_true = material_theta(th0, CLASSES[truth], cfg.freq)
    r = rm.measure(cfg, rm.rsrp_mean_phase(cfg, th_true, u, h0, torch.zeros(N)), gen)
    u_hat = u + torch.cat([SIGMA_U * torch.randn(N, 2, generator=gen), torch.zeros(N, 1)], 1)
    geo = th0[:2] + torch.tensor([2.0 * rng.standard_normal(), math.radians(1.0) * rng.standard_normal()])
    costs = []
    for name in CLASSES:
        _, c = fit(cfg, r, u_hat, h0, SIGMA_U, material_theta(geo, name, cfg.freq), False, free=(1.0, 1.0, 0.0, 0.0))
        costs.append(c)
    return task[:4], int(np.argmin(costs)), np.array(costs)


def main():
    from multiprocessing import Pool
    tasks, seed = [], 5000
    for f in (["3.5"] if QUICK else ["3.5", "28"]):
        for cls_name in ["LOS", "NLOS"]:
            for N in ([10] if QUICK else NS):
                for truth in range(len(CLASSES)):
                    for _ in range(2 if QUICK else TRIALS):
                        seed += 1
                        tasks.append((f, cls_name, truth, N, seed))
    t0 = time.time()
    with Pool(18) as p:
        out = p.map(trial, tasks, chunksize=1)
    conf, lines = {}, []
    for (f, cls_name, truth, N), pred, _ in out:
        conf.setdefault((f, cls_name, N), np.zeros((len(CLASSES), len(CLASSES)), int))[truth, pred] += 1
    for (f, cls_name, N), M in sorted(conf.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        acc = np.trace(M) / M.sum()
        per = " ".join(f"{CLASSES[i][:5]} {M[i, i] / M[i].sum():.2f}" for i in range(len(CLASSES)))
        lines.append(f"{f:>3s} GHz {cls_name:4s} N={N:2d}: accuracy {acc:.2f} | {per}")
    lines.append(f"total {time.time() - t0:.0f}s")
    print("\n".join(lines))
    np.savez_compressed(os.path.join(ART, "step9_material" + ("_quick" if QUICK else "") + ".npz"),
                        classes=np.array(CLASSES), **{f"{f}_{c}_{N}": M for (f, c, N), M in conf.items()})
    open(os.path.join(ART, "step9_material" + ("_quick" if QUICK else "") + ".txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
