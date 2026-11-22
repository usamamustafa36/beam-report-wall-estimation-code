"""M7: 1% GLRT threshold on users served in the unmoved scene and in both moved meshes.

Reports are the unmoved full channel. The twin is the complete map twin.
200 trials per class. The empirical 99th percentile replaces the full-pool threshold 9.61.
Stored moved statistics are then rescored. They are not recomputed.
"""
import os
import time
import numpy as np
import torch
import rsrp_model as rm
import scene_data as sd
import step5_detect as s5
from step4_mle import setup

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
N, TRIALS = 10, 200


def overlap():
    z = np.load(sd.facade_file("B", "3.5"))
    e = np.load(os.path.join(ART, "e2e_munich_facadeB_3.5GHz.npz"))
    m2 = np.load(os.path.join(ART, "e2e_munich_facadeB_3.5GHz_moved2.npz"))
    m4 = np.load(os.path.join(ART, "e2e_munich_facadeB_3.5GHz_moved4.npz"))
    assert (m2["idx"] == e["idx"]).all() and (m4["idx"] == e["idx"]).all()
    keep = e["found"] & m2["found"] & m4["found"]
    pos = np.searchsorted(np.where(z["fval"])[0], e["idx"])[keep]
    t = lambda x: torch.tensor(x[keep]).to(torch.complex128)
    h0s, hm, ha = t(e["h0_same"]), t(e["h_model"]), t(e["h_all"])
    hrest = ha - (hm - h0s)
    return pos, hrest, ha


POS, HREST, HALL = overlap()


def trial(task):
    torch.set_num_threads(1)
    cls_name, seed = task
    cfg, th0, U, H0, los = setup("B", "3.5")
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    keep = los[POS] if cls_name == "LOS" else ~los[POS]
    j = rng.choice(np.where(keep)[0], N, replace=False)
    idx = POS[j]
    mu = 10 * torch.log10(cfg.ptx * torch.abs(HALL[j] @ cfg.C.conj().T) ** 2 + cfg.noise)
    r = rm.measure(cfg, mu, gen)
    u_hat = U[idx] + torch.cat([s5.SIGMA_U * torch.randn(N, 2, generator=gen), torch.zeros(N, 1)], 1)
    return cls_name, float(s5.statistic(cfg, th0, r, u_hat, HREST[j], rng))


def main():
    from multiprocessing import get_context
    cfg, th0, U, H0, los = setup("B", "3.5")
    nlos = int((~los[POS]).sum())
    n_los = int(los[POS].sum())
    print(f"overlap users LOS {n_los} NLOS {nlos}", flush=True)
    tasks, seed = [], 210000
    for cls_name in ["LOS", "NLOS"]:
        for _ in range(TRIALS):
            seed += 1
            tasks.append((cls_name, seed))
    t0 = time.time()
    ctx = get_context("fork")
    with ctx.Pool(12) as p:
        out = p.map(trial, tasks, chunksize=1)
    res = {"LOS": [], "NLOS": []}
    for name, t in out:
        res[name].append(t)
    lines = [f"trials {TRIALS} per class, overlap LOS {n_los} NLOS {nlos}, wall {time.time()-t0:.0f}s"]
    thr = {}
    for name, ts in res.items():
        ts = np.array(ts)
        thr[name] = float(np.percentile(ts, 99))
        lines.append(f"H0 {name}: n {len(ts)} median {np.median(ts):.2f} thr99 {thr[name]:.2f} max {ts.max():.2f}")
    moved = np.load(os.path.join(ART, "step11_moved_detect.npz"))
    for key in ["LOS_2_T", "NLOS_2_T", "LOS_4_T", "NLOS_4_T"]:
        name = key.split("_")[0]
        T = moved[key][:, 0]
        p = float((T > thr[name]).mean())
        lines.append(f"{key} complete-twin P_D at overlap thr {thr[name]:.2f}: {p:.3f}  (old 9.61: {(T>9.61).mean():.3f})")
    text = "\n".join(lines)
    print(text, flush=True)
    open(os.path.join(ART, "step11c_h0.txt"), "w").write(text + "\n")
    np.savez(os.path.join(ART, "step11c_h0.npz"), **{k: np.array(v) for k, v in res.items()},
             thr_LOS=thr["LOS"], thr_NLOS=thr["NLOS"])


if __name__ == "__main__":
    main()
