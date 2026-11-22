"""Step 11 (reviewer C1): detection when the wall really moved in the ray tracer.

Reports come from Sionna's full channel with facade B moved by Delta (step11_moved_trace.py). The twin
still holds the map: a complete twin knows every path of the UNMOVED scene except the single-bounce
facade path (h_rest, incl. two-bounce facade paths frozen at the map position); a single-bounce twin
knows only the paths that avoid the facade. T as in step5_detect.statistic (material known),
N = 10, 300 user sets per class and Delta, users whose facade path exists in both scenes.
Thresholds: model-calibrated, P_FA = 1% (step5b_detect_ci.npz if present, else step5_detect.npz).
Output: artifacts/step11_moved_detect.npz/.txt
"""
import math, os, sys, time
import numpy as np
import torch
import rsrp_model as rm
import scene_data as sd
import step5_detect as s5
from step4_mle import setup

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
QUICK = len(sys.argv) > 1
N, SETS, DELTAS = 10, (4 if QUICK else 300), [2.0, 4.0]


def channels(delta):
    z = np.load(sd.facade_file("B", "3.5"))
    e = np.load(os.path.join(ART, "e2e_munich_facadeB_3.5GHz.npz"))
    m = np.load(os.path.join(ART, f"e2e_munich_facadeB_3.5GHz_moved{delta:g}.npz"))
    assert (m["idx"] == e["idx"]).all()
    keep = e["found"] & m["found"]
    pos = np.searchsorted(np.where(z["fval"])[0], e["idx"])[keep]
    t = lambda x: torch.tensor(x[keep]).to(torch.complex128)
    h0s, hm, ha = t(e["h0_same"]), t(e["h_model"]), t(e["h_all"])
    return pos, h0s, ha - (hm - h0s), t(m["h_all"])


def trial(task):
    torch.set_num_threads(1)
    delta, cls_name, seed = task
    cfg, th0, U, H0, los = setup("B", "3.5")
    pos, h0s, hrest, hmoved = channels(delta)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    keep = los[pos] if cls_name == "LOS" else ~los[pos]
    j = torch.tensor(rng.choice(np.where(keep)[0], N, replace=False))
    idx = pos[j.numpy()]
    mu = 10 * torch.log10(cfg.ptx * torch.abs(hmoved[j] @ cfg.C.conj().T) ** 2 + cfg.noise)
    r = rm.measure(cfg, mu, gen)
    u_hat = U[idx] + torch.cat([s5.SIGMA_U * torch.randn(N, 2, generator=gen), torch.zeros(N, 1)], 1)
    return task[:2], (s5.statistic(cfg, th0, r, u_hat, hrest[j], rng), s5.statistic(cfg, th0, r, u_hat, h0s[j], rng))


def thresholds():
    f = os.path.join(ART, "step5b_detect_ci.npz")
    if os.path.exists(f):
        z = np.load(f)
        return {c: float(z[f"{c}_pfa1_thr"]) for c in ["LOS", "NLOS"]}, "step5b (2000 trials)"
    z = np.load(os.path.join(ART, "step5_detect.npz"))
    return {c: float(z[f"{c}_10_thr"]) for c in ["LOS", "NLOS"]}, "step5 (300 trials)"


def main():
    from multiprocessing import Pool
    tasks, seed = [], 110000
    for delta in DELTAS:
        for cls_name in ["LOS", "NLOS"]:
            for _ in range(SETS):
                seed += 1
                tasks.append((delta, cls_name, seed))
    t0 = time.time()
    with Pool(18 if not QUICK else 4) as p:
        out = p.map(trial, tasks, chunksize=1)
    thr, src = thresholds()
    res, save, lines = {}, {}, [f"thresholds P_FA=1% from {src}: " + ", ".join(f"{c} {v:.2f}" for c, v in thr.items())]
    for key, ts in out:
        res.setdefault(key, []).append(ts)
    for (delta, cls_name), ts in sorted(res.items()):
        ts = np.array(ts)
        n_sets = len(ts)
        pc, p1 = (ts[:, 0] > thr[cls_name]).mean(), (ts[:, 1] > thr[cls_name]).mean()
        save[f"{cls_name}_{delta:g}_T"] = ts
        users = int((setup("B", "3.5")[4][channels(delta)[0]] == (cls_name == "LOS")).sum())
        lines.append(f"Delta {delta:g} m {cls_name:4s}: P_D {pc:.2f} with a complete twin frozen at the map "
                     f"(median T {np.median(ts[:, 0]):.1f}), {p1:.2f} with a single-bounce twin "
                     f"(median T {np.median(ts[:, 1]):.1f})  [{n_sets} sets from {users} users]")
    lines.append(f"total {time.time() - t0:.0f}s")
    print("\n".join(lines))
    suf = "_quick" if QUICK else ""
    np.savez_compressed(os.path.join(ART, f"step11_moved_detect{suf}.npz"), **save)
    open(os.path.join(ART, f"step11_moved_detect{suf}.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
