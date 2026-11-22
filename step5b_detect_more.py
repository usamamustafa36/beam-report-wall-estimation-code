"""Step 5b: more trials, confidence intervals, and an unknown material (reviewer M5, M6).

Facade B, 3.5 GHz, N = 10, model reports (step5_detect.trial, kind "model"); seeds disjoint from step 5.
  part "ci" : material known (marble). 2000 no-change and 200 moved-wall trials per point; thresholds
              at P_FA = 1% and 5% with bootstrap 95% CIs; P_D with Wilson 95% CIs.
  part "mat": material unknown. The true material is drawn uniformly from the 7 ITU classes;
              C0 and C1 are each minimised over the classes (step5_detect.costs per class), so
              T = 2 (min_c C0(c) - min_c C1(c)). 500 no-change and 100 moved-wall trials per point.
  facade E (third argument "E", part "ci" only): 1000 no-change and 100 moved-wall trials per point,
              plus 300 no-change trials on Sionna's full channel for a complete and a single-bounce
              twin (step5_detect.trial kinds "sionna", "sionna1b").
Usage: python3 step5b_detect_more.py ci|mat [B|E] [quick] -> artifacts/step5b_detect_<part>[_E].npz/.txt
"""
import math, os, sys, time
import numpy as np
import torch
import rsrp_model as rm
import step5_detect as s5
from step4_mle import setup
from step9_material import CLASSES, material_theta

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
PART = sys.argv[1]
FACADE = "E" if "E" in sys.argv[2:] else "B"
QUICK = "quick" in sys.argv[2:]
s5.FACADE = FACADE
N = 10
DELTAS = {"ci": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0], "mat": [0.0, 2.0, 4.0, 8.0]}[PART]
H0_T, H1_T = {"ci": (2000, 200), "mat": (500, 100)}[PART]
SIONNA_T = 0
if FACADE == "E":
    H0_T, H1_T, SIONNA_T = 1000, 100, 300
if QUICK:
    H0_T, H1_T, SIONNA_T = 6, 3, 3 if FACADE == "E" else 0


def trial_mat(task):
    torch.set_num_threads(1)
    cls_name, delta, seed = task
    cfg, th0, U, H0, los = setup(FACADE, "3.5")
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    pool = np.where(los if cls_name == "LOS" else ~los)[0]
    idx = rng.choice(pool, N, replace=False)
    truth = int(rng.integers(len(CLASSES)))
    th_true = material_theta(th0, CLASSES[truth], cfg.freq)
    th_true[0] += delta
    h0 = H0[idx]
    r = rm.measure(cfg, rm.rsrp_mean_phase(cfg, th_true, U[idx], h0, torch.zeros(N)), gen)
    u_hat = U[idx] + torch.cat([s5.SIGMA_U * torch.randn(N, 2, generator=gen), torch.zeros(N, 1)], 1)
    c = np.array([s5.costs(cfg, material_theta(th0, name, cfg.freq), r, u_hat, h0) for name in CLASSES])
    c0, c1 = c[:, 0].min(), min(c[:, 1].min(), c[:, 0].min())
    return (cls_name, delta), 2 * (c0 - c1), truth


def trial_ci(task):
    cls_name, delta, seed, kind = task
    key, t = s5.trial((kind, cls_name, N, delta, seed))
    return (cls_name, delta) if kind == "model" else (cls_name, kind), t, -1


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    from multiprocessing import Pool
    tasks, seed = [], {"ci": 70000, "mat": 90000}[PART]
    for cls_name in ["LOS", "NLOS"]:
        for delta in DELTAS:
            for _ in range(H0_T if delta == 0 else H1_T):
                seed += 1
                tasks.append((cls_name, delta, seed) if PART == "mat" else (cls_name, delta, seed, "model"))
        for kind in ("sionna", "sionna1b"):
            for _ in range(SIONNA_T):
                seed += 1
                tasks.append((cls_name, 0.0, seed, kind))
    t0 = time.time()
    with Pool(18 if not QUICK else 4) as p:
        out = p.map(trial_mat if PART == "mat" else trial_ci, tasks, chunksize=1)
    T = {}
    for key, t, _ in out:
        T.setdefault(key, []).append(t)
    rng = np.random.default_rng(5)
    lines, save = [f"part {PART}, facade {FACADE}, N={N}, chi2_2 quantiles: 95% 5.99, 99% 9.21"], {"deltas": np.array(DELTAS[1:])}
    for cls_name in ["LOS", "NLOS"]:
        t0_ = np.array(T[(cls_name, 0.0)])
        save[f"{cls_name}_T0"] = t0_
        for pfa in [1, 5]:
            thr = np.percentile(t0_, 100 - pfa)
            boot = [np.percentile(rng.choice(t0_, len(t0_)), 100 - pfa) for _ in range(2000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            parts = []
            pd, pdl, pdh = [], [], []
            for dl in DELTAS[1:]:
                t1 = np.array(T[(cls_name, dl)])
                k = int((t1 > thr).sum())
                a, b = wilson(k, len(t1))
                pd.append(k / len(t1)); pdl.append(a); pdh.append(b)
                parts.append(f"{dl:g}m:{k / len(t1):.2f}[{a:.2f},{b:.2f}]")
            save[f"{cls_name}_pfa{pfa}_thr"] = thr
            save[f"{cls_name}_pfa{pfa}_pd"], save[f"{cls_name}_pfa{pfa}_pdlo"], save[f"{cls_name}_pfa{pfa}_pdhi"] = \
                np.array(pd), np.array(pdl), np.array(pdh)
            lines.append(f"{cls_name:4s} P_FA={pfa}%: threshold {thr:.2f} [{lo:.2f}, {hi:.2f}] "
                         f"({len(t0_)} H0 trials, median T {np.median(t0_):.2f}); P_D " + " ".join(parts)
                         + f" ({len(T[(cls_name, DELTAS[1])])} trials each)")
            for kind, lab in (("sionna", "complete twin"), ("sionna1b", "single-bounce twin")):
                if (cls_name, kind) in T:
                    ts = np.array(T[(cls_name, kind)])
                    save[f"{cls_name}_{kind}_T"] = ts
                    lines.append(f"     full Sionna, no change, {lab}: P_FA {np.mean(ts > thr):.3f} at this threshold "
                                 f"(median T {np.median(ts):.1f}, {len(ts)} sets)")
    lines.append(f"total {time.time() - t0:.0f}s")
    print("\n".join(lines))
    suf = ("" if FACADE == "B" else f"_{FACADE}") + ("_quick" if QUICK else "")
    np.savez_compressed(os.path.join(ART, f"step5b_detect_{PART}{suf}.npz"), **save)
    open(os.path.join(ART, f"step5b_detect_{PART}{suf}.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
