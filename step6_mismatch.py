"""Step 6: what a wrong model costs.

(a) Stale map prior. The map says the wall offset is m = d0 + Delta (it moved by Delta) with stated
    accuracy sigma_map (0.5 m, 1 deg). Linearised around the truth, with robust information J of
    N users (step 3, sigma_u = 1 m):
      radio only : MSE_d = [J^-1]_dd
      map only   : MSE_d = sigma_map^2 + Delta^2
      fused (MAP): MSE_d = [(J + Jp)^-1]_dd + ([(J + Jp)^-1 Jp e_d]_d Delta)^2
    (map value = truth + Delta + w, w ~ N(0, Jp^-1): error covariance A (J + Jp) A = A, bias A Jp e_d Delta;
    checked by Monte Carlo. [A J A]_dd would apply only to a map without random error.)
    and the displacement Delta* beyond which the fused estimate is worse than radio only.
(b) Physics mismatch. Reports from the full Sionna channel (all paths incl. two-bounce facade paths,
    full polarisation; e2e users) are fitted, noise-free, with the single-bounce TE model
    (material known, nuisance phases free, positions exact; the known paths h0 come from the same
    Sionna run): the minimiser is the pseudo-true wall.
    Its offset from the true wall is the bias the model mismatch causes; control: the same fit on
    the model's own paths (h_model) must return the truth.
Usage: python3 step6_mismatch.py [a|b]   -> artifacts/step6_mismatch_<part>.npz/.txt
"""
import math, os, sys, time
import numpy as np
import torch
import rsrp_model as rm
import scene_data as sd

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
PART = sys.argv[1] if len(sys.argv) > 1 else "a"
SIG_MAP = (0.5, math.radians(1.0))
DELTAS = np.array([0, 0.25, 0.5, 1, 2, 4, 8, 16])


def part_a():
    z = np.load(os.path.join(ART, "step3_crb_B_3.5GHz.npz"))
    Jn, los = z["Jn_robust1"], z["los"]
    Jp = np.diag([1 / SIG_MAP[0] ** 2, 1 / SIG_MAP[1] ** 2, 0, 0])
    rng = np.random.default_rng(3)
    lines, save = [], {"deltas": DELTAS}
    for name, pool in [("LOS", np.where(los)[0]), ("NLOS", np.where(~los)[0])]:
        for N in [5, 10, 20]:
            radio, fused, cross = [], [], []
            for _ in range(300):
                J = Jn[rng.choice(pool, N, replace=False)].sum(0)
                if np.linalg.eigvalsh(J)[0] <= 0:
                    continue
                r = np.linalg.inv(J)[0, 0]
                A = np.linalg.inv(J + Jp)
                g = (A @ Jp)[0, 0]                    # bias per metre of map error in d
                f = A[0, 0] + (g * DELTAS) ** 2
                radio.append(r)
                fused.append(f)
                cross.append(math.sqrt(max(r - A[0, 0], 0)) / g if g > 0 else np.inf)
            radio, fused, cross = np.array(radio), np.array(fused), np.array(cross)
            mapo = SIG_MAP[0] ** 2 + DELTAS ** 2
            save[f"{name}_{N}_radio"], save[f"{name}_{N}_fused"], save[f"{name}_{N}_cross"] = radio, fused, cross
            lines.append(f"{name:4s} N={N:2d}: RMSE d radio {np.sqrt(np.median(radio)):.2f} m | fused at Delta=0 "
                         f"{np.sqrt(np.median(fused[:, 0])):.2f} m | map-only at 0 {SIG_MAP[0]:.2f} m | "
                         f"fused worse than radio beyond Delta* = {np.median(cross):.2f} m "
                         f"[{np.percentile(cross, 10):.2f}, {np.percentile(cross, 90):.2f}] | "
                         f"fused RMSE at Delta=4 m {np.sqrt(np.median(fused[:, 5])):.2f} m")
    return lines, save


def pseudo_true(task):
    torch.set_num_threads(1)
    from step4_mle import fit, setup
    which, cls_name, N, seed = task
    cfg, th0, U, H0, los = setup("B", "3.5")
    pos, h0s, hmod, hall, hrest = sd.e2e("B", "3.5")
    keep = los[pos] if cls_name == "LOS" else ~los[pos]
    rng = np.random.default_rng(seed)
    j = rng.choice(np.where(keep)[0], N, replace=False)
    idx = pos[j]
    jj = torch.tensor(j)
    h = (hmod if which == "model" else hall)[jj]
    known = (hrest if which == "full" else h0s)[jj]       # "full": complete twin; "full1b": single-bounce twin
    r = 10 * torch.log10(cfg.ptx * torch.abs(h @ cfg.C.conj().T) ** 2 + cfg.noise)
    th, _ = fit(cfg, r, U[idx], known, 1e-3, th0.clone(), False, free=(1.0, 1.0, 0.0, 0.0))
    return task[:3], (th - th0).numpy()


def part_b():
    from multiprocessing import Pool
    tasks, seed = [], 700
    for which in ["model", "full", "full1b"]:
        for cls_name in ["LOS", "NLOS"]:
            for N in [10, 20]:
                for _ in range(40):
                    seed += 1
                    tasks.append((which, cls_name, N, seed))
    with Pool(18) as p:
        out = p.map(pseudo_true, tasks, chunksize=1)
    res = {}
    for k, b in out:
        res.setdefault(k, []).append(b)
    crb = np.load(os.path.join(ART, "step3_crb_B_3.5GHz.npz"))
    lines, save = [], {}
    for (which, cls_name, N), b in sorted(res.items()):
        b = np.array(b)
        ref = np.median(crb[f"crb_robust1_{cls_name}_{N}"][:, 0])
        save[f"{which}_{cls_name}_{N}"] = b
        lines.append(f"{which:6s} {cls_name:4s} N={N:2d}: pseudo-true bias |d| median {np.median(np.abs(b[:, 0])):.3f} m "
                     f"(90% {np.percentile(np.abs(b[:, 0]), 90):.3f}), |phi| median {math.degrees(np.median(np.abs(b[:, 1]))):.3f} deg; "
                     f"robust sqrt CRB d {ref:.2f} m")
    return lines, save


if __name__ == "__main__":
    t0 = time.time()
    lines, save = part_a() if PART == "a" else part_b()
    lines.append(f"total {time.time() - t0:.0f}s")
    print("\n".join(lines))
    np.savez_compressed(os.path.join(ART, f"step6_mismatch_{PART}.npz"), **save)
    open(os.path.join(ART, f"step6_mismatch_{PART}.txt"), "w").write("\n".join(lines) + "\n")
