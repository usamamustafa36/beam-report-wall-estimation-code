"""Step 13: what a twin loses to a stale map, end to end (test, then fuse or not).

Facade B, 3.5 GHz, N = 10, material known, sigma_e = 1 dB, sigma_u = 1 m. Per trial the true wall is
the scene's wall moved by Delta along its normal; the map value carries its own error,
m = (d0, phi0) + w, w ~ N(0, diag(0.5 m, 1 deg)^2) (the model of Eq. (10)). Estimators of (d, phi):
  radio  : the free fit of the likelihood-ratio test (step5_detect.costs, H1 estimate);
  fused  : (J + Jm)^-1 (J radio + Jm m), J the robust information on (d, phi) at the radio
           estimate (material known), Jm = Sigma_map^-1  (linearised MAP fusion, Eq. (10));
  policy : fused if the test accepts the map, else radio; tau = empirical 99th percentile of the
           statistic at Delta = 0 in this experiment (P_FA = 1%). Two tests:
    glrt : T = 2 (C0 - C1), wall held exactly at the map value (step5_detect.costs). Under no change it
           also detects the map's own error w, so its threshold is large (v1 result:
           artifacts/step13_policy_v1_glrt_at_map.txt).
    mahal: T_m = (radio - m)^T (J^-1 + Sigma_map)^-1 (radio - m), the consistency of the radio estimate
           with a map of the stated accuracy (linearised GLRT of "true wall = m - w").
Output: artifacts/step13_policy.npz/.txt (offset RMSE per Delta and estimator, rejection rate).
"""
import math, os, sys, time
import numpy as np
import torch
import rsrp_model as rm
import step5_detect as s5
from step4_mle import setup

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
QUICK = len(sys.argv) > 1
N, SIG_MAP = 10, (0.5, math.radians(1.0))
DELTAS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
T0_TRIALS, T1_TRIALS = (6, 3) if QUICK else (300, 150)


def trial(task):
    torch.set_num_threads(1)
    cls_name, delta, seed = task
    cfg, th0, U, H0, los = setup("B", "3.5")
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    idx = rng.choice(np.where(los if cls_name == "LOS" else ~los)[0], N, replace=False)
    u, h0 = U[idx], H0[idx]
    th_true = th0.clone()
    th_true[0] += delta
    th_map = th0.clone()
    th_map[0] += SIG_MAP[0] * rng.standard_normal()
    th_map[1] += SIG_MAP[1] * rng.standard_normal()
    r = rm.measure(cfg, rm.rsrp_mean_phase(cfg, th_true, u, h0, torch.zeros(N)), gen)
    u_hat = u + torch.cat([s5.SIGMA_U * torch.randn(N, 2, generator=gen), torch.zeros(N, 1)], 1)
    c0, c1, th_r = s5.costs(cfg, th_map, r, u_hat, h0, ret_th=True)
    J = rm.fisher_robust(cfg, th_r, u_hat, h0, sigma_u=s5.SIGMA_U).sum(0).numpy()[:2, :2]
    Jm = np.diag([1 / SIG_MAP[0] ** 2, 1 / SIG_MAP[1] ** 2])
    radio, m = th_r[:2].numpy(), th_map[:2].numpy()
    fused = np.linalg.solve(J + Jm, J @ radio + Jm @ m)
    dm = radio - m
    t_m = float(dm @ np.linalg.solve(np.linalg.inv(J) + np.linalg.inv(Jm), dm))
    truth = th_true[:2].numpy()
    return (cls_name, delta), (2 * (c0 - c1), t_m), radio - truth, fused - truth


def main():
    from multiprocessing import Pool
    tasks, seed = [], 130000
    for cls_name in ["LOS", "NLOS"]:
        for delta in DELTAS:
            for _ in range(T0_TRIALS if delta == 0 else T1_TRIALS):
                seed += 1
                tasks.append((cls_name, delta, seed))
    t0 = time.time()
    with Pool(18 if not QUICK else 4) as p:
        out = p.map(trial, tasks, chunksize=1)
    res = {}
    for key, t, er, ef in out:
        d = res.setdefault(key, {"T": [], "radio": [], "fused": []})  # T: (glrt, mahal) per trial
        d["T"].append(t)
        d["radio"].append(er)
        d["fused"].append(ef)
    lines, save = [], {"deltas": np.array(DELTAS)}
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    for cls_name in ["LOS", "NLOS"]:
        for ti, test in enumerate(["glrt", "mahal"]):
            tau = np.percentile(np.array(res[(cls_name, 0.0)]["T"])[:, ti], 99)
            lines.append(f"{cls_name} {test}: tau (P_FA 1%) = {tau:.2f}")
            rows = []
            for delta in DELTAS:
                d = res[(cls_name, delta)]
                T, er, ef = np.array(d["T"])[:, ti], np.array(d["radio"])[:, 0], np.array(d["fused"])[:, 0]
                rej = T > tau
                ep = np.where(rej, er, ef)
                rows.append([delta, rmse(er), rmse(ef), rmse(ep), rej.mean()])
                lines.append(f"  Delta {delta:3.1f} m: d RMSE radio {rmse(er):.2f} | always fuse {rmse(ef):.2f} | "
                             f"test then fuse {rmse(ep):.2f} m | map rejected in {100 * rej.mean():.0f}%  ({len(T)} trials)")
            save[f"{cls_name}_{test}_rows"] = np.array(rows)
            save[f"{cls_name}_{test}_tau"] = tau
        for delta in DELTAS:  # per-trial values for later analysis
            d = res[(cls_name, delta)]
            save[f"{cls_name}_{delta:g}_T"] = np.array(d["T"])
            save[f"{cls_name}_{delta:g}_radio"] = np.array(d["radio"])
            save[f"{cls_name}_{delta:g}_fused"] = np.array(d["fused"])
    lines.append(f"total {time.time() - t0:.0f}s")
    print("\n".join(lines))
    suf = "_quick" if QUICK else ""
    np.savez_compressed(os.path.join(ART, f"step13_policy{suf}.npz"), **save)
    open(os.path.join(ART, f"step13_policy{suf}.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
