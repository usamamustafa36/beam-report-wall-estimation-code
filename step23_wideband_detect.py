"""Step 23: stale-twin GLRT on the two-path model, power averaged over 20 MHz.

Same statistic as step5_detect.costs. N = 10, facade B, 3.5 GHz, sigma_e = 1 dB,
sigma_u = 1 m. Direct path on for LOS only. Deltas 0, 2, 3, 4 m. The threshold is the
99th percentile of the no-change trials in this run. Counts: 200 no-change and 60
change trials per class. The nuisance grid is evaluated in phase blocks of 4; it is
checked against step5_detect.nuisance_grid on one trial before the pool starts.

Output: artifacts/step23_wideband_detect.txt / .npz
"""
import math
import os
import time
import numpy as np
import torch
import rsrp_model as rm
import wideband as wb
from step4_mle import setup
import step5_detect as det

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
BAND = 20.0
N = 10
DELTAS = [0.0, 2.0, 3.0, 4.0]
H0_TRIALS, H1_TRIALS = 200, 60
WORKERS = 8
PARTIAL = os.path.join(ART, "step23_wideband_detect_partial.npz")


def nuisance_grid_fast(cfg, th, r, u_hat, h0, step=0.2, half=2.5):
    """Same search as step5_detect.nuisance_grid, four phases per call."""
    g = torch.arange(-half, half + 1e-9, step)
    gx, gy = [a.reshape(-1) for a in torch.meshgrid(g, g, indexing="ij")]
    m = len(gx)
    ps = torch.arange(16) * (2 * math.pi / 16)
    prior = (gx ** 2 + gy ** 2) / (2 * det.SIGMA_U ** 2)
    dxy, psi = torch.zeros(len(u_hat), 2), torch.zeros(len(u_hat))
    with torch.no_grad():
        for n in range(len(u_hat)):
            best = None
            for p0 in range(0, 16, 4):
                pp = ps[p0:p0 + 4]
                P = len(pp)
                u = u_hat[n].repeat(m * P, 1)
                u[:, 0] += gx.repeat(P)
                u[:, 1] += gy.repeat(P)
                phase = pp.repeat_interleave(m)
                mu = rm.rsrp_mean_phase(cfg, th, u, h0[n].repeat(m * P, 1), phase)
                c = ((r[n] - mu) ** 2).sum(1) / (2 * cfg.sigma_e ** 2)
                c = c.reshape(P, m).T + prior[:, None]
                local = int(c.reshape(-1).argmin())
                val = float(c.reshape(-1)[local])
                ig, ip = divmod(local, P)
                if best is None or val < best[0]:
                    best = (val, ig, p0 + ip)
            dxy[n] = torch.stack([gx[best[1]], gy[best[1]]])
            psi[n] = ps[best[2]]
    return dxy, psi


def trial(task):
    torch.set_num_threads(1)
    wb.install(BAND)
    det.nuisance_grid = nuisance_grid_fast
    cls_name, delta, seed = task
    cfg, th0, U, H0, los = setup("B", "3.5")
    cfg.use_direct = cls_name == "LOS"
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    pool = np.where(los if cls_name == "LOS" else ~los)[0]
    idx = rng.choice(pool, N, replace=False)
    th_true = th0.clone()
    th_true[0] += delta
    h0 = torch.zeros(N, cfg.n_ant, dtype=torch.complex128)
    u = U[idx]
    mu = rm.rsrp_mean_phase(cfg, th_true, u, h0, torch.zeros(N))
    r = rm.measure(cfg, mu, gen)
    u_hat = u + torch.cat([det.SIGMA_U * torch.randn(N, 2, generator=gen), torch.zeros(N, 1)], 1)
    c0, c1 = det.costs(cfg, th0, r, u_hat, h0)
    return cls_name, delta, seed, 2 * (c0 - c1)


def all_tasks():
    tasks, seed = [], 23000
    for cls_name in ("LOS", "NLOS"):
        for delta in DELTAS:
            n = H0_TRIALS if delta == 0 else H1_TRIALS
            for _ in range(n):
                seed += 1
                tasks.append((cls_name, delta, seed))
    return tasks


def load_done():
    if not os.path.exists(PARTIAL):
        return {}, []
    z = np.load(PARTIAL)
    rows = [{"cls": str(c), "delta": float(d), "seed": int(s), "T": float(t)}
            for c, d, s, t in zip(z["cls"], z["delta"], z["seed"], z["T"])]
    return {r["seed"]: r for r in rows}, rows


def save_rows(rows):
    np.savez_compressed(
        PARTIAL,
        cls=np.array([r["cls"] for r in rows]),
        delta=np.array([r["delta"] for r in rows]),
        seed=np.array([r["seed"] for r in rows]),
        T=np.array([r["T"] for r in rows]),
    )


def write_report(rows, seconds):
    lines = [f"two-path wideband GLRT, B={BAND:g} MHz, N={N}, direct path on for LOS only, "
             f"H0={H0_TRIALS}, H1={H1_TRIALS}, {len(rows)} trials, {seconds:.0f}s"]
    save = {}
    for cls_name in ("LOS", "NLOS"):
        t0 = np.array([r["T"] for r in rows if r["cls"] == cls_name and r["delta"] == 0.0])
        thr = float(np.percentile(t0, 99))
        bits = [f"{cls_name}: threshold {thr:.2f} (99th of {len(t0)} no-change; median T {np.median(t0):.2f})"]
        save[f"{cls_name}_T0"] = t0
        save[f"{cls_name}_thr"] = thr
        for dl in DELTAS[1:]:
            t = np.array([r["T"] for r in rows if r["cls"] == cls_name and r["delta"] == dl])
            k = int(np.sum(t > thr))
            # Clopper-Pearson-style normal interval is not used; report the count and a percentile bootstrap.
            rng = np.random.default_rng(1)
            boots = [np.mean(rng.choice(t, len(t), replace=True) > thr) for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            bits.append(f"  {dl:g} m: {k}/{len(t)} = {k / len(t):.3f}  [{lo:.3f}, {hi:.3f}]")
            save[f"{cls_name}_T{dl:g}"] = t
        lines.extend(bits)
    text = "\n".join(lines) + "\n"
    open(os.path.join(ART, "step23_wideband_detect.txt"), "w").write(text)
    np.savez_compressed(os.path.join(ART, "step23_wideband_detect.npz"), **save)
    print(text, flush=True)


def check_grid():
    torch.set_num_threads(4)
    wb.install(BAND)
    cfg, th0, U, H0, los = setup("B", "3.5")
    cfg.use_direct = True
    idx = np.where(los)[0][:10]
    h0 = torch.zeros(10, cfg.n_ant, dtype=torch.complex128)
    gen = torch.Generator().manual_seed(0)
    mu = rm.rsrp_mean_phase(cfg, th0, U[idx], h0, torch.zeros(10))
    r = mu + torch.randn(mu.shape, generator=gen)
    a = det.nuisance_grid(cfg, th0, r, U[idx], h0)
    b = nuisance_grid_fast(cfg, th0, r, U[idx], h0)
    dd = (a[0] - b[0]).abs().max().item()
    dp = (a[1] - b[1]).abs().max().item()
    print(f"grid check max |dxy| {dd:.3e}  max |phase| {dp:.3e}", flush=True)
    if dd > 1e-8 or dp > 1e-8:
        raise SystemExit("fast nuisance grid disagrees with step5_detect.nuisance_grid")


def main():
    check_grid()
    done, rows = load_done()
    pending = [t for t in all_tasks() if t[-1] not in done]
    print(f"step23 pending {len(pending)}  already {len(done)}", flush=True)
    t0 = time.time()
    if pending:
        from multiprocessing import get_context
        with get_context("fork").Pool(WORKERS) as pool:
            for i, (cls_name, delta, seed, T) in enumerate(pool.imap_unordered(trial, pending, chunksize=1), 1):
                rows.append({"cls": cls_name, "delta": delta, "seed": seed, "T": T})
                if i % 20 == 0 or i == len(pending):
                    save_rows(rows)
                    print(f"step23 {len(rows)}  {time.time() - t0:.0f}s  last T={T:.2f}", flush=True)
    write_report(rows, time.time() - t0)
    print("DONE step23", flush=True)


if __name__ == "__main__":
    main()
