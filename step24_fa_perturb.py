"""Step 24: false-alarm rate when other paths are wrong and the wall has not moved.

Reports are the narrowband model at the map wall, with the stored h0. The detector's
h0 is the step14 perturbation: the non-direct part is scaled by a random ITU |Gamma|
ratio against marble and delayed by U[-0.5, 0.5] m. The threshold is the paper's
narrowband value 9.61, calibrated on a correct twin. 200 trials per class, N = 10,
facade B, 3.5 GHz, sigma_e = 1 dB. Also records the offset of the free fit.

Output: artifacts/step24_fa_perturb.txt / .npz
"""
import math
import os
import time
import numpy as np
import torch
import rsrp_model as rm
from step4_mle import setup
from step10_learned import CLASSES
from step14_complete import direct
import step5_detect as det

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
N = 10
TRIALS = 200
THR = 9.61
WORKERS = 6
PARTIAL = os.path.join(ART, "step24_fa_perturb_partial.npz")


def one_h0(cfg, u, h0, is_los, rng):
    hd = torch.where(torch.tensor(is_los)[:, None], direct(cfg, u), torch.zeros_like(h0))
    rest = h0 - hd
    er_m, sg_m = itu_at(cfg, "marble")
    g_m = rm.slab_te(eps_of(er_m, sg_m, cfg), torch.tensor(0.5), 0.1, cfg.lam).abs()
    scales = []
    for cname in rng.choice(CLASSES, len(u)):
        er, sg = itu_at(cfg, cname)
        g = rm.slab_te(eps_of(er, sg, cfg), torch.tensor(0.5), 0.1, cfg.lam).abs()
        scales.append(float(g / g_m))
    delay = torch.tensor(rng.uniform(-0.5, 0.5, len(u)))
    phase = torch.exp(-1j * (2 * math.pi / cfg.lam) * delay)
    return hd + torch.tensor(scales)[:, None] * phase[:, None] * rest


def itu_at(cfg, name):
    a, b, c, d = rm.ITU[name]
    f = cfg.freq / 1e9
    return a * f ** b, c * f ** d


def eps_of(er, sig, cfg):
    return er - 1j * sig / (2 * math.pi * cfg.freq * rm.EPS0)


def trial(task):
    torch.set_num_threads(1)
    cls_name, seed = task
    cfg, th0, U, H0, los = setup("B", "3.5")
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    pool = np.where(los if cls_name == "LOS" else ~los)[0]
    idx = rng.choice(pool, N, replace=False)
    u, h0 = U[idx], H0[idx]
    h0p = one_h0(cfg, u, h0, los[idx], rng)
    mu = rm.rsrp_mean_phase(cfg, th0, u, h0, torch.zeros(N))
    r = rm.measure(cfg, mu, gen)
    u_hat = u + torch.cat([det.SIGMA_U * torch.randn(N, 2, generator=gen), torch.zeros(N, 1)], 1)
    c0, c1, th = det.costs(cfg, th0, r, u_hat, h0p, ret_th=True)
    return cls_name, seed, 2 * (c0 - c1), float(th[0] - th0[0])


def all_tasks():
    tasks, seed = [], 24000
    for cls_name in ("LOS", "NLOS"):
        for _ in range(TRIALS):
            seed += 1
            tasks.append((cls_name, seed))
    return tasks


def load_done():
    if not os.path.exists(PARTIAL):
        return {}, []
    z = np.load(PARTIAL)
    rows = [{"cls": str(c), "seed": int(s), "T": float(t), "bias": float(b)}
            for c, s, t, b in zip(z["cls"], z["seed"], z["T"], z["bias"])]
    return {r["seed"]: r for r in rows}, rows


def save_rows(rows):
    np.savez_compressed(
        PARTIAL,
        cls=np.array([r["cls"] for r in rows]),
        seed=np.array([r["seed"] for r in rows]),
        T=np.array([r["T"] for r in rows]),
        bias=np.array([r["bias"] for r in rows]),
    )


def write_report(rows, seconds):
    lines = [f"GLRT with perturbed non-wall channel, wall unmoved, threshold {THR} "
             f"(correct-twin calibration), N={N}, {TRIALS} trials/class, {len(rows)} done, {seconds:.0f}s"]
    save = {}
    for cls_name in ("LOS", "NLOS"):
        sub = [r for r in rows if r["cls"] == cls_name]
        T = np.array([r["T"] for r in sub])
        b = np.array([r["bias"] for r in sub])
        fa = float(np.mean(T > THR))
        lines.append(
            f"{cls_name}: P_FA {fa:.3f} ({int(np.sum(T > THR))}/{len(T)}) at {THR}; "
            f"median T {np.median(T):.2f}; 99th {np.percentile(T, 99):.2f}; "
            f"median |offset error| {np.median(np.abs(b)):.2f} m "
            f"(10th--90th {np.percentile(np.abs(b), 10):.2f}--{np.percentile(np.abs(b), 90):.2f})"
        )
        save[f"{cls_name}_T"] = T
        save[f"{cls_name}_bias"] = b
    text = "\n".join(lines) + "\n"
    open(os.path.join(ART, "step24_fa_perturb.txt"), "w").write(text)
    np.savez_compressed(os.path.join(ART, "step24_fa_perturb.npz"), **save)
    print(text, flush=True)


def main():
    # drop the unused placeholder so a reader does not call it
    done, rows = load_done()
    pending = [t for t in all_tasks() if t[-1] not in done]
    print(f"step24 pending {len(pending)}  already {len(done)}", flush=True)
    t0 = time.time()
    if pending:
        from multiprocessing import get_context
        with get_context("fork").Pool(WORKERS) as pool:
            for i, (cls_name, seed, T, bias) in enumerate(pool.imap_unordered(trial, pending, chunksize=1), 1):
                rows.append({"cls": cls_name, "seed": seed, "T": T, "bias": bias})
                if i % 10 == 0 or i == len(pending):
                    save_rows(rows)
                    print(f"step24 {len(rows)}  {time.time() - t0:.0f}s  T={T:.2f}", flush=True)
    write_report(rows, time.time() - t0)
    print("DONE step24", flush=True)


if __name__ == "__main__":
    main()
