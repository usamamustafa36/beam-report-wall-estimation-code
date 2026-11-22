"""M2 residual: RMS report gap after the best offset on the same perturbation as step14.

The stored step14 line is the gap at the true wall. This script also records the gap
at the offset, on a +/-3 m grid, that minimises the mean-square report error.
Facade B, 3.5 GHz, the first 80 of the 300 sets from rng seed 14.
"""
import math
import os
import numpy as np
import torch
import rsrp_model as rm
import scene_data as sd
from step10_learned import CLASSES
from step14_complete import direct

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
N, SETS = 10, 300


def eps_of(er, sig, f=3.5e9):
    return er - 1j * sig / (2 * math.pi * f * rm.EPS0)


def main():
    _, cfg, th0, U, H0, los = sd.load("B", "3.5")
    rng = np.random.default_rng(14)
    pools = {"LOS": np.where(los)[0], "NLOS": np.where(~los)[0]}
    sets = {k: [rng.choice(v, N, replace=False) for _ in range(SETS)] for k, v in pools.items()}
    hd_all = direct(cfg, U)
    itu = {}
    for name in CLASSES:
        a, b, c, d = rm.ITU[name]
        itu[name] = (a * 3.5 ** b, c * 3.5 ** d)
    lines = []
    for name in ["LOS", "NLOS"]:
        bias, gap0, gapb, edge, h0norm = [], [], [], [], []
        for s in sets[name][:80]:
            u, h0 = U[s], H0[s]
            hd = torch.where(torch.tensor(los[s])[:, None], hd_all[s], torch.zeros_like(hd_all[s]))
            rest = h0 - hd
            h0norm.append(float(rest.abs().mean()))
            cls = rng.choice(CLASSES, len(s))
            er_m, sg_m = itu["marble"]
            g_m = rm.slab_te(eps_of(er_m, sg_m), torch.tensor(0.5), 0.1, cfg.lam).abs()
            scales = []
            for cname in cls:
                er, sg = itu[cname]
                g = rm.slab_te(eps_of(er, sg), torch.tensor(0.5), 0.1, cfg.lam).abs()
                scales.append(float(g / g_m))
            delay = torch.tensor(rng.uniform(-0.5, 0.5, len(s)))
            phase = torch.exp(-1j * (2 * math.pi / cfg.lam) * delay)
            h0p = hd + torch.tensor(scales, dtype=h0.dtype)[:, None] * phase[:, None] * rest
            mu = rm.rsrp_mean_phase(cfg, th0, u, h0, torch.zeros(len(s)))
            best, bd = 1e18, 0.0
            grid = np.linspace(-3, 3, 31)
            for dlt in grid:
                th = th0.clone()
                th[0] = th0[0] + float(dlt)
                mup = rm.rsrp_mean_phase(cfg, th, u, h0p, torch.zeros(len(s)))
                c = float(((mu - mup) ** 2).mean())
                if c < best:
                    best, bd = c, float(dlt)
            mup0 = rm.rsrp_mean_phase(cfg, th0, u, h0p, torch.zeros(len(s)))
            gap0.append(float(((mu - mup0) ** 2).mean().sqrt()))
            gapb.append(math.sqrt(best))
            bias.append(bd)
            edge.append(abs(bd) >= 2.99)
        b = np.abs(bias)
        lines.append(
            f"{name}: median |bias| {np.median(b):.2f} m, "
            f"gap at truth {np.median(gap0):.2f} dB, "
            f"gap after best offset {np.median(gapb):.2f} dB, "
            f"10th-90th residual [{np.percentile(gapb, 10):.2f}, {np.percentile(gapb, 90):.2f}], "
            f"grid edge {np.mean(edge):.2f}, "
            f"median |rest| {np.median(h0norm):.3e}"
        )
    text = "\n".join(lines)
    print(text, flush=True)
    open(os.path.join(ART, "step14b_residual.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
