"""Turn step15b raw diffraction powers into the paper's dB gap.

gap = 10 log10(closed-form isotropic specular wall power / summed diffracted power).
On the original eight users this is median 14.93 dB, range 4.31 to 29.17 dB.
"""
import os
import numpy as np
import torch
import rsrp_model as rm
import scene_data as sd

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def main():
    raw = np.load(os.path.join(ART, "step15b_diffraction_raw.npz"))
    z = np.load(os.path.join(ART, "munich_facadeB_3.5GHz.npz"))
    _, cfg, th0, _, _, _ = sd.load("B", "3.5")
    idx = raw["idx"]
    users = torch.tensor(z["users"][idx], dtype=torch.float64)
    d, phi = th0[0], th0[1]
    _, Lr, _, _, cosi, valid = rm.geometry(cfg, d, phi, users)
    gr = cfg.lam / (4 * math.pi * Lr)
    eps = th0[2] - 1j * (10 ** th0[3]) / (2 * math.pi * cfg.freq * rm.EPS0)
    gam = rm.reflect_te(cfg, eps, cosi)
    p_iso = ((gam.abs() * gr) ** 2).detach().numpy()
    valid = valid.detach().numpy().astype(bool)
    pd, ps = raw["pd"], raw["ps"]
    with np.errstate(divide="ignore"):
        gap = 10 * np.log10(p_iso / np.where(pd > 0, pd, np.nan))
    ok = (pd > 0) & valid & np.isfinite(gap)
    lines = [
        f"users {len(idx)} served NLOS with |h0|~0, samples {int(raw['samples'])}, trace {float(raw['seconds']):.0f}s",
        "gap = 10 log10(closed-form isotropic specular wall power / summed diffracted power that avoids the facade)",
        "The same definition on the original 8 users gives median 14.93 dB and range 4.31 to 29.17 dB.",
        f"diffracted path present {int((pd > 0).sum())}/{len(idx)}; specular path that avoids the facade {int((ps > 0).sum())}/{len(idx)}",
        f"valid specular wall {int(valid.sum())}/{len(idx)}",
        f"gap dB on {int(ok.sum())} users: median {np.median(gap[ok]):.2f}, "
        f"10th {np.percentile(gap[ok], 10):.2f}, 90th {np.percentile(gap[ok], 90):.2f}, "
        f"min {np.min(gap[ok]):.2f}, max {np.max(gap[ok]):.2f}",
    ]
    text = "\n".join(lines)
    print(text, flush=True)
    open(os.path.join(ART, "step15b_diffraction.txt"), "w").write(text + "\n")
    np.savez_compressed(
        os.path.join(ART, "step15b_diffraction.npz"),
        idx=idx, pd=pd, ps=ps, nd=raw["nd"], p_iso=p_iso, valid=valid, gap_db=gap,
    )


if __name__ == "__main__":
    import math
    main()
