"""Step 3b: compare the closed-form facade path with Sionna RT on every traced user.

Path length, specular point and departure direction must agree to numerical precision. The
coefficient is compared with Gamma_TE lam/(4 pi L) for an ITU slab (scene material) of 0.1 m (Sionna's model).
Sionna's path coefficient a excludes the carrier phase exp(-j k L), which it carries in tau, so the
phase is compared without it. Sionna uses V-pol antennas and full polarimetric reflection, our
model the TE coefficient only; the remaining gap is the TM part and is reported. Writes artifacts/step3_validate.txt.
"""
import math, os
import numpy as np
import torch
import rsrp_model as rm
import scene_data as sd

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
lines = []
for F in "ABE":
    for f in ["3.5", "28"]:
        z = np.load(sd.facade_file(F, f))
        cfg = rm.Config(freq=float(z["freq"]), bs=tuple(z["bs"]), axis=tuple(z["axis"]))
        v = z["fval"] & z["fval_sionna"]
        U = torch.tensor(z["users"][v])
        n = z["n"]
        phi, d = math.atan2(n[1], n[0]), float(z["d"])
        L_d, L_r, k_d, k_r, cos_i, valid = rm.geometry(cfg, torch.tensor(d), torch.tensor(phi), U)
        L_s = z["tau_f"][v] * rm.C0
        th, ph = z["theta_f"][v], z["phi_f"][v]
        k_s = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
        gam = rm.reflect_te(cfg, torch.tensor(rm.itu_eps(str(z["material"]), cfg.freq)), cos_i).numpy()
        model = gam * cfg.lam / (4 * math.pi * L_r.numpy())      # Sionna a excludes the carrier phase exp(-j k L) (it is in tau)
        a = z["a_f"][v]
        amp_db = 20 * np.log10(np.abs(a) / np.abs(model))
        ph_err = np.angle(a / model)
        ds = np.abs(k_s @ z["axis"] - (k_r.numpy() @ z["axis"]))
        lines += [f"facade {F} {f} GHz: {v.sum()} users ({(v & z['los']).sum()} LOS, {(v & ~z['los']).sum()} NLOS); "
                  f"paths touching the facade at bounce 2 (left out): {z['n_f2'][v].sum()}",
                  f"  path length       max |L_sionna - L_model| = {np.abs(L_s - L_r.numpy()).max():.2e} m",
                  f"  departure s=k'e   max |s_sionna - s_model| = {ds.max():.2e}",
                  f"  amplitude         Sionna/model [dB]: median {np.median(amp_db):+.2f}, 5-95% [{np.percentile(amp_db, 5):+.2f}, {np.percentile(amp_db, 95):+.2f}]",
                  f"  phase             Sionna - model [rad]: median {np.median(ph_err):+.3f}, max |.| {np.abs(ph_err).max():.3f}",
                  f"  closed-form validity flag true for {int(valid.sum())}/{v.sum()}"]
# end to end, 16 wide-beam RSRP, same-run Sionna channels (every 3rd user):
#   Sionna's model paths (h0 + Sionna facade path) vs closed form (same h0 + closed-form facade path)
#   Sionna's full channel (adds two-bounce facade paths) vs the same closed form
for F in "BE":
    _, cfg, th0, U, _, los = sd.load(F, "3.5")
    pos, h0s, hmod, hall, hrest = sd.e2e(F, "3.5")
    h_cf = rm.channel(cfg, th0, U[pos], None, mode="material", h0=h0s)
    h_cf_rest = rm.channel(cfg, th0, U[pos], None, mode="material", h0=hrest)
    rs = lambda h: 10 * torch.log10(cfg.ptx * torch.abs(h @ cfg.C.conj().T) ** 2 + cfg.noise)
    for name, h, ref in [("Sionna model paths", hmod, h_cf), ("all, 1-bounce twin", hall, h_cf), ("all, complete twin", hall, h_cf_rest)]:
        e = (rs(h) - rs(ref)).abs().numpy()
        ch = (torch.linalg.norm(h - ref, dim=1) / torch.linalg.norm(h, dim=1)).numpy()
        for cls, m in [("LOS", los[pos]), ("NLOS", ~los[pos])]:
            lines.append(f"facade {F} 3.5 GHz e2e, {name:18s} {cls:4s} ({m.sum()} users): RSRP |err| median "
                         f"{np.median(e[m]):.2f} dB, 95% {np.percentile(e[m], 95):.2f} dB; channel rel err median {np.median(ch[m]):.3f}")
print("\n".join(lines))
open(os.path.join(ART, "step3_validate.txt"), "w").write("\n".join(lines) + "\n")
