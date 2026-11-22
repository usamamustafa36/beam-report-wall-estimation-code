"""Step 21: two-path wideband robust CRB for NLOS users.

The stored non-wall channel has no per-path delay, so the average is the geometric
direct path (off for NLOS) plus the specular wall, on the same 0.24 MHz grid as
step12_wideband.py. LOS medians are already published; this script checks the LOS
Fisher against artifacts/step12_wideband.npz and fills the NLOS rows.

Facade B, 3.5 GHz, sigma_u = 1 m, sigma_e = 1 dB, 300 sets, seed 21.
Output: artifacts/step21_wideband_crb.txt / .npz
"""
import os
import time
import numpy as np
import torch
import rsrp_model as rm
import scene_data as sd
import wideband as wb

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
BANDS = [0.0, 7.2, 20.0, 100.0]
SETS, NS = 300, [5, 10, 20]


def make_fn(band, direct):
    flag = torch.tensor(bool(direct))

    def f(cfg, th, users, h0, psi, mode="material", material=None):
        return wb.rsrp(cfg, th, users, psi, band, flag.expand(users.shape[0]), mode, material)
    return f


def per_user(cfg, th0, users, band, direct):
    h0 = torch.zeros(len(users), cfg.n_ant, dtype=torch.complex128)
    fn = make_fn(band, direct)
    parts = []
    for i in range(0, len(users), 16):
        parts.append(rm.fisher_robust(cfg, th0, users[i:i + 16], h0[i:i + 16], sigma_u=1.0, rsrp_fn=fn))
        print(f"  {('LOS' if direct else 'NLOS')} B={band:g}  {min(i + 16, len(users))}/{len(users)}", flush=True)
    return torch.cat(parts).numpy()


def summarise(Jn, sets, th0):
    lines = []
    out = {}
    for N, ss in sets.items():
        crb = []
        for s in ss:
            ev = np.linalg.eigvalsh(Jn[s].sum(0))
            if ev[0] <= 1e-8:
                crb.append([np.nan, np.nan])
            else:
                d = np.diag(np.linalg.inv(Jn[s].sum(0)))
                crb.append([np.sqrt(d[0]), np.sqrt(d[2]) / float(th0[2])])
        crb = np.array(crb)
        d, e = crb[:, 0], crb[:, 1]
        ok = np.isfinite(d)
        lines.append(
            f"N={N:2d}: sqrt CRB d median {np.median(d[ok]):.2f} m "
            f"[{np.percentile(d[ok], 10):.2f}, {np.percentile(d[ok], 90):.2f}]; "
            f"eps_r {100 * np.median(e[ok]):.2f}%  singular {int((~ok).sum())}/{len(ok)}"
        )
        out[N] = crb
    return lines, out


def main():
    torch.set_num_threads(8)
    _, cfg, th0, U, H0, los = sd.load("B", "3.5")
    los_u, nlos_u = U[los], U[~los]
    ref = np.load(os.path.join(ART, "step12_wideband.npz"))
    Jchk = per_user(cfg, th0, los_u[:16], 20.0, True)
    err = np.max(np.abs(Jchk - ref["Jn_20"][:16]))
    print(f"LOS check vs step12 Jn_20, 16 users, max |J| {err:.3e}", flush=True)
    if err > 1e-6:
        raise SystemExit("wideband Fisher does not match step12; not writing NLOS bounds")
    rng = np.random.default_rng(21)
    sets = {N: [rng.choice(len(nlos_u), N, replace=False) for _ in range(SETS)] for N in NS}
    lines, save = [f"LOS Fisher check max |J-J_step12| = {err:.3e} on 16 users at 20 MHz"], {}
    for B in BANDS:
        t0 = time.time()
        Jn = per_user(cfg, th0, nlos_u, B, False)
        bits, tab = summarise(Jn, sets, th0)
        lines.append(f"B={B:5.1f} MHz ({len(wb.frequencies(cfg, B)):3d} tones) NLOS  ({time.time() - t0:.0f}s)")
        lines.extend("  " + b for b in bits)
        save[f"Jn_{B:g}"] = Jn
        for N, crb in tab.items():
            save[f"crb_N{N}_B{B:g}"] = crb
        print(lines[-1 - len(bits)], flush=True)
        for b in bits:
            print(" ", b, flush=True)
        open(os.path.join(ART, "step21_wideband_crb.txt"), "w").write("\n".join(lines) + "\n")
    np.savez_compressed(os.path.join(ART, "step21_wideband_crb.npz"), **save)
    print("DONE step21", flush=True)


if __name__ == "__main__":
    main()
