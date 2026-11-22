"""Step 12: does wideband RSRP keep the information? (limitation raised in the paper's conclusion)

Real RSRP averages the per-subcarrier power over the reference-signal bandwidth B, which averages the
two-path interference term over frequency. Facade B, 3.5 GHz, LOS users. Because the Sionna h0 sums
paths without their delays, the known path here is the geometric direct path (free space), at every
subcarrier; the narrowband case (B = 0) uses the same channel at the carrier only, so all cases are
comparable. Subcarrier grid 0.24 MHz (resolves path-length differences up to ~1.2 km).
B in {0, 7.2 MHz (SSB, 240 subcarriers at 30 kHz), 20 MHz, 100 MHz (CSI-RS)}.
Robust bound (sigma_u = 1 m, wall-path phase nuisance), 300 random sets of N in {5, 10, 20} users.
Output: artifacts/step12_wideband.npz/.txt
"""
import copy, math, os, sys, time
import numpy as np
import torch
import rsrp_model as rm
import scene_data as sd

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
BANDS_MHZ, SPACING_MHZ, SETS, NS = [0.0, 7.2, 20.0, 100.0], 0.24, 300, [5, 10, 20]
QUICK = len(sys.argv) > 1


def wideband_rsrp(cfg, band_mhz):
    """rsrp_fn for rm.fisher_robust: mean power over subcarriers of (direct path + wall path)."""
    k = max(1, int(round(band_mhz / SPACING_MHZ)) + 1) if band_mhz > 0 else 1
    freqs = cfg.freq + (np.linspace(-0.5, 0.5, k) * band_mhz * 1e6 if k > 1 else np.zeros(1))
    cfgs = []
    for f in freqs:
        c = copy.copy(cfg)
        c.freq, c.lam = float(f), rm.C0 / float(f)
        cfgs.append(c)

    def f(_cfg, th, users, h0, psi, mode="material", material=None):
        los = torch.ones(users.shape[0], dtype=torch.bool)
        p = sum(torch.abs(rm.channel(c, th, users, los, mode, material, None, psi) @ cfg.C.conj().T) ** 2
                for c in cfgs) / len(cfgs)
        return 10 * torch.log10(cfg.ptx * p + cfg.noise)
    return f, len(freqs)


def main():
    _, cfg, th0, U, H0, los = sd.load("B", "3.5")
    users = U[los][:40] if QUICK else U[los]
    h0 = torch.zeros(len(users), cfg.n_ant, dtype=torch.complex128)        # ignored by wideband_rsrp
    rng = np.random.default_rng(12)
    sets = {N: [rng.choice(len(users), N, replace=False) for _ in range(SETS)] for N in NS}
    lines, save = [], {}
    for B in BANDS_MHZ:
        fn, nf = wideband_rsrp(cfg, B)
        t0 = time.time()
        Jn = torch.cat([rm.fisher_robust(cfg, th0, users[i:i + 16], h0[i:i + 16], sigma_u=1.0, rsrp_fn=fn)
                        for i in range(0, len(users), 16)]).numpy()
        save[f"Jn_{B:g}"] = Jn
        for N in NS:
            crb = np.array([np.diag(np.linalg.pinv(Jn[s].sum(0))) for s in sets[N]])
            d, e = np.sqrt(crb[:, 0]), np.sqrt(crb[:, 2]) / float(th0[2])
            lines.append(f"B={B:5.1f} MHz ({nf:3d} subcarriers) LOS N={N:2d}: sqrt CRB d median {np.median(d):.2f} m "
                         f"[{np.percentile(d, 10):.2f}, {np.percentile(d, 90):.2f}]; eps_r {100 * np.median(e):.2f}%  "
                         f"({time.time() - t0:.0f}s)")
            print(lines[-1], flush=True)
    suf = "_quick" if QUICK else ""
    np.savez_compressed(os.path.join(ART, f"step12_wideband{suf}.npz"), **save)
    open(os.path.join(ART, f"step12_wideband{suf}.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
