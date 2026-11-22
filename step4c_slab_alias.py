"""Step 4c: the permittivity branch of the Step 4b fits is the slab's Fabry-Perot alias.

Compares |Gamma_TE| of ITU marble (0.1 m slab, 3.5 GHz) with the best slab of larger eps_r over incidence
cosines 0.1-0.95, and counts the Step 4b fits (facade B, 1 dB) that end on that branch.
Output: artifacts/step4c_slab_alias.txt
"""
import os
import numpy as np

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
F, T, E0 = 3.5e9, 0.1, 8.854e-12
K = 2 * np.pi * F / 3e8
ER0, S0 = 7.074, 0.0055 * 3.5 ** 0.9262


def gamma(c, er, s):
    e = er - 1j * s / (2 * np.pi * F * E0)
    rho = np.sqrt(e - 1 + c ** 2)
    r, q = (c - rho) / (c + rho), K * T * rho
    return r * (1 - np.exp(-2j * q)) / (1 - r ** 2 * np.exp(-2j * q))


cs = np.linspace(0.1, 0.95, 18)
ref = 20 * np.log10(abs(gamma(cs, ER0, S0)))
ers = np.arange(8, 11, 0.01)
diff = [20 * np.log10(abs(gamma(cs, er, S0))) - ref for er in ers]
i = int(np.argmin([np.sqrt(np.mean(d ** 2)) for d in diff]))
lines = [f"marble eps_r {ER0}, sigma {S0:.4f} S/m; alias eps_r {ers[i]:.2f}, same sigma: "
         f"RMS {np.sqrt(np.mean(diff[i] ** 2)):.2f} dB, max {abs(diff[i]).max():.2f} dB over incidence cosines 0.1-0.95"]
z = np.load(os.path.join(ART, "step4b_mle_far.npz"))
for c in ["LOS", "NLOS"]:
    e = z[f"{c}_se1.0_err"][:, 2]
    lines.append(f"{c}, 1 dB: eps_r error in [2.1, 2.6] (alias) {np.mean((e > 2.1) & (e < 2.6)):.2f}; |error| < 0.5 {np.mean(abs(e) < 0.5):.2f}")
print("\n".join(lines))
open(os.path.join(ART, "step4c_slab_alias.txt"), "w").write("\n".join(lines) + "\n")
