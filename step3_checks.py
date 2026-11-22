"""Step 3 guard: the nuisance-robust information must be ordered, user by user,
coherent >= phase nuisance >= position (0.1 mm) >= position (1 m) >= position (3 m),
and the 0.1 mm case must match the phase-only case. Writes artifacts/step3_checks.txt."""
import math, os, sys
import numpy as np
import torch
import rsrp_model as rm
import scene_data as sd

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
lines, ok = [], True


def worst(A, B):
    return min(torch.linalg.eigvalsh(A[i] - B[i]).min().item() / max(torch.linalg.eigvalsh(A[i]).max().item(), 1e-30)
               for i in range(len(A)))


for F in "ABE":
    for f in ["3.5", "28"]:
        _, cfg, th, U, h0, _ = sd.load(F, f)
        J = [rm.fisher(cfg, th, U, None, mode="material", per_user=True, h0=h0)]
        J += [rm.fisher_robust(cfg, th, U, h0, sigma_u=s) for s in [None, 1e-4, 1.0, 3.0]]
        w = [worst(J[i], J[i + 1]) for i in range(4)]
        rel = ((J[2] - J[1]).abs().amax((1, 2)) / J[1].abs().amax((1, 2))).max().item()
        good = min(w) > -1e-9 and rel < 1e-2              # 0.1 mm only approximates the sigma_u -> 0 limit
        ok &= good
        lines.append(f"[{'PASS' if good else 'FAIL'}] facade {F} {f} GHz: ordering margins {', '.join(f'{x:.1e}' for x in w)}; "
                     f"0.1 mm vs phase-only max rel diff {rel:.1e}")
print("\n".join(lines))
open(os.path.join(ART, "step3_checks.txt"), "w").write("\n".join(lines) + "\n")
sys.exit(0 if ok else 1)
