"""Step 3c (Sionna part): 64-element ULA channels from Sionna itself, for the end-to-end check and
for the model-mismatch experiments (steps 5 and 6).

Runs in the Sionna venv:  ~/.venvs/sionna/bin/python step3_e2e_trace.py <freq_GHz> <facade>
Uses every 3rd user the facade reaches (artifacts/<scene>_facade<F>_<f>GHz.npz). BS: 1x64
PlanarArray, iso, V-pol, boresight horizontal towards the facade centre (same axis as
step3_trace.py). Depth 2, LOS + specular, batch 100, 8e6 rays. From ONE run per user it stores
  h0_same : every path that never touches the facade (what the twin knows)
  h_model : h0_same + Sionna's single-bounce facade path (what the model describes)
  h_all   : every path, incl. two-bounce paths via the facade (the truth)
all with the carrier phase, and found = whether Sionna found the single-bounce facade path
(users without it are not used). Output: artifacts/e2e_<scene>_facade<F>_<f>GHz.npz
"""
import os, sys
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

FREQ, FACADE = float(sys.argv[1]) * 1e9, sys.argv[2]
SCENE = {"A": "munich", "B": "munich", "E": "etoile"}[FACADE]
MESH, TRIS = {"A": ("no-name-4", (16356, 16357)), "B": ("no-name-4", (2981, 2982)),
              "E": ("no-name-3", (5862, 5863))}[FACADE]
BATCH, SAMPLES = 100, 8_000_000
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
z = np.load(os.path.join(ART, f"{SCENE}_facade{FACADE}_{FREQ / 1e9:g}GHz.npz"))
idx = np.where(z["fval"])[0][::3]
users, bs = z["users"][idx], z["bs"]
centre = z["facade_tri"].mean(0)

sc = load_scene(getattr(rt.scene, SCENE))
sc.frequency = FREQ
fid = sc.objects[MESH].object_id
sc.tx_array = PlanarArray(num_rows=1, num_cols=64, pattern="iso", polarization="V")
sc.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
solver = PathSolver()
U = len(idx)
h0, hm, ha, found = np.zeros((U, 64), complex), np.zeros((U, 64), complex), np.zeros((U, 64), complex), np.zeros(U, bool)
for i in range(0, U, BATCH):
    for o in list(sc.transmitters) + list(sc.receivers):
        sc.remove(o)
    sc.add(Transmitter("tx", position=mi.Point3f(*map(float, bs)),
                       look_at=mi.Point3f(float(centre[0]), float(centre[1]), float(bs[2]))))
    b = users[i:i + BATCH]
    for j, p in enumerate(b):
        sc.add(Receiver(f"rx{j}", position=mi.Point3f(*map(float, p))))
    P = solver(sc, max_depth=2, los=True, specular_reflection=True, diffuse_reflection=False, refraction=False,
               synthetic_array=True, samples_per_src=SAMPLES, max_num_paths_per_src=3 * SAMPLES)
    a = np.array(P.a[0])[:, 0, 0, :, :] + 1j * np.array(P.a[1])[:, 0, 0, :, :]          # (rx, 64, paths)
    a = a * np.exp(-2j * np.pi * FREQ * np.array(P.tau)[:, 0, :])[:, None, :]
    valid = np.array(P.valid)[:, 0, :]
    inter = np.array(P.interactions)[:, :, 0, :]
    on_f = (np.array(P.objects)[:, :, 0, :] == fid) & np.isin(np.array(P.primitives)[:, :, 0, :], TRIS) & (inter != 0)
    depth = (inter != 0).sum(0)
    touch = on_f.any(0)
    single = on_f[0] & (depth == 1)
    sl = slice(i, i + len(b))
    h0[sl] = (a * (valid & ~touch)[:, None, :]).sum(-1)
    hm[sl] = (a * (valid & (~touch | single))[:, None, :]).sum(-1)
    ha[sl] = (a * valid[:, None, :]).sum(-1)
    found[sl] = (valid & single).any(1)
np.savez_compressed(os.path.join(ART, f"e2e_{SCENE}_facade{FACADE}_{FREQ / 1e9:g}GHz.npz"), idx=idx,
                    h0_same=h0, h_model=hm, h_all=ha, found=found)
print("saved", U, "users; facade path found for", found.sum())
