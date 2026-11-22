"""Step 11 (reviewer C1, Sionna part): re-trace the end-to-end users with facade B moved.

Runs in the Sionna venv:  ~/.venvs/sionna/bin/python step11_moved_trace.py <Delta_m>
The four vertices of facade B (no-name-4 triangles 2981, 2982) are translated by Delta along the
facade normal n (towards the BS, d -> d + Delta, as in step5_detect); the faces sharing them stretch,
so the building front moves as a whole. Everything else is as in step3_e2e_trace.py (same users,
64-element ULA, depth 2, batch 100, 8e6 rays). Gate: the re-traced single-bounce facade path must
match the closed form at d0 + Delta (path length), else the run aborts.
Stores h_all (every path of the moved scene, the true channel) and found (single-bounce facade path
present). Output: artifacts/e2e_munich_facadeB_3.5GHz_moved<Delta>.npz
"""
import os, sys
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

DELTA, FREQ = float(sys.argv[1]), 3.5e9
MESH, TRIS = "no-name-4", (2981, 2982)
BATCH, SAMPLES, C0 = 100, 8_000_000, 299792458.0
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
z = np.load(os.path.join(ART, "munich_facadeB_3.5GHz.npz"))
e = np.load(os.path.join(ART, "e2e_munich_facadeB_3.5GHz.npz"))
idx = e["idx"]                                      # the same users as the unmoved e2e trace
users, bs, n, d0 = z["users"][idx], z["bs"], z["n"].astype(float), float(z["d"])
centre = z["facade_tri"].mean(0)                    # array axis as in the unmoved trace (twin's view)

sc = load_scene(rt.scene.munich)
sc.frequency = FREQ
params = mi.traverse(sc.mi_scene)
F = np.array(params[f"{MESH}.faces"]).reshape(-1, 3)
V = np.array(params[f"{MESH}.vertex_positions"]).reshape(-1, 3)
vid = np.unique(F[list(TRIS)].ravel())
V[vid] += DELTA * n
params[f"{MESH}.vertex_positions"] = mi.Float(V.ravel().astype(np.float32))
params.update()
fid = sc.objects[MESH].object_id
sc.tx_array = PlanarArray(num_rows=1, num_cols=64, pattern="iso", polarization="V")
sc.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
solver = PathSolver()
U = len(idx)
ha, found, lerr = np.zeros((U, 64), complex), np.zeros(U, bool), []
bb = bs - 2 * (n @ bs - (d0 + DELTA)) * n
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
    tau = np.array(P.tau)[:, 0, :]
    a = a * np.exp(-2j * np.pi * FREQ * tau)[:, None, :]
    valid = np.array(P.valid)[:, 0, :]
    inter = np.array(P.interactions)[:, :, 0, :]
    on_f = (np.array(P.objects)[:, :, 0, :] == fid) & np.isin(np.array(P.primitives)[:, :, 0, :], TRIS) & (inter != 0)
    single = on_f[0] & ((inter != 0).sum(0) == 1) & valid
    sl = slice(i, i + len(b))
    ha[sl] = (a * valid[:, None, :]).sum(-1)
    found[sl] = single.any(1)
    for j in range(len(b)):
        if single[j].any():
            lerr.append(np.abs(tau[j][single[j]] * C0 - np.linalg.norm(b[j] - bb)).min())
lmax = float(np.max(lerr)) if lerr else np.inf
print(f"Delta {DELTA} m: facade path found for {found.sum()}/{U} users; max |L_sionna - L_closed(d0+Delta)| = {lmax:.2e} m")
assert lmax < 1e-3, "moved facade does not match the closed form at d0 + Delta"
np.savez_compressed(os.path.join(ART, f"e2e_munich_facadeB_3.5GHz_moved{DELTA:g}.npz"), idx=idx, h_all=ha,
                    found=found, delta=DELTA, max_len_err=lmax)
