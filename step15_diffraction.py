"""M2 probe: diffraction on NLOS users whose specular h0 is ~0.

Eight served NLOS users of facade B. samples_per_src is 5e5.
The paper's specular trace used 8e6, so this is a probe, not a replacement count.
Depth 2. Specular, diffuse and diffraction on; refraction off.
"""
import os
import time
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver, InteractionType

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
BS = np.array((8.5, 21.0, 27.0))
SAMPLES = 500_000
N_USERS = 8


def main():
    z = np.load(os.path.join(ART, "munich_facadeB_3.5GHz.npz"))
    served = z["fval"].astype(bool)
    nlos = served & ~z["los"].astype(bool)
    amp = np.abs(z["h0"]).mean(1)
    pool = np.where(nlos & (amp < 1e-8))[0]
    rng = np.random.default_rng(15)
    pick = rng.choice(pool, N_USERS, replace=False)
    users = z["users"][pick]
    t0 = time.time()
    sc = load_scene(rt.scene.munich)
    sc.frequency = 3.5e9
    sc.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.add(Transmitter("tx", position=mi.Point3f(*map(float, BS))))
    for j, p in enumerate(users):
        sc.add(Receiver(f"rx{j}", position=mi.Point3f(*map(float, p))))
    solver = PathSolver()
    P = solver(
        sc, max_depth=2, los=True, specular_reflection=True, diffuse_reflection=True,
        refraction=False, diffraction=True, edge_diffraction=True, synthetic_array=True,
        samples_per_src=SAMPLES, max_num_paths_per_src=3 * SAMPLES, seed=15,
    )
    a = np.array(P.a[0])[:, 0, 0, 0, :] + 1j * np.array(P.a[1])[:, 0, 0, 0, :]
    valid = np.array(P.valid)[:, 0, :]
    inter = np.array(P.interactions)[:, :, 0, :]
    fid = sc.objects["no-name-4"].object_id
    on_f = (np.array(P.objects)[:, :, 0, :] == fid) & np.isin(
        np.array(P.primitives)[:, :, 0, :], (2981, 2982)) & (inter != 0)
    diff = (inter == InteractionType.DIFFRACTION).any(0)
    spec = (inter == InteractionType.SPECULAR).any(0) & ~diff
    lines = [f"users {N_USERS} drawn from {len(pool)} served NLOS with |h0|~0, samples {SAMPLES}, wall {time.time()-t0:.0f}s"]
    powers = []
    for j in range(N_USERS):
        v = valid[j]
        touch = on_f[:, j].any(0)
        avoid = v & ~touch
        d = avoid & diff[j]
        s = avoid & spec[j]
        pd = float((np.abs(a[j, d]) ** 2).sum()) if d.any() else 0.0
        ps = float((np.abs(a[j, s]) ** 2).sum()) if s.any() else 0.0
        powers.append((pd, ps, int(d.sum()), int(s.sum()), int(avoid.sum())))
        lines.append(f"user {int(pick[j])}: diffracted paths {int(d.sum())} power {pd:.3e}, specular-avoid {int(s.sum())} power {ps:.3e}, avoid-facade paths {int(avoid.sum())}")
    pd = np.array([p[0] for p in powers])
    ps = np.array([p[1] for p in powers])
    lines.append(
        f"median diffracted power {np.median(pd):.3e}, users with a diffracted path {int((pd>0).sum())}/{N_USERS}, "
        f"median specular-avoid power {np.median(ps):.3e}"
    )
    text = "\n".join(lines)
    print(text, flush=True)
    open(os.path.join(ART, "step15_diffraction.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
