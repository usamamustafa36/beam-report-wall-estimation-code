"""Step 25: diffuse power on the 215 NLOS users whose stored h0 is ~0.

Same solver flags, sample count, batch size, and seeds as step15b_diffraction.py.
That trace enabled diffuse scattering and did not store it. This one stores the
avoid-facade diffuse power next to the diffracted and specular-avoid powers.
Run with the Sionna environment:

    ~/.venvs/sionna/bin/python step25_diffuse.py
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
BATCH = 12


def trace_batch(sc, solver, users, seed):
    for o in list(sc.transmitters) + list(sc.receivers):
        sc.remove(o)
    sc.add(Transmitter("tx", position=mi.Point3f(*map(float, BS))))
    for j, p in enumerate(users):
        sc.add(Receiver(f"rx{j}", position=mi.Point3f(*map(float, p))))
    P = solver(
        sc, max_depth=2, los=True, specular_reflection=True, diffuse_reflection=True,
        refraction=False, diffraction=True, edge_diffraction=True, synthetic_array=True,
        samples_per_src=SAMPLES, max_num_paths_per_src=3 * SAMPLES, seed=seed,
    )
    a = np.array(P.a[0])[:, 0, 0, 0, :] + 1j * np.array(P.a[1])[:, 0, 0, 0, :]
    valid = np.array(P.valid)[:, 0, :]
    inter = np.array(P.interactions)[:, :, 0, :]
    fid = sc.objects["no-name-4"].object_id
    on_f = (np.array(P.objects)[:, :, 0, :] == fid) & np.isin(
        np.array(P.primitives)[:, :, 0, :], (2981, 2982)) & (inter != 0)
    diff = (inter == InteractionType.DIFFRACTION).any(0)
    spec = (inter == InteractionType.SPECULAR).any(0) & ~diff
    scat = (inter == InteractionType.DIFFUSE).any(0)
    pd = np.zeros(len(users))
    ps = np.zeros(len(users))
    psc = np.zeros(len(users))
    nd = np.zeros(len(users), int)
    ns = np.zeros(len(users), int)
    for j in range(len(users)):
        v = valid[j]
        touch = on_f[:, j].any(0)
        avoid = v & ~touch
        dsel = avoid & diff[j]
        ssel = avoid & spec[j] & ~scat[j]
        csel = avoid & scat[j]
        pd[j] = float((np.abs(a[j, dsel]) ** 2).sum()) if dsel.any() else 0.0
        ps[j] = float((np.abs(a[j, ssel]) ** 2).sum()) if ssel.any() else 0.0
        psc[j] = float((np.abs(a[j, csel]) ** 2).sum()) if csel.any() else 0.0
        nd[j] = int(dsel.sum())
        ns[j] = int(csel.sum())
    return pd, ps, psc, nd, ns


def main():
    z = np.load(os.path.join(ART, "munich_facadeB_3.5GHz.npz"))
    served = z["fval"].astype(bool)
    nlos = served & ~z["los"].astype(bool)
    amp = np.abs(z["h0"]).mean(1)
    pool = np.where(nlos & (amp < 1e-8))[0]
    users = z["users"][pool]
    prev = np.load(os.path.join(ART, "step15b_diffraction_raw.npz"))
    if not np.array_equal(prev["idx"], pool):
        raise SystemExit("user pool differs from step15b; refusing to compare powers")
    t0 = time.time()
    sc = load_scene(rt.scene.munich)
    sc.frequency = 3.5e9
    sc.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    solver = PathSolver()
    pd = np.zeros(len(pool))
    ps = np.zeros(len(pool))
    psc = np.zeros(len(pool))
    nd = np.zeros(len(pool), int)
    ns = np.zeros(len(pool), int)
    for i in range(0, len(pool), BATCH):
        sl = slice(i, min(i + BATCH, len(pool)))
        pd[sl], ps[sl], psc[sl], nd[sl], ns[sl] = trace_batch(sc, solver, users[sl], seed=15 + i)
        print(f"batch {i}:{sl.stop} / {len(pool)}  {time.time() - t0:.0f}s", flush=True)
    path = os.path.join(ART, "step25_diffuse_raw.npz")
    np.savez_compressed(
        path, idx=pool, pd=pd, ps=ps, psc=psc, nd=nd, ns=ns,
        samples=SAMPLES, seconds=time.time() - t0,
    )
    same = np.max(np.abs(pd - prev["pd"]))
    print(f"wrote {path}  diffuse>0 {int((psc > 0).sum())}/{len(pool)}  "
          f"max |pd - step15b| {same:.3e}  {time.time() - t0:.0f}s", flush=True)
    print("DONE step25", flush=True)


if __name__ == "__main__":
    main()
