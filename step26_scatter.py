"""Step 26: same 215 users with every material's scattering coefficient set to 0.5.

The scene default is 0, so step25 finds no diffuse path. This trace states the
coefficient explicitly. Solver flags, samples, and seeds match step15b.
Run with ~/.venvs/sionna/bin/python.
"""
import os
import time
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, PathSolver
from step25_diffuse import trace_batch, ART, SAMPLES, BATCH

SCAT = 0.5


def main():
    z = np.load(os.path.join(ART, "munich_facadeB_3.5GHz.npz"))
    served = z["fval"].astype(bool)
    nlos = served & ~z["los"].astype(bool)
    amp = np.abs(z["h0"]).mean(1)
    pool = np.where(nlos & (amp < 1e-8))[0]
    users = z["users"][pool]
    sc = load_scene(rt.scene.munich)
    sc.frequency = 3.5e9
    sc.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    nset = 0
    for obj in sc.objects.values():
        obj.radio_material.scattering_coefficient = mi.Float(SCAT)
        nset += 1
    got = float(np.array(sc.objects["no-name-4"].radio_material.scattering_coefficient).reshape(-1)[0])
    print(f"set scattering_coefficient={SCAT:g} on {nset} objects; facade reads {got:g}", flush=True)
    if abs(got - SCAT) > 1e-6:
        raise SystemExit("scattering coefficient did not stick")
    solver = PathSolver()
    pd = np.zeros(len(pool))
    ps = np.zeros(len(pool))
    psc = np.zeros(len(pool))
    nd = np.zeros(len(pool), int)
    ns = np.zeros(len(pool), int)
    t0 = time.time()
    for i in range(0, len(pool), BATCH):
        sl = slice(i, min(i + BATCH, len(pool)))
        pd[sl], ps[sl], psc[sl], nd[sl], ns[sl] = trace_batch(sc, solver, users[sl], seed=15 + i)
        print(f"batch {i}:{sl.stop} / {len(pool)}  {time.time() - t0:.0f}s", flush=True)
    path = os.path.join(ART, "step26_scatter_raw.npz")
    np.savez_compressed(path, idx=pool, pd=pd, ps=ps, psc=psc, nd=nd, ns=ns,
                        samples=SAMPLES, scattering=SCAT, seconds=time.time() - t0)
    print(f"wrote {path}  diffuse>0 {int((psc > 0).sum())}/{len(pool)}  {time.time() - t0:.0f}s", flush=True)
    print("DONE step26", flush=True)


if __name__ == "__main__":
    main()
