"""Step 3a': rank the vertical facades of Munich by how many users they serve with a single
specular bounce, split by whether those users have LOS to the BS.

Runs in the Sionna venv:  step3_find_facades.py [scene] [bs "x,y,z"]  (default: munich, TCCN BS).
Users on a 3 m outdoor grid within 200 m.
Coplanar triangles (same mesh, normal and offset within 1 cm) are merged into one facade.
Prints the top facades and writes artifacts/<scene>_facades.npz.
"""
import os, sys
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

SCENE = sys.argv[1] if len(sys.argv) > 1 else "munich"
BS = np.array([float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "8.5,21.0,27.0").split(",")])
RADIUS, STEP, USER_H = 200.0, 3.0, 1.5
BATCH = int(os.environ.get("RT_BATCH", "3000"))
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def outdoor_users(ms):
    g = np.mgrid[-RADIUS:RADIUS:STEP, -RADIUS:RADIUS:STEP].reshape(2, -1).T
    g = g[np.linalg.norm(g, axis=1) <= RADIUS] + BS[:2]
    o = mi.Point3f(*[mi.Float(x) for x in [g[:, 0], g[:, 1], np.full(len(g), 200.0)]])
    dvec = mi.Vector3f(mi.Float(np.zeros(len(g))), mi.Float(np.zeros(len(g))), mi.Float(-np.ones(len(g))))
    si = ms.ray_intersect(mi.Ray3f(o, dvec))
    out = np.array(si.is_valid()) & (200.0 - np.array(si.t) < 0.3)
    return np.c_[g[out], np.full(out.sum(), USER_H)]


def main():
    sc = load_scene(getattr(rt.scene, SCENE))
    sc.frequency = 3.5e9
    id2name = {o.object_id: name for name, o in sc.objects.items()}
    tris = {}
    for name, o in sc.objects.items():
        m = o.mi_mesh
        V = np.array(m.vertex_positions_buffer()).reshape(-1, 3)
        F = np.array(m.faces_buffer()).reshape(-1, 3)
        tris[name] = V[F]
    users = outdoor_users(sc.mi_scene)
    print("users", len(users), flush=True)
    sc.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    solver = PathSolver()
    rec = []                                   # (user, has_los, object name, primitive)
    for i in range(0, len(users), BATCH):
        for o in list(sc.transmitters) + list(sc.receivers):
            sc.remove(o)
        sc.add(Transmitter("tx", position=mi.Point3f(*map(float, BS))))
        b = users[i:i + BATCH]
        for j, p in enumerate(b):
            sc.add(Receiver(f"rx{j}", position=mi.Point3f(*map(float, p))))
        P = solver(sc, max_depth=1, los=True, specular_reflection=True, diffuse_reflection=False,
                   refraction=False, synthetic_array=True, samples_per_src=2_000_000,
                   max_num_paths_per_src=2_000_000)
        valid = np.array(P.valid)[:, 0, :]
        inter = np.array(P.interactions)[0, :, 0, :]
        obj = np.array(P.objects)[0, :, 0, :]
        prim = np.array(P.primitives)[0, :, 0, :]
        for j in range(len(b)):
            has_los = bool((valid[j] & (inter[j] == 0)).any())
            for k in np.where(valid[j] & (inter[j] != 0))[0]:
                rec.append((i + j, has_los, id2name[int(obj[j, k])], int(prim[j, k])))
        print(f"  {i + len(b)}/{len(users)}", flush=True)
    # merge triangles into planes
    planes = {}
    for u, has_los, name, p in rec:
        if name == "ground":
            continue
        tri = tris[name][p]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        n = n / np.linalg.norm(n)
        if abs(n[2]) > 1e-3:
            continue                                  # vertical facades only
        if n @ (BS - tri[0]) < 0:
            n = -n
        key = (name, round(float(np.arctan2(n[1], n[0])), 2), round(float(n @ tri[0]), 1))
        e = planes.setdefault(key, {"los": set(), "nlos": set(), "prims": set(), "n": n, "d": float(n @ tri[0])})
        e["los" if has_los else "nlos"].add(u)
        e["prims"].add(p)
    rows = sorted(planes.items(), key=lambda kv: -len(kv[1]["nlos"]))
    print("\nTop facades by NLOS users served (single specular bounce):")
    for (name, phi, d), e in rows[:12]:
        t = np.array([-e["n"][1], e["n"][0], 0.0])
        pts = np.concatenate([tris[name][p] for p in e["prims"]])
        w = (pts @ t).max() - (pts @ t).min()
        h = pts[:, 2].max()
        print(f"  {name:10s} {sc.objects[name].radio_material.name:9s} phi {phi:+.2f} d {d:+8.1f}  width {w:5.1f} m  height {h:5.1f} m  "
              f"NLOS users {len(e['nlos']):4d}  LOS users {len(e['los']):4d}  prims {sorted(e['prims'])}")
    np.savez_compressed(os.path.join(ART, f"{SCENE}_facades.npz"), users=users,
                        keys=np.array([str(k) for k, _ in rows]),
                        n_nlos=np.array([len(e["nlos"]) for _, e in rows]),
                        n_los=np.array([len(e["los"]) for _, e in rows]),
                        prims=np.array([str(sorted(e["prims"])) for _, e in rows]),
                        top_nlos_users=users[sorted(rows[0][1]["nlos"])], top_los_users=users[sorted(rows[0][1]["los"])])


if __name__ == "__main__":
    main()
