"""Step 3a: user pool, facade path and known paths for one facade, from the scene geometry and Sionna RT.

Runs in the Sionna venv:  ~/.venvs/sionna/bin/python step3_trace.py <freq_GHz> <facade A|B|E>
  A: Munich, TCCN BS (8.5, 21.0, 27.0); the facade the TCCN RIS was mounted on (mesh no-name-4,
     triangles 16356/16357, 25.0 x 16.9 m, marble).
  B: Munich, same BS; mesh no-name-4 triangles 2981/2982 (36.2 x 20.6 m, marble), the facade that
     serves the most NLOS users with one specular bounce (step3_find_facades.py).
  E: Etoile, TCCN Etoile BS (5, -90, 21); mesh no-name-3 triangles 5862/5863 (24.2 x 17.7 m,
     marble), the facade with the most users of both kinds (step3_find_facades.py etoile, batch 300).
Users: outdoor points on a 2 m grid within 150 m of the facade centre, in front of it, 1.5 m high.

Which users the facade reaches, and LOS, are decided exactly from the geometry, because Sionna's
shoot-and-bounce detection of a given path depends on the receiver batch and the number of rays
(e.g. facade B at 3.5 GHz: 884, 602, 984 or 1017 users for different settings):
  * the image-source specular point lies inside the facade triangles, and the segments
    BS -> specular point and specular point -> user are unobstructed (Mitsuba ray casts);
  * LOS: the segment BS -> user is unobstructed.
The facade path itself is the closed form (rsrp_model.py). h0 is every other path the twin knows:
the 64-element ULA channel of all Sionna paths of up to two bounces that never touch the facade
(Sionna's a excludes the carrier phase; h0 includes it). For validation, Sionna's own single-bounce
facade path is stored where Sionna finds it (fval_sionna, a_f, tau_f, theta_f, phi_f, x_f).
The ULA (half wavelength, centred on the BS) is horizontal and broadside to the facade centre.
Output: artifacts/<scene>_facade<F>_<f>GHz.npz
"""
import os, sys, time
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

FREQ = float(sys.argv[1]) * 1e9 if len(sys.argv) > 1 else 3.5e9
FACADE = sys.argv[2] if len(sys.argv) > 2 else "B"
SCENE, BS, FACADE_MESH, FACADE_TRIS = {
    "A": ("munich", (8.5, 21.0, 27.0), "no-name-4", (16356, 16357)),
    "B": ("munich", (8.5, 21.0, 27.0), "no-name-4", (2981, 2982)),
    "E": ("etoile", (5.0, -90.0, 21.0), "no-name-3", (5862, 5863))}[FACADE]
BS = np.array(BS)
RADIUS, STEP, USER_H, M = 150.0, 2.0, 1.5, 64
BATCH = int(os.environ.get("RT_BATCH", "100"))
SAMPLES = int(float(os.environ.get("RT_SAMPLES", "8e6")))
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
OUT = os.path.join(ART, f"{SCENE}_facade{FACADE}_{FREQ / 1e9:g}GHz{os.environ.get('RT_SUFFIX', '')}.npz")
TOL = 0.05                                     # metres, for "the first hit is the end point"


def facade_plane(sc):
    m = sc.objects[FACADE_MESH].mi_mesh
    V = np.array(m.vertex_positions_buffer()).reshape(-1, 3)
    F = np.array(m.faces_buffer()).reshape(-1, 3)
    tri = V[F[list(FACADE_TRIS)]]                                         # (k, 3, 3)
    n = np.cross(tri[0, 1] - tri[0, 0], tri[0, 2] - tri[0, 0])
    n = n / np.linalg.norm(n)
    if n @ (BS - tri[0, 0]) < 0:                                          # normal towards the BS side
        n = -n
    t = np.array([-n[1], n[0], 0.0])
    return n, float(n @ tri[0, 0]), tri, t


def first_hit(ms, o, e):
    """Distance to the first surface along the segments o -> e (inf if none)."""
    v = e - o
    L = np.linalg.norm(v, axis=1)
    dvec = v / L[:, None]
    ray = mi.Ray3f(mi.Point3f(*[mi.Float(o[:, i]) for i in range(3)]),
                   mi.Vector3f(*[mi.Float(dvec[:, i]) for i in range(3)]))
    si = ms.ray_intersect(ray)
    t = np.array(si.t)
    return np.where(np.array(si.is_valid()), t, np.inf), L


def in_triangles(p, tri):
    """True where points p (U, 3), on the facade plane, lie inside any of the triangles."""
    inside = np.zeros(len(p), bool)
    for a, b, c in tri:
        v0, v1, v2 = c - a, b - a, p - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d20, d21 = v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        u = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        inside |= (u >= 0) & (w >= 0) & (u + w <= 1)
    return inside


def outdoor_users(ms, centre):
    g = np.mgrid[-RADIUS:RADIUS:STEP, -RADIUS:RADIUS:STEP].reshape(2, -1).T
    g = g[np.linalg.norm(g, axis=1) <= RADIUS] + centre[:2]
    o = mi.Point3f(*[mi.Float(x) for x in [g[:, 0], g[:, 1], np.full(len(g), 200.0)]])
    dvec = mi.Vector3f(mi.Float(np.zeros(len(g))), mi.Float(np.zeros(len(g))), mi.Float(-np.ones(len(g))))
    si = ms.ray_intersect(mi.Ray3f(o, dvec))
    out = np.array(si.is_valid()) & (200.0 - np.array(si.t) < 0.3)      # first hit is the ground
    return np.c_[g[out], np.full(out.sum(), USER_H)]


def geometry_flags(ms, n, d, tri, users):
    """Exact facade reachability and LOS for every user."""
    U = len(users)
    bsr = np.repeat(BS[None], U, 0)
    b_img = BS - 2 * (n @ BS - d) * n
    v = users - b_img
    x_r = b_img + ((d - n @ b_img) / (v @ n))[:, None] * v                # specular points
    on = in_triangles(x_r, tri)
    t1, L1 = first_hit(ms, bsr, x_r)
    seen = np.abs(t1 - L1) < TOL                                          # the first thing the BS ray hits is the facade point
    t2, L2 = first_hit(ms, x_r + 0.01 * n, users)
    clear = t2 > L2 - TOL
    t3, L3 = first_hit(ms, bsr, users)
    los = t3 > L3 - TOL
    return on & seen & clear, los, x_r


def main():
    sc = load_scene(getattr(rt.scene, SCENE))
    sc.frequency = FREQ
    ms = sc.mi_scene
    n, d, tri, t = facade_plane(sc)
    centre = tri.reshape(-1, 3).mean(0)
    fid = sc.objects[FACADE_MESH].object_id
    users = outdoor_users(ms, centre)
    users = users[(users - centre) @ n > 1.0]                               # in front of the facade
    fval, los, x_geo = geometry_flags(ms, n, d, tri, users)
    look = centre - BS
    look[2] = 0
    axis = np.array([-look[1], look[0], 0.0]) / np.linalg.norm(look)       # ULA axis, broadside to the facade
    m = np.arange(M) - (M - 1) / 2                                         # array centred on the BS, as in Sionna
    print(f"{SCENE} {FREQ / 1e9:g} GHz facade {FACADE}: n={n.round(4)} d={d:.3f}  users {len(users)}, "
          f"facade reaches {fval.sum()} ({(fval & los).sum()} LOS, {(fval & ~los).sum()} NLOS)", flush=True)
    keep = np.where(fval)[0]                                               # only these are needed downstream
    sc.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    solver = PathSolver()
    U = len(users)
    fval_s, a_f, tau_f = np.zeros(U, bool), np.zeros(U, complex), np.zeros(U)
    th_f, ph_f, x_f = np.zeros(U), np.zeros(U), np.zeros((U, 3))
    h0, n_f2, los_s = np.zeros((U, M), complex), np.zeros(U, int), np.zeros(U, bool)
    kw = dict(los=True, specular_reflection=True, diffuse_reflection=False, refraction=False,
              synthetic_array=True, samples_per_src=SAMPLES, max_num_paths_per_src=SAMPLES)
    t0 = time.time()
    for i in range(0, len(keep), BATCH):
        for o in list(sc.transmitters) + list(sc.receivers):
            sc.remove(o)
        sc.add(Transmitter("tx", position=mi.Point3f(*map(float, BS))))
        b = keep[i:i + BATCH]
        for j, p in enumerate(users[b]):
            sc.add(Receiver(f"rx{j}", position=mi.Point3f(*map(float, p))))
        # depth 1: Sionna's own facade path, for validation only
        P = solver(sc, max_depth=1, **kw)
        a = np.array(P.a[0])[:, 0, 0, 0, :] + 1j * np.array(P.a[1])[:, 0, 0, 0, :]       # (rx, paths)
        valid = np.array(P.valid)[:, 0, :]
        inter = np.array(P.interactions)[0, :, 0, :]
        obj = np.array(P.objects)[0, :, 0, :]
        prim = np.array(P.primitives)[0, :, 0, :]
        tau = np.array(P.tau)[:, 0, :]
        th, ph = np.array(P.theta_t)[:, 0, :], np.array(P.phi_t)[:, 0, :]
        vert = np.array(P.vertices)[0, :, 0, :, :]
        for j, u in enumerate(b):
            los_s[u] = bool((valid[j] & (inter[j] == 0)).any())
            k_f = np.where(valid[j] & (inter[j] != 0) & (obj[j] == fid) & np.isin(prim[j], FACADE_TRIS))[0]
            if len(k_f):
                k = k_f[0]
                fval_s[u], a_f[u], tau_f[u] = True, a[j, k], tau[j, k]
                th_f[u], ph_f[u], x_f[u] = th[j, k], ph[j, k], vert[j, k]
        # depth 2: every path that never touches the facade forms h0
        P = solver(sc, max_depth=2, **dict(kw, max_num_paths_per_src=3 * SAMPLES))
        a = np.array(P.a[0])[:, 0, 0, 0, :] + 1j * np.array(P.a[1])[:, 0, 0, 0, :]
        a = a * np.exp(-2j * np.pi * FREQ * np.array(P.tau)[:, 0, :])     # add the carrier phase
        valid = np.array(P.valid)[:, 0, :]
        inter_all = np.array(P.interactions)[:, :, 0, :]
        on_f = (np.array(P.objects)[:, :, 0, :] == fid) & np.isin(np.array(P.primitives)[:, :, 0, :], FACADE_TRIS) \
            & (inter_all != 0)
        th, ph = np.array(P.theta_t)[:, 0, :], np.array(P.phi_t)[:, 0, :]
        for j, u in enumerate(b):
            touch_f = on_f[:, j].any(0)
            n_f2[u] = int((valid[j] & touch_f & ((inter_all[:, j] != 0).sum(0) == 2)).sum())
            for k in np.where(valid[j] & ~touch_f)[0]:
                kk = np.array([np.sin(th[j, k]) * np.cos(ph[j, k]), np.sin(th[j, k]) * np.sin(ph[j, k]), np.cos(th[j, k])])
                h0[u] += a[j, k] * np.exp(1j * np.pi * m * (kk @ axis))
        if (i // BATCH) % 10 == 0:
            print(f"  {i + len(b)}/{len(keep)}  Sionna finds the facade path for {fval_s[keep[:i + len(b)]].sum()}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    agree = (los_s[keep] == los[keep]).mean()
    print(f"  Sionna found the facade path for {fval_s.sum()}/{len(keep)} reachable users; "
          f"LOS flags agree for {100 * agree:.1f}%", flush=True)
    np.savez_compressed(OUT, freq=FREQ, bs=BS, users=users, n=n, d=d, t=t, facade_tri=tri.reshape(-1, 3),
                        los=los, fval=fval, x_geo=x_geo, fval_sionna=fval_s, los_sionna=los_s,
                        a_f=a_f, tau_f=tau_f, theta_f=th_f, phi_f=ph_f, x_f=x_f,
                        material=sc.objects[FACADE_MESH].radio_material.name, facade=FACADE, scene=SCENE,
                        axis=axis, h0=h0, n_f2=n_f2)
    print("saved", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
