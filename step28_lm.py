"""Grid profile with Levenberg-Marquardt on the nuisances.

The multi-start of step27 misses the true wall on the distant non-line-of-sight
fits. This script minimises the same cost on a grid of offset and angle.
Material, user position and wall-path phase are fitted at each node by
damped Gauss-Newton, from the same two material starts as the LBFGS profile.
The likelihood is the step27 likelihood: no censored clipping term, beams
taken from the noise-free mean.

  python3 step28_lm.py run
"""
import math
import os
import sys
import time
import traceback
import numpy as np
import torch

import step27_prior as s
import step28_grid as g

ART = s.ART
PARTIAL = os.path.join(ART, "step28_lm_partial.npz")
TXT = os.path.join(ART, "step28_lm.txt")
N_ITER = 12


def _cost(cfg, r, users_hat, h0, pick, d, phi, eps, sig, zd, psi, prec, d_map, phi_map, var_d):
    n = len(users_hat)
    th = torch.tensor([d, phi, float(eps), float(sig)], dtype=r.dtype)
    with torch.no_grad():
        u = users_hat + torch.cat([zd, torch.zeros(n, 1, dtype=r.dtype)], 1)
        y = s.reduced_mean(cfg, th, u, h0, psi.reshape(-1), pick)
        e = r - y
        nll = 0.5 * torch.einsum("nk,kl,nl->", e, prec, e)
        pen = (d - d_map) ** 2 / (2.0 * var_d) + (phi - phi_map) ** 2 / (2.0 * s.VAR_PHI)
        return float(nll + (zd ** 2).sum() / 2 + pen)


def _lm_start(cfg, r, users_hat, h0, pick, d, phi, d_map, phi_map, var_d, prec, white,
              eps0, sig0, n_iter=N_ITER):
    dtype = r.dtype
    n, b = pick.shape
    eps = torch.tensor(float(eps0), dtype=dtype)
    sig = torch.tensor(float(sig0), dtype=dtype)
    zd = torch.zeros(n, 2, dtype=dtype)
    th = torch.tensor([d, phi, float(eps0), float(sig0)], dtype=dtype)
    psi = g.phase_init(cfg, th, users_hat, h0, pick, r, prec, 8).to(dtype)
    hr = torch.view_as_real(h0)
    d_t = torch.tensor(float(d), dtype=dtype)
    phi_t = torch.tensor(float(phi), dtype=dtype)
    W = white

    def f(eps_, sig_, zd_, psi_, user, hr_):
        th_ = torch.stack([d_t, phi_t, eps_, sig_])
        u = (user + torch.cat([zd_, user.new_zeros(1)]))[None]
        h = torch.view_as_complex(hr_)[None]
        return s.rm.rsrp_mean_phase(cfg, th_, u, h, psi_.reshape(1))[0]

    jac = torch.func.vmap(
        torch.func.jacfwd(f, argnums=(0, 1, 2, 3)),
        in_dims=(None, None, 0, 0, 0, 0),
    )

    def reduce_j(J):
        js = J[torch.arange(n), pick[:, 0]]
        rows = [js]
        for i in range(1, b):
            rows.append(J[torch.arange(n), pick[:, i]] - js)
        return torch.stack(rows, 1)

    lam = 1e-2
    cost = _cost(cfg, r, users_hat, h0, pick, d, phi, eps, sig, zd, psi, prec, d_map, phi_map, var_d)
    small = 0
    eye_m = torch.eye(2, dtype=dtype)
    eye_u = torch.eye(3, dtype=dtype)
    for it in range(n_iter):
        Je, Js, Jz, Jp = jac(eps, sig, zd, psi.reshape(n, 1), users_hat, hr)
        with torch.no_grad():
            y = s.reduced_mean(
                cfg, torch.tensor([float(d), float(phi), float(eps), float(sig)], dtype=dtype),
                users_hat + torch.cat([zd, torch.zeros(n, 1, dtype=dtype)], 1),
                h0, psi.reshape(n), pick)
            e = torch.einsum("ab,nb->na", W, r - y)
            Je = torch.einsum("ab,nb->na", W, reduce_j(Je))
            Js = torch.einsum("ab,nb->na", W, reduce_j(Js))
            Jz = torch.einsum("ab,nbc->nac", W, reduce_j(Jz))
            Jp = torch.einsum("ab,nb->na", W, reduce_j(Jp.squeeze(-1)))
            Jm = torch.stack([Je, Js], -1)
            Ju = torch.cat([Jz, Jp.unsqueeze(-1)], -1)
            Gmm = torch.einsum("nbi,nbj->ij", Jm, Jm)
            gm = torch.einsum("nbi,nb->i", Jm, e)
            Guu = torch.einsum("nbi,nbj->nij", Ju, Ju)
            Guu[:, 0, 0] += 1.0
            Guu[:, 1, 1] += 1.0
            Gum = torch.einsum("nbi,nbj->nij", Ju, Jm)
            gu = torch.einsum("nbi,nb->ni", Ju, e)
            gu[:, 0] -= zd[:, 0]
            gu[:, 1] -= zd[:, 1]
            Guu_l = Guu + lam * eye_u
            Gmm_l = Gmm + lam * eye_m
            try:
                Guu_inv = torch.linalg.inv(Guu_l)
                mid = torch.einsum("nab,nbc->nac", Guu_inv, Gum)
                S = Gmm_l - torch.einsum("nba,nbc->ac", Gum, mid)
                rhs = gm - torch.einsum("nba,nb->a", Gum, torch.einsum("nab,nb->na", Guu_inv, gu))
                dm = torch.linalg.solve(S, rhs)
                du = torch.einsum("nab,nb->na", Guu_inv, gu) - torch.einsum("nac,c->na", mid, dm)
            except RuntimeError:
                lam = min(lam * 10, 1e6)
                if lam >= 1e6:
                    break
                continue
            accepted = False
            before = cost
            for alpha in (1.0, 0.5, 0.25, 0.1):
                eps_n = (eps + alpha * dm[0]).clamp(1.0, 30.0)
                sig_n = (sig + alpha * dm[1]).clamp(-6.0, 2.0)
                zd_n = zd + alpha * du[:, :2]
                psi_n = psi + alpha * du[:, 2]
                c_n = _cost(
                    cfg, r, users_hat, h0, pick, d, phi, eps_n, sig_n, zd_n, psi_n,
                    prec, d_map, phi_map, var_d)
                if np.isfinite(c_n) and c_n < cost - 1e-4:
                    eps, sig, zd, psi = eps_n, sig_n, zd_n, psi_n
                    cost = c_n
                    lam = max(lam * 0.3, 1e-8)
                    accepted = True
                    break
            if not accepted:
                lam = min(lam * 5, 1e6)
                small = 0
                if lam >= 1e6:
                    break
                continue
            if it >= 5 and before - cost < 0.5:
                small += 1
                if small >= 2:
                    break
            else:
                small = 0
    return cost


def lm_profile(cfg, r, users_hat, h0, pick, d, phi, d_map, phi_map, var_d, prec, white, n_iter=N_ITER):
    best = float("inf")
    for eps0, sig0 in g.MATS:
        c = _lm_start(
            cfg, r, users_hat, h0, pick, d, phi, d_map, phi_map, var_d, prec, white,
            eps0, sig0, n_iter=n_iter)
        if c < best:
            best = c
    return best


def _axes_coarse():
    d = np.arange(-14.0, 14.0 + 1e-9, 2.0)
    phi = np.deg2rad(np.arange(-6.0, 6.0 + 1e-9, 2.0))
    return d, phi


def _eval(cfg, r, u_hat, h0, pick, d_map, phi_map, var_d, prec, white, d_axis, phi_axis, tag):
    nodes = []
    t0 = time.time()
    n_phi = len(phi_axis)
    for dd in d_axis:
        for pp in phi_axis:
            c = lm_profile(
                cfg, r, u_hat, h0, pick, d_map + float(dd), phi_map + float(pp),
                d_map, phi_map, var_d, prec, white)
            nodes.append((float(dd), float(pp), c))
        if tag and len(nodes) % (2 * n_phi) == 0:
            print(f"progress {tag} nodes={len(nodes)} {time.time() - t0:.0f}s", flush=True)
    return nodes


def lm_trial(task):
    torch.set_num_threads(1)
    kind, n_beams, n_users, half, name, sid, idx, seed, err_d, err_phi, bcrb_ref = task
    try:
        return _lm_trial(task)
    except Exception:
        traceback.print_exc()
        return {
            "seed": int(seed), "sid": int(sid), "name": str(name), "users": int(n_users),
            "true": np.nan, "map": np.nan, "mean": np.nan, "map_phi": np.nan, "true_phi": np.nan,
            "map_c": np.nan, "truth_c": np.nan, "fit_c": np.nan, "ms": np.nan,
            "bcrb": np.nan, "bcrb_ok": 0.0, "seconds": np.nan, "n_nodes": 0, "edge": 0.0, "cap": 0,
        }


def _lm_trial(task):
    kind, n_beams, n_users, half, name, sid, idx, seed, err_d, err_phi, bcrb_ref = task
    cfg, th0, U, H0, _ = s.prepare(kind)
    var_d = half ** 2 / 3.0
    users, h0 = U[idx], H0[idx]
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(int(seed))
    th_true = th0.clone()
    th_true[0] = th0[0] + float(rng.normal(0.0, math.sqrt(var_d)))
    th_true[1] = th0[1] + float(rng.normal(0.0, math.sqrt(s.VAR_PHI)))
    r, pick, _, _ = s.make_report(cfg, th_true, users, h0, n_beams, gen)
    u_hat = users + torch.cat(
        [s.SIGMA_U * torch.randn(len(idx), 2, generator=gen), torch.zeros(len(idx), 1)], 1)
    prec, white, _ = s.report_precision(cfg.sigma_e, n_beams, r.dtype)
    J, _ = s.total_information(cfg, th_true, users, h0, n_beams)
    bcrb = s.dd_variance((J + s.prior_info(var_d, J.dtype)).detach().numpy())
    bcrb_ok = 0.0
    if np.isfinite(bcrb_ref) and np.isfinite(bcrb):
        bcrb_ok = 1.0 if abs(bcrb - float(bcrb_ref)) <= 1e-4 * max(1.0, abs(float(bcrb_ref))) else 0.0
    d_map, phi_map = float(th0[0]), float(th0[1])
    true_off = float(th_true[0] - d_map)
    true_phi = float(th_true[1] - phi_map)
    t0 = time.time()
    tag = f"N={n_users} {name} seed={seed}"
    d_axis, phi_axis = _axes_coarse()
    coarse = _eval(cfg, r, u_hat, h0, pick, d_map, phi_map, var_d, prec, white, d_axis, phi_axis, tag)
    bi = int(np.nanargmin([n[2] for n in coarse]))
    fine = _eval(
        cfg, r, u_hat, h0, pick, d_map, phi_map, var_d, prec, white,
        *g.refine_axes(coarse[bi][0], coarse[bi][1]), tag + " fine")
    map_off, map_phi, map_c, mean_off = g.posterior(coarse, fine)
    truth_c = lm_profile(
        cfg, r, u_hat, h0, pick, float(th_true[0]), float(th_true[1]),
        d_map, phi_map, var_d, prec, white)
    if np.isfinite(err_d):
        ms = true_off + float(err_d)
    else:
        ms = float("nan")
    edge = 1.0 if abs(abs(map_off) - 14.0) < 0.11 or abs(abs(map_phi) - math.radians(6.0)) < 1e-3 else 0.0
    return {
        "seed": int(seed), "sid": int(sid), "name": str(name), "users": int(n_users),
        "true": true_off, "map": float(map_off), "mean": float(mean_off),
        "map_phi": float(map_phi), "true_phi": true_phi,
        "map_c": float(map_c), "truth_c": float(truth_c), "fit_c": float("nan"),
        "ms": float(ms), "bcrb": float(bcrb), "bcrb_ok": bcrb_ok,
        "seconds": time.time() - t0, "n_nodes": len(coarse) + len(fine), "edge": edge, "cap": 0,
    }


def _order(tasks):
    def key(t):
        rank = {200: 0, 100: 1, 50: 2, 10: 3, 20: 4}.get(int(t[2]), 9)
        return (rank, 0 if t[4] == "NLOS" else 1, int(t[7]))
    return sorted(tasks, key=key)


def _load():
    if not os.path.exists(PARTIAL):
        return []
    z = np.load(PARTIAL, allow_pickle=True)
    rows = []
    for i in range(len(z["seed"])):
        rows.append({k: (str(z[k][i]) if k == "name" else z[k][i]) for k in z.files})
        rows[-1]["seed"] = int(rows[-1]["seed"])
        rows[-1]["sid"] = int(rows[-1]["sid"])
        rows[-1]["users"] = int(rows[-1]["users"])
        rows[-1]["n_nodes"] = int(rows[-1]["n_nodes"])
    return rows


def _save(rows):
    if not rows:
        return
    np.savez_compressed(
        PARTIAL,
        seed=np.array([int(r["seed"]) for r in rows]),
        sid=np.array([int(r["sid"]) for r in rows]),
        name=np.array([r["name"] for r in rows]),
        users=np.array([int(r["users"]) for r in rows]),
        true=np.array([r["true"] for r in rows], dtype=float),
        map=np.array([r["map"] for r in rows], dtype=float),
        mean=np.array([r["mean"] for r in rows], dtype=float),
        map_phi=np.array([r["map_phi"] for r in rows], dtype=float),
        true_phi=np.array([r["true_phi"] for r in rows], dtype=float),
        map_c=np.array([r["map_c"] for r in rows], dtype=float),
        truth_c=np.array([r["truth_c"] for r in rows], dtype=float),
        fit_c=np.array([r["fit_c"] for r in rows], dtype=float),
        ms=np.array([r["ms"] for r in rows], dtype=float),
        bcrb=np.array([r["bcrb"] for r in rows], dtype=float),
        bcrb_ok=np.array([r["bcrb_ok"] for r in rows], dtype=float),
        seconds=np.array([r["seconds"] for r in rows], dtype=float),
        n_nodes=np.array([int(r["n_nodes"]) for r in rows]),
        edge=np.array([r["edge"] for r in rows], dtype=float),
        cap=np.array([int(r["cap"]) for r in rows]),
    )


def run():
    from multiprocessing import get_context
    torch.set_num_threads(1)
    tasks = _order(g.campaign_tasks())
    rows = _load()
    have = {int(r["seed"]) for r in rows}
    todo = [t for t in tasks if int(t[7]) not in have]
    print(f"lm resume {len(rows)} done, {len(todo)} left", flush=True)
    t0 = time.time()
    if todo:
        workers = int(os.environ.get("WORKERS", "4"))
        with get_context("fork").Pool(workers) as pool:
            for k, row in enumerate(pool.imap_unordered(lm_trial, todo, chunksize=1), 1):
                rows.append(row)
                if k % 2 == 0 or k == len(todo):
                    _save(rows)
                    print(
                        f"lm {k}/{len(todo)} new, {len(rows)} stored, {time.time() - t0:.0f}s "
                        f"last N={row['users']} {row['name']} sec={row['seconds']} "
                        f"bcrb_ok={row['bcrb_ok']} map={row['map']} true={row['true']} "
                        f"map_c={row['map_c']} truth_c={row['truth_c']}",
                        flush=True,
                    )
    # summarise expects the grid row schema
    lines = [
        "LM grid. Coarse 2 m by 2 deg on [-14, 14] m by [-6, 6] deg. "
        "Fine window 0.5 m by 0.5 deg. Nuisance fit is damped Gauss-Newton, "
        f"at most {N_ITER} iterations, two material starts. Same step27 likelihood."
    ]
    lines.extend(g.summarise(rows)[1:])
    lines.append(f"lm seconds {time.time() - t0:.0f}")
    text = "\n".join(lines) + "\n"
    open(TXT, "w").write(text)
    print(text, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run()
    else:
        raise SystemExit("usage: step28_lm.py run")
