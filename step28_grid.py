"""Grid profile of the four-beam posterior over offset and angle.

The multi-start fit in step27 can stop above the cost of the true wall.
This script profiles the same negative log posterior on a grid of the two
geometric coordinates. Material, wall-path phase and user position are
optimised at each node. The grid minimum is the profiled MAP. The
profile-posterior mean uses those profiled costs on a uniform grid, with a
finer window around the coarse minimum. An oracle profile at the true offset
and angle is recorded and is not used as an estimate.

The likelihood is the step27 likelihood: independent beam noise, quantised
reports, no censored term at -30 dB, beams chosen from the noise-free mean.
That match is intentional. A censored likelihood would not say whether the
published search missed the true wall.

Modes:
  python3 step28_grid.py time
  python3 step28_grid.py run
"""
import math
import os
import sys
import time
import traceback
import numpy as np
import torch
import step27_prior as s

ART = s.ART
SIGMA_U = s.SIGMA_U
VAR_PHI = s.VAR_PHI
PARTIAL = os.path.join(ART, "step28_grid_partial.npz")
TXT = os.path.join(ART, "step28_grid.txt")
DIAG = os.path.join(ART, "step28_diag_partial.npz")
DIAG_TXT = os.path.join(ART, "step28_diag.txt")
MATS = ((3.0, -1.5), (8.0, -1.5))
MAX_ITER = 400
# A healthy material start finishes well under this. The cap stops a Wolfe
# line search that would otherwise run for hours on a flat node.
EVAL_CAP = 800


def phase_init(cfg, th, users_hat, h0, pick, r, prec, nphase):
    n = len(users_hat)
    grid = torch.arange(nphase) * (2 * math.pi / nphase)
    cols = []
    for g in grid:
        y = s.reduced_mean(cfg, th, users_hat, h0, g.repeat(n), pick)
        e = r - y
        cols.append(torch.einsum("nk,kl,nl->n", e, prec, e))
    return grid[torch.stack(cols, 1).argmin(1)]


def profile(cfg, r, users_hat, h0, pick, d, phi, d_map, phi_map, var_d, prec,
            mat_starts, max_iter=MAX_ITER, nphase=8):
    """Minimise the step27 cost at a fixed offset and angle."""
    n = len(users_hat)
    best_c, best = float("inf"), None
    for eps0, sig0 in mat_starts:
        th_fixed = torch.tensor([d, phi], dtype=r.dtype)
        zt = torch.tensor([eps0, sig0], dtype=r.dtype, requires_grad=True)
        zd = torch.zeros(n, 2, dtype=r.dtype, requires_grad=True)
        th = torch.cat([th_fixed, zt.detach()])
        psi = phase_init(cfg, th, users_hat, h0, pick, r, prec, nphase).clone().requires_grad_(True)
        opt = torch.optim.LBFGS(
            [zt, zd, psi], lr=1, max_iter=max_iter, tolerance_grad=1e-7,
            tolerance_change=1e-11, history_size=20, line_search_fn="strong_wolfe")
        seen = {"n": 0, "c": float("inf"), "zt": None, "zd": None, "psi": None, "cap": False}

        def cost():
            opt.zero_grad()
            th = torch.cat([th_fixed, zt])
            dxy = SIGMA_U * zd
            u = users_hat + torch.cat([dxy, torch.zeros(n, 1)], 1)
            y = s.reduced_mean(cfg, th, u, h0, psi, pick)
            e = r - y
            nll = 0.5 * torch.einsum("nk,kl,nl->", e, prec, e)
            pen = (d - d_map) ** 2 / (2 * var_d) + (phi - phi_map) ** 2 / (2 * VAR_PHI)
            c = nll + (zd ** 2).sum() / 2 + pen
            seen["n"] += 1
            v = float(c.detach())
            if v < seen["c"]:
                seen["c"] = v
                seen["zt"] = zt.detach().clone()
                seen["zd"] = zd.detach().clone()
                seen["psi"] = psi.detach().clone()
            if seen["n"] > EVAL_CAP:
                seen["cap"] = True
                raise RuntimeError("eval cap")
            c.backward()
            return c

        try:
            opt.step(cost)
        except RuntimeError:
            pass
        if seen["cap"]:
            profile.cap_hits = getattr(profile, "cap_hits", 0) + 1
        if seen["zt"] is not None and seen["c"] < best_c:
            best_c = seen["c"]
            best = (float(seen["zt"][0]), float(seen["zt"][1]))
    return best_c, best


def coarse_axes():
    """2 m by 1 deg, covering the normal prior out to several standard deviations."""
    d = np.arange(-14.0, 14.0 + 1e-9, 2.0)
    phi = np.deg2rad(np.arange(-6.0, 6.0 + 1e-9, 1.0))
    return d, phi


def refine_axes(d0, phi0):
    d = d0 + np.arange(-2.0, 2.0 + 1e-9, 0.5)
    phi = phi0 + np.deg2rad(np.arange(-1.0, 1.0 + 1e-9, 0.5))
    return d, phi


def _weights(costs, areas):
    c = np.asarray(costs, dtype=float)
    a = np.asarray(areas, dtype=float)
    ok = np.isfinite(c)
    c, a = c[ok], a[ok]
    if len(c) == 0:
        return None
    w = a * np.exp(-np.clip(c - np.min(c), 0, 60))
    sw = w.sum()
    if not np.isfinite(sw) or sw <= 0:
        return None
    return c, a, w / sw, ok


def posterior(coarse, fine):
    """MAP over every node. Mean over the fine window plus coarse nodes outside it.

    coarse and fine are lists of (offset_m, angle_rad, cost). Cell area keeps
    the finer window from dominating the mean by having more nodes.
    """
    nodes = list(coarse) + list(fine)
    costs = [n[2] for n in nodes]
    i = int(np.nanargmin(costs))
    map_off, map_phi, map_c = nodes[i]
    if not fine:
        return map_off, map_phi, map_c, float("nan")
    ds = [p[0] for p in fine]
    ps = [p[1] for p in fine]
    d0, d1, p0, p1 = min(ds), max(ds), min(ps), max(ps)
    area_c = 2.0 * math.radians(1.0)
    area_f = 0.5 * math.radians(0.5)
    off, cost, area = [], [], []
    for d, p, c in coarse:
        if d0 - 1e-6 <= d <= d1 + 1e-6 and p0 - 1e-9 <= p <= p1 + 1e-9:
            continue
        off.append(d)
        cost.append(c)
        area.append(area_c)
    for d, p, c in fine:
        off.append(d)
        cost.append(c)
        area.append(area_f)
    packed = _weights(cost, area)
    if packed is None:
        return map_off, map_phi, map_c, float("nan")
    _, _, w, ok = packed
    mean_off = float(np.sum(w * np.asarray(off)[ok]))
    return float(map_off), float(map_phi), float(map_c), mean_off


def _eval_grid(cfg, r, u_hat, h0, pick, d_map, phi_map, var_d, prec, d_axis, phi_axis, tag=""):
    nodes = []
    t0 = time.time()
    n_phi = len(phi_axis)
    for dd in d_axis:
        for pp in phi_axis:
            c, _ = profile(
                cfg, r, u_hat, h0, pick, d_map + float(dd), phi_map + float(pp),
                d_map, phi_map, var_d, prec, MATS, max_iter=MAX_ITER, nphase=8)
            nodes.append((float(dd), float(pp), c))
        if tag and len(nodes) % (3 * n_phi) == 0:
            print(f"progress {tag} nodes={len(nodes)} {time.time() - t0:.0f}s", flush=True)
    return nodes


def one_trial(task):
    torch.set_num_threads(1)
    try:
        return _one_trial(task)
    except Exception:
        traceback.print_exc()
        kind, n_beams, n_users, half, name, sid, idx, seed = task[:8]
        return _nan_row(seed, sid, name, n_users)


def _nan_row(seed, sid, name, n_users):
    return {
        "seed": int(seed), "sid": int(sid), "name": str(name), "users": int(n_users),
        "true": np.nan, "map": np.nan, "mean": np.nan, "map_phi": np.nan, "true_phi": np.nan,
        "map_c": np.nan, "truth_c": np.nan, "fit_c": np.nan, "ms": np.nan,
        "bcrb": np.nan, "bcrb_ok": 0.0, "seconds": np.nan, "n_nodes": 0, "edge": 0.0,
        "cap": 0,
    }


def _one_trial(task):
    kind, n_beams, n_users, half, name, sid, idx, seed, err_d, err_phi, bcrb_ref = task
    cfg, th0, U, H0, _ = s.prepare(kind)
    var_d = half ** 2 / 3.0
    users, h0 = U[idx], H0[idx]
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(int(seed))
    th_true = th0.clone()
    th_true[0] = th0[0] + float(rng.normal(0.0, math.sqrt(var_d)))
    th_true[1] = th0[1] + float(rng.normal(0.0, math.sqrt(VAR_PHI)))
    r, pick, _, _ = s.make_report(cfg, th_true, users, h0, n_beams, gen)
    u_hat = users + torch.cat(
        [SIGMA_U * torch.randn(len(idx), 2, generator=gen), torch.zeros(len(idx), 1)], 1)
    prec, _, _ = s.report_precision(cfg.sigma_e, n_beams, r.dtype)
    J, _ = s.total_information(cfg, th_true, users, h0, n_beams)
    bcrb = s.dd_variance((J + s.prior_info(var_d, J.dtype)).detach().numpy())
    bcrb_ok = 0.0
    if np.isfinite(bcrb_ref) and np.isfinite(bcrb):
        bcrb_ok = 1.0 if abs(bcrb - bcrb_ref) <= 1e-4 * max(1.0, abs(bcrb_ref)) else 0.0
    d_map, phi_map = float(th0[0]), float(th0[1])
    t0 = time.time()
    profile.cap_hits = 0
    d_axis, phi_axis = coarse_axes()
    tag = f"N={n_users} {name} seed={seed}"
    coarse = _eval_grid(cfg, r, u_hat, h0, pick, d_map, phi_map, var_d, prec, d_axis, phi_axis, tag)
    bi = int(np.nanargmin([n[2] for n in coarse]))
    fine = _eval_grid(
        cfg, r, u_hat, h0, pick, d_map, phi_map, var_d, prec,
        *refine_axes(coarse[bi][0], coarse[bi][1]), tag + " fine")
    map_off, map_phi, map_c, mean_off = posterior(coarse, fine)
    truth_c, _ = profile(
        cfg, r, u_hat, h0, pick, float(th_true[0]), float(th_true[1]),
        d_map, phi_map, var_d, prec, MATS, max_iter=MAX_ITER, nphase=8)
    true_off = float(th_true[0] - d_map)
    true_phi = float(th_true[1] - phi_map)
    if np.isfinite(err_d):
        fit_c, _ = profile(
            cfg, r, u_hat, h0, pick,
            float(th_true[0]) + float(err_d), float(th_true[1]) + float(err_phi),
            d_map, phi_map, var_d, prec, MATS, max_iter=MAX_ITER, nphase=8)
        ms = true_off + float(err_d)
    else:
        fit_c, ms = float("nan"), float("nan")
    edge = 1.0 if abs(abs(map_off) - 14.0) < 0.11 or abs(abs(map_phi) - math.radians(6.0)) < 1e-4 else 0.0
    return {
        "seed": int(seed), "sid": int(sid), "name": str(name), "users": int(n_users),
        "true": true_off, "map": float(map_off), "mean": float(mean_off),
        "map_phi": float(map_phi), "true_phi": true_phi,
        "map_c": float(map_c), "truth_c": float(truth_c), "fit_c": float(fit_c),
        "ms": float(ms), "bcrb": float(bcrb), "bcrb_ok": bcrb_ok,
        "seconds": time.time() - t0, "n_nodes": len(coarse) + len(fine), "edge": edge,
        "cap": int(getattr(profile, "cap_hits", 0)),
    }


def diagnose_trial(task):
    """Profile only the true wall and the multi-start wall. Same cost as the grid."""
    torch.set_num_threads(1)
    kind, n_beams, n_users, half, name, sid, idx, seed, err_d, err_phi, bcrb_ref = task
    try:
        return _diagnose(task)
    except Exception:
        traceback.print_exc()
        return {
            "seed": int(seed), "sid": int(sid), "name": str(name), "users": int(n_users),
            "true": np.nan, "ms": np.nan, "truth_c": np.nan, "fit_c": np.nan,
            "bcrb": np.nan, "bcrb_ok": 0.0, "seconds": np.nan, "cap": 0,
        }


def _diagnose(task):
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
        [SIGMA_U * torch.randn(len(idx), 2, generator=gen), torch.zeros(len(idx), 1)], 1)
    prec, _, _ = s.report_precision(cfg.sigma_e, n_beams, r.dtype)
    J, _ = s.total_information(cfg, th_true, users, h0, n_beams)
    bcrb = s.dd_variance((J + s.prior_info(var_d, J.dtype)).detach().numpy())
    bcrb_ok = 0.0
    if np.isfinite(bcrb_ref) and np.isfinite(bcrb):
        bcrb_ok = 1.0 if abs(bcrb - bcrb_ref) <= 1e-4 * max(1.0, abs(bcrb_ref)) else 0.0
    d_map, phi_map = float(th0[0]), float(th0[1])
    true_off = float(th_true[0] - d_map)
    t0 = time.time()
    profile.cap_hits = 0
    truth_c, _ = profile(
        cfg, r, u_hat, h0, pick, float(th_true[0]), float(th_true[1]),
        d_map, phi_map, var_d, prec, MATS, max_iter=MAX_ITER, nphase=8)
    if np.isfinite(err_d):
        fit_c, _ = profile(
            cfg, r, u_hat, h0, pick,
            float(th_true[0]) + float(err_d), float(th_true[1]) + float(err_phi),
            d_map, phi_map, var_d, prec, MATS, max_iter=MAX_ITER, nphase=8)
        ms = true_off + float(err_d)
    else:
        fit_c, ms = float("nan"), float("nan")
    return {
        "seed": int(seed), "sid": int(sid), "name": str(name), "users": int(n_users),
        "true": true_off, "ms": float(ms), "truth_c": float(truth_c), "fit_c": float(fit_c),
        "bcrb": float(bcrb), "bcrb_ok": bcrb_ok,
        "seconds": time.time() - t0, "cap": int(getattr(profile, "cap_hits", 0)),
    }


def _ms_lookup():
    """Multi-start offset error and conditional variance, keyed by trial seed."""
    z = np.load(s.FIT_PARTIAL, allow_pickle=True)
    out = {}
    for kind, beams, half, seed, err, bcrb in zip(
            z["kind"], z["beams"], z["half"], z["seed"], z["err"], z["bcrb"]):
        if str(kind) != "nb" or int(beams) != 4 or abs(float(half) - 6.0) > 1e-6:
            continue
        out[int(seed)] = (float(err[0]), float(err[1]), float(bcrb))
    return out


def campaign_tasks():
    """Table VII seeds, plus a 20-user row that step27 did not fit."""
    lookup = _ms_lookup()
    tasks = []
    for t in s.fit_tasks():
        kind, n_beams, n_users, half, name, sid, idx, seed = t
        if kind != "nb" or n_beams != 4 or abs(half - 6.0) > 1e-6:
            continue
        err_d, err_phi, bcrb = lookup[int(seed)]
        tasks.append((kind, n_beams, n_users, half, name, sid, idx, seed, err_d, err_phi, bcrb))
    cfg, th0, U, H0, los = s.prepare("nb")
    del cfg, th0, U, H0
    pools = {"LOS": np.where(los)[0], "NLOS": np.where(~los)[0]}
    rng = np.random.default_rng(28)
    seed = 281000
    for name, pool in pools.items():
        sets = [rng.choice(pool, 20, replace=False) for _ in range(16)]
        for sid, idx in enumerate(sets):
            for _ in range(3):
                seed += 1
                tasks.append(("nb", 4, 20, 6.0, name, sid, idx, seed, np.nan, np.nan, np.nan))
    return tasks


def _rows_from_npz(path):
    if not os.path.exists(path):
        return []
    z = np.load(path, allow_pickle=True)
    rows = []
    n = len(z["seed"])
    for i in range(n):
        rows.append({k: (str(z[k][i]) if k == "name" else z[k][i]) for k in z.files})
        rows[-1]["seed"] = int(rows[-1]["seed"])
        rows[-1]["sid"] = int(rows[-1]["sid"])
        rows[-1]["users"] = int(rows[-1]["users"])
        rows[-1]["n_nodes"] = int(rows[-1]["n_nodes"])
        if "cap" not in rows[-1]:
            rows[-1]["cap"] = -1
        else:
            rows[-1]["cap"] = int(rows[-1]["cap"])
    return rows


def save_rows(rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    payload = {}
    for k in keys:
        if k == "name":
            payload[k] = np.array([r[k] for r in rows])
        elif k in ("seed", "sid", "users", "n_nodes"):
            payload[k] = np.array([r[k] for r in rows], dtype=np.int64)
        else:
            payload[k] = np.array([r[k] for r in rows], dtype=float)
    np.savez_compressed(PARTIAL, **payload)


def _rmse_ci(err, sid, rng):
    ok = np.isfinite(err)
    if ok.sum() == 0:
        return float("nan"), float("nan"), float("nan"), 0
    e, sid = err[ok], sid[ok]
    rmse = math.sqrt(np.mean(e ** 2))
    sets = np.unique(sid)
    boot = []
    for _ in range(2000):
        pick = rng.choice(sets, len(sets), replace=True)
        m = np.concatenate([np.where(sid == s)[0] for s in pick])
        boot.append(math.sqrt(np.mean(e[m] ** 2)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return rmse, float(lo), float(hi), int(ok.sum())


def summarise(rows):
    lines = [
        "Grid profile, same likelihood as step27. "
        "Coarse grid 2 m by 1 deg on [-14, 14] m by [-6, 6] deg. "
        "Fine window 0.5 m by 0.5 deg on +/-2 m by +/-1 deg around the coarse minimum. "
        f"Nuisance LBFGS max_iter={MAX_ITER}, two material starts, 8 phase starts. "
        "truth_c and fit_c use that same profile. The oracle is not an estimate."
    ]
    keys = sorted(set((int(r["users"]), r["name"]) for r in rows))
    for n_users, name in keys:
        sub = [r for r in rows if int(r["users"]) == n_users and r["name"] == name]
        true = np.array([r["true"] for r in sub], dtype=float)
        sid = np.array([r["sid"] for r in sub])
        rng = np.random.default_rng(28000 + n_users * 10 + (0 if name == "LOS" else 1))
        g_err = np.array([r["map"] for r in sub], dtype=float) - true
        m_err = np.array([r["mean"] for r in sub], dtype=float) - true
        ms = np.array([r["ms"] for r in sub], dtype=float)
        ms_err = ms - true
        bcrb = np.array([r["bcrb"] for r in sub], dtype=float)
        map_c = np.array([r["map_c"] for r in sub], dtype=float)
        truth_c = np.array([r["truth_c"] for r in sub], dtype=float)
        fit_c = np.array([r["fit_c"] for r in sub], dtype=float)
        g_rmse, g_lo, g_hi, n = _rmse_ci(g_err, sid, rng)
        m_rmse, m_lo, m_hi, _ = _rmse_ci(m_err, sid, rng)
        ms_rmse, ms_lo, ms_hi, n_ms = _rmse_ci(ms_err, sid, rng)
        gap = map_c - truth_c
        both = np.isfinite(gap)
        truth_better = int(np.sum(gap[both] > 0.5))
        map_better = int(np.sum(gap[both] < -0.5))
        tie = int(np.sum(np.abs(gap[both]) <= 0.5))
        fit_gap = fit_c - truth_c
        have_fit = np.isfinite(fit_gap)
        distant = have_fit & np.isfinite(bcrb) & (np.abs(ms_err) > 3 * np.sqrt(np.maximum(bcrb, 0)))
        dist_truth = int(np.sum(distant & (fit_gap > 0.5)))
        dist_fit = int(np.sum(distant & (fit_gap < -0.5)))
        bcrb_ok = int(np.sum([r["bcrb_ok"] for r in sub]))
        n_b = int(np.sum(np.isfinite(np.array([r["ms"] for r in sub], dtype=float))))
        edge = int(np.sum([r["edge"] for r in sub]))
        secs = np.array([r["seconds"] for r in sub], dtype=float)
        lines.append(
            f"N={n_users} {name} n={n} grid={g_rmse:.3f} [{g_lo:.3f}, {g_hi:.3f}] "
            f"mean={m_rmse:.3f} [{m_lo:.3f}, {m_hi:.3f}] "
            f"multistart={ms_rmse:.3f} [{ms_lo:.3f}, {ms_hi:.3f}] n_ms={n_ms} "
            f"truth_better={truth_better} map_better={map_better} tie={tie} "
            f"median_map_minus_truth={np.median(gap[both]) if both.any() else float('nan'):.2f} "
            f"distant={int(distant.sum())}/{n_b} distant_truth_lower={dist_truth} "
            f"distant_fit_lower={dist_fit} bcrb_ok={bcrb_ok}/{n_b} edge={edge} "
            f"sec_med={np.nanmedian(secs):.1f}"
        )
    return lines


def run_campaign():
    from multiprocessing import get_context
    torch.set_num_threads(1)
    tasks = campaign_tasks()
    rows = _rows_from_npz(PARTIAL)
    have = set(int(r["seed"]) for r in rows)
    todo = [t for t in tasks if int(t[7]) not in have]
    print(f"grid resume {len(rows)} done, {len(todo)} left, tasks {len(tasks)}", flush=True)
    t0 = time.time()
    if todo:
        workers = int(os.environ.get("WORKERS", "14"))
        with get_context("fork").Pool(workers) as pool:
            for k, row in enumerate(pool.imap_unordered(one_trial, todo, chunksize=1), 1):
                rows.append(row)
                if k % 4 == 0 or k == len(todo):
                    save_rows(rows)
                    print(
                        f"grid {k}/{len(todo)} new, {len(rows)} stored, {time.time() - t0:.0f}s "
                        f"last N={row['users']} {row['name']} sec={row['seconds']} "
                        f"bcrb_ok={row['bcrb_ok']} map_c={row['map_c']} truth_c={row['truth_c']}",
                        flush=True,
                    )
    lines = summarise(rows)
    lines.append(f"grid seconds {time.time() - t0:.0f}")
    text = "\n".join(lines) + "\n"
    open(TXT, "w").write(text)
    print(text, flush=True)


def time_one():
    tasks = campaign_tasks()
    for want in (10, 200):
        task = next(t for t in tasks if t[2] == want and t[4] == ("LOS" if want == 10 else "NLOS"))
        out = one_trial(task)
        gap = out["map_c"] - out["truth_c"]
        fit_gap = out["fit_c"] - out["truth_c"]
        print(
            f"N={out['users']} {out['name']} nodes={out['n_nodes']} sec={out['seconds']:.1f} "
            f"true={out['true']:.2f} map={out['map']:.2f} mean={out['mean']:.2f} ms={out['ms']:.2f} "
            f"map_c={out['map_c']:.2f} truth_c={out['truth_c']:.2f} fit_c={out['fit_c']:.2f} "
            f"map_minus_truth={gap:.2f} fit_minus_truth={fit_gap:.2f} bcrb_ok={out['bcrb_ok']}",
            flush=True,
        )


def _save_diag(rows):
    if not rows:
        return
    np.savez_compressed(
        DIAG,
        seed=np.array([r["seed"] for r in rows], dtype=np.int64),
        sid=np.array([r["sid"] for r in rows], dtype=np.int64),
        name=np.array([r["name"] for r in rows]),
        users=np.array([r["users"] for r in rows], dtype=np.int64),
        true=np.array([r["true"] for r in rows], dtype=float),
        ms=np.array([r["ms"] for r in rows], dtype=float),
        truth_c=np.array([r["truth_c"] for r in rows], dtype=float),
        fit_c=np.array([r["fit_c"] for r in rows], dtype=float),
        bcrb=np.array([r["bcrb"] for r in rows], dtype=float),
        bcrb_ok=np.array([r["bcrb_ok"] for r in rows], dtype=float),
        seconds=np.array([r["seconds"] for r in rows], dtype=float),
        cap=np.array([r["cap"] for r in rows], dtype=np.int64),
    )


def _load_diag():
    rows = []
    if os.path.exists(PARTIAL):
        for r in _rows_from_npz(PARTIAL):
            if not np.isfinite(float(r["fit_c"])):
                continue
            rows.append({
                "seed": int(r["seed"]), "sid": int(r["sid"]), "name": str(r["name"]),
                "users": int(r["users"]), "true": float(r["true"]), "ms": float(r["ms"]),
                "truth_c": float(r["truth_c"]), "fit_c": float(r["fit_c"]),
                "bcrb": float(r["bcrb"]), "bcrb_ok": float(r["bcrb_ok"]),
                "seconds": 0.0, "cap": -1,
            })
    if os.path.exists(DIAG):
        z = np.load(DIAG, allow_pickle=True)
        have = {r["seed"] for r in rows}
        for i in range(len(z["seed"])):
            seed = int(z["seed"][i])
            if seed in have:
                continue
            rows.append({
                "seed": seed, "sid": int(z["sid"][i]), "name": str(z["name"][i]),
                "users": int(z["users"][i]), "true": float(z["true"][i]), "ms": float(z["ms"][i]),
                "truth_c": float(z["truth_c"][i]), "fit_c": float(z["fit_c"][i]),
                "bcrb": float(z["bcrb"][i]), "bcrb_ok": float(z["bcrb_ok"][i]),
                "seconds": float(z["seconds"][i]), "cap": int(z["cap"][i]),
            })
    return rows


def summarise_diag(rows):
    lines = [
        "Cost at the true wall versus cost at the multi-start wall. "
        "Same profile as the grid: two material starts, LBFGS, eval cap "
        f"{EVAL_CAP}. A positive fit_minus_truth means the true wall is cheaper, so the search missed it."
    ]
    keys = sorted(set((int(r["users"]), r["name"]) for r in rows))
    for n_users, name in keys:
        sub = [r for r in rows if int(r["users"]) == n_users and r["name"] == name]
        fit_c = np.array([r["fit_c"] for r in sub], dtype=float)
        truth_c = np.array([r["truth_c"] for r in sub], dtype=float)
        ms = np.array([r["ms"] for r in sub], dtype=float)
        true = np.array([r["true"] for r in sub], dtype=float)
        bcrb = np.array([r["bcrb"] for r in sub], dtype=float)
        gap = fit_c - truth_c
        ok = np.isfinite(gap)
        err = np.abs(ms - true)
        distant = ok & np.isfinite(bcrb) & (err > 3.0 * np.sqrt(np.maximum(bcrb, 0.0)))
        lines.append(
            f"N={n_users} {name} n={int(ok.sum())} "
            f"truth_cheaper={int(np.sum(gap[ok] > 0.5))} "
            f"fit_cheaper={int(np.sum(gap[ok] < -0.5))} "
            f"tie={int(np.sum(np.abs(gap[ok]) <= 0.5))} "
            f"median_gap={np.median(gap[ok]):.2f} "
            f"distant={int(distant.sum())} "
            f"distant_truth_cheaper={int(np.sum(distant & (gap > 0.5)))} "
            f"distant_fit_cheaper={int(np.sum(distant & (gap < -0.5)))} "
            f"bcrb_ok={int(np.sum([r['bcrb_ok'] for r in sub]))}"
        )
    return lines


def run_diagnose():
    from multiprocessing import get_context
    torch.set_num_threads(1)
    tasks = [t for t in campaign_tasks() if np.isfinite(t[8])]
    rows = _load_diag()
    have = {int(r["seed"]) for r in rows}
    todo = [t for t in tasks if int(t[7]) not in have]
    print(f"diag resume {len(rows)} done, {len(todo)} left, paired {len(tasks)}", flush=True)
    t0 = time.time()
    if todo:
        workers = int(os.environ.get("WORKERS", "4"))
        with get_context("fork").Pool(workers) as pool:
            for k, row in enumerate(pool.imap_unordered(diagnose_trial, todo, chunksize=1), 1):
                rows.append(row)
                if k % 8 == 0 or k == len(todo):
                    _save_diag(rows)
                    print(
                        f"diag {k}/{len(todo)} new, {len(rows)} stored, {time.time() - t0:.0f}s "
                        f"last N={row['users']} {row['name']} sec={row['seconds']} "
                        f"bcrb_ok={row['bcrb_ok']} fit_c={row['fit_c']} truth_c={row['truth_c']}",
                        flush=True,
                    )
    lines = summarise_diag(rows)
    lines.append(f"diag seconds {time.time() - t0:.0f}")
    text = "\n".join(lines) + "\n"
    open(DIAG_TXT, "w").write(text)
    print(text, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mode = sys.argv[1] if len(sys.argv) > 1 else "time"
    if mode == "time":
        time_one()
    elif mode == "run":
        run_campaign()
    elif mode == "diagnose":
        run_diagnose()
    elif mode == "summarise":
        print("\n".join(summarise(_rows_from_npz(PARTIAL))), flush=True)
    else:
        raise SystemExit("usage: step28_grid.py time|run|diagnose|summarise")
