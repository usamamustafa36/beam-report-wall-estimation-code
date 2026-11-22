"""Step 22: Table II protocol on the two-path model, power averaged over 20 MHz.

Same user-set generator and wall prior as step4b_mle_far.py (rng 7). sigma_e = 1 dB
only, 30 sets x 2 noise draws, with a paired 1 dB quantised copy. Starts at the map
with d in {0, +-3, +-6} and eps_r in {3, 8}. The direct path is on for LOS users only.
The stored h0 is not in the channel: it has no per-path delay.
N = 10, sigma_u = 1 m, facade B, 3.5 GHz. Workers = 10, one thread each.

Output: artifacts/step22_wideband_map.txt / .npz, with a partial file rewritten as results arrive.
"""
import math
import os
import time
import numpy as np
import torch
import rsrp_model as rm
import wideband as wb
from step4_mle import fit, setup
from step4b_mle_far import SIGMA_U, N_USERS, SETS, B_BOOT, best_fit, boot_ci

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
BAND = 20.0
WORKERS = 10
SIGMA_E_RUN = [1.0]
DRAWS_RUN = 2
PARTIAL = os.path.join(ART, "step22_wideband_map_partial.npz")


def trial(task):
    torch.set_num_threads(1)
    wb.install(BAND)
    name, se, sid, idx, seed = task
    cfg, th0, U, H0, _ = setup("B")
    cfg.sigma_e = se
    cfg.use_direct = name == "LOS"
    u = U[idx]
    h0 = torch.zeros(len(idx), cfg.n_ant, dtype=torch.complex128)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    th_true = th0.clone()
    th_true[0] += rng.uniform(-6, 6)
    th_true[1] += math.radians(rng.uniform(-2, 2))
    J = rm.fisher_robust(cfg, th_true, u, h0, sigma_u=SIGMA_U).sum(0).numpy()
    mu = rm.rsrp_mean_phase(cfg, th_true, u, h0, torch.zeros(len(idx)))
    r = mu + se * torch.randn(mu.shape, generator=gen)
    u_hat = u + torch.cat([SIGMA_U * torch.randn(len(idx), 2, generator=gen), torch.zeros(len(idx), 1)], 1)
    out = {"crb": np.diag(np.linalg.inv(J))}
    th = best_fit(cfg, r, u_hat, h0, th0, False)
    out["err"] = (th - th_true).numpy() if th is not None else np.full(4, np.nan)
    if se == 1.0:
        rq = torch.clamp(torch.round(r), -156.0, -31.0)
        thq = best_fit(cfg, rq, u_hat, h0, th0, True)
        out["err_q"] = (thq - th_true).numpy() if thq is not None else np.full(4, np.nan)
        out["crb_q"] = out["crb"] * (se ** 2 + 1 / 12) / se ** 2
    return task[:3] + (seed,), out


def tasks_all():
    _, _, _, _, los = setup("B")
    pools = {"LOS": np.where(los)[0], "NLOS": np.where(~los)[0]}
    rng = np.random.default_rng(7)
    tasks, seed = [], 40000
    for name, pool in pools.items():
        sets = [rng.choice(pool, N_USERS, replace=False) for _ in range(SETS)]
        for se in SIGMA_E_RUN:
            for sid, idx in enumerate(sets):
                for _ in range(DRAWS_RUN):
                    seed += 1
                    tasks.append((name, se, sid, idx, seed))
    return tasks


def load_done():
    if not os.path.exists(PARTIAL):
        return {}, []
    z = np.load(PARTIAL, allow_pickle=True)
    rows = []
    done = {}
    for i, seed in enumerate(z["seed"].tolist()):
        rec = {
            "name": str(z["name"][i]), "se": float(z["se"][i]), "sid": int(z["sid"][i]),
            "seed": int(seed), "err": z["err"][i], "crb": z["crb"][i],
        }
        if "err_q" in z.files and np.isfinite(z["err_q"][i][0]):
            rec["err_q"] = z["err_q"][i]
            rec["crb_q"] = z["crb_q"][i]
        rows.append(rec)
        done[int(seed)] = rec
    return done, rows


def save_rows(rows):
    if not rows:
        return
    kw = dict(
        name=np.array([r["name"] for r in rows]),
        se=np.array([r["se"] for r in rows]),
        sid=np.array([r["sid"] for r in rows]),
        seed=np.array([r["seed"] for r in rows]),
        err=np.stack([r["err"] for r in rows]),
        crb=np.stack([r["crb"] for r in rows]),
    )
    eq = np.full((len(rows), 4), np.nan)
    cq = np.full((len(rows), 4), np.nan)
    for i, r in enumerate(rows):
        if "err_q" in r:
            eq[i], cq[i] = r["err_q"], r["crb_q"]
    kw["err_q"], kw["crb_q"] = eq, cq
    np.savez_compressed(PARTIAL, **kw)


def write_report(rows, seconds):
    res = {}
    for r in rows:
        variants = [("", r["err"], r["crb"])]
        if "err_q" in r:
            variants.append(("_q", r["err_q"], r["crb_q"]))
        for tag, e, c in variants:
            d = res.setdefault((r["name"], r["se"], tag), {"err": [], "crb": [], "sid": []})
            d["err"].append(e)
            d["crb"].append(c)
            d["sid"].append(r["sid"])
    lines = [f"two-path wideband MAP, B={BAND:g} MHz, direct path on for LOS only, h0 unused, "
             f"{len(rows)} trials, {seconds:.0f}s"]
    brng = np.random.default_rng(1)
    save = {}
    for (name, se, tag), d in sorted(res.items()):
        err, crb, sid = np.array(d["err"]), np.array(d["crb"]), np.array(d["sid"])
        ok = np.isfinite(err[:, 0])
        rmse = np.sqrt(np.nanmean(err ** 2, 0))
        bound = np.sqrt(np.nanmean(crb, 0))
        (rl, rh), (bl, bh) = boot_ci(err, crb, sid, 0, brng)
        out3 = np.mean(np.abs(err[ok, 0]) > 3 * np.sqrt(crb[ok, 0])) if ok.any() else float("nan")
        key = f"{name}_se{se}{tag}"
        save[key + "_err"], save[key + "_crb"], save[key + "_sid"] = err, crb, sid
        lines.append(
            f"{name:4s} sigma_e={se:4.2f} dB{' quantised' if tag else '          '}: d RMSE {rmse[0]:.2f} m "
            f"[{rl:.2f}, {rh:.2f}] / bound {bound[0]:.2f} [{bl:.2f}, {bh:.2f}]; phi RMSE "
            f"{math.degrees(rmse[1]):.2f} / {math.degrees(bound[1]):.2f} deg; |d err| > 3 sqrt(CRB) in "
            f"{100 * out3:.0f}%  [{ok.sum()} fits, {(~ok).sum()} failed, {len(np.unique(sid))} sets]"
        )
    text = "\n".join(lines) + "\n"
    open(os.path.join(ART, "step22_wideband_map.txt"), "w").write(text)
    np.savez_compressed(os.path.join(ART, "step22_wideband_map.npz"), **save)
    print(text, flush=True)


def main():
    from multiprocessing import get_context
    done, rows = load_done()
    pending = [t for t in tasks_all() if t[-1] not in done]
    print(f"step22 pending {len(pending)}  already {len(done)}", flush=True)
    t0 = time.time()
    if pending:
        ctx = get_context("fork")
        with ctx.Pool(WORKERS) as pool:
            for i, (key, out) in enumerate(pool.imap_unordered(trial, pending, chunksize=1), 1):
                rec = {"name": key[0], "se": key[1], "sid": key[2], "seed": key[3],
                       "err": out["err"], "crb": out["crb"]}
                if "err_q" in out:
                    rec["err_q"], rec["crb_q"] = out["err_q"], out["crb_q"]
                rows.append(rec)
                if i % 10 == 0 or i == len(pending):
                    save_rows(rows)
                    print(f"step22 {len(rows)} / {len(rows) + len(pending) - i}  {time.time() - t0:.0f}s", flush=True)
    write_report(rows, time.time() - t0)
    print("DONE step22", flush=True)


if __name__ == "__main__":
    main()
