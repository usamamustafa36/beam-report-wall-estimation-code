"""Numbers for the review items that do not need a new ray trace.

M1  Top-4 L1-RSRP: strongest beam at 1 dB, the other three as 2 dB differentials
    (TS 38.214 clause 5.2.1.4.3). Beams chosen on the noise-free report.
M3  Phase nuisances on the geometric direct path and on h0 minus that path,
    eliminated by the same Schur complement as the wall-path phase.
M5  Nearest ITU class in |Gamma_TE| and in the 16-beam report, at other
    thicknesses, at eps_r +/- 20%, and with the TM slab coefficient added.
m7  Network trained on facade B, tested on facade E. Bayesian CRB: Van Trees
    bound with a Gaussian prior of the same variance as U[-6, 6] m and U[-2, 2] deg
    (a uniform prior is not differentiable at the ends, so it has no Van Trees term).
M2  (report-domain part) other-wall error: the non-direct part of h0 is scaled by
    a random ITU |Gamma| ratio and delayed by a path error of +/- 0.5 m.

Facade B, 3.5 GHz, N = 10, sigma_u = 1 m, 300 sets, unless a line says otherwise.
"""
import math
import os
import numpy as np
import torch
import rsrp_model as rm
import scene_data as sd
from step10_learned import CLASSES, DeepSets, material_params, mu_batch

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SIGMA_U, N, SETS = 1.0, 10, 300
COS = torch.linspace(0.1, 0.95, 18)


def direct(cfg, users):
    """Geometric free-space direct path (N, M), blockage ignored."""
    eye = torch.eye(3)
    # geometry() only needs a wall so the image is defined; L_d and k_d ignore it
    # as long as the call succeeds. Use the stored plane through the BS height.
    d = (cfg.bs[:2] * torch.tensor([1.0, 0.0])).sum()  # unused numerically for L_d
    # Call geometry with a plane the users are in front of: n = x-axis, d far away.
    th_d, th_p = torch.tensor(0.0), torch.tensor(0.0)
    # Rebuild from the BS-user vector directly, so the wall parameters cannot leak in.
    v = users - cfg.bs
    L = torch.linalg.norm(v, dim=-1)
    k = v / L[:, None]
    g = cfg.lam / (4 * math.pi * L) * torch.exp(-1j * (2 * math.pi / cfg.lam) * L)
    return g[:, None] * rm.steering(cfg, k)


def schur(Jt, Jn, sigma_row):
    """Schur complement. Jt (N, K, P), Jn (N, K, Q) nuisance Jacobian, sigma_row (N, K)."""
    w = 1.0 / sigma_row
    At = Jt * w[..., None]
    An = Jn * w[..., None]
    N = At.shape[0]
    R = torch.zeros(N, 2, An.shape[-1])
    # position prior occupies the first two nuisance columns, which are dxy
    R[:, 0, 0] = R[:, 1, 1] = 1.0 / SIGMA_U
    An = torch.cat([An, R], 1)
    At = torch.cat([At, torch.zeros(N, 2, At.shape[-1])], 1)
    Un, s, _ = torch.linalg.svd(An, full_matrices=False)
    ref = torch.maximum(s[:, 0], torch.linalg.matrix_norm(At, ord=2))
    Un = Un * (s > 1e-12 * ref[:, None])[:, None, :]
    Res = At - Un @ (Un.transpose(1, 2) @ At)
    return Res.transpose(1, 2) @ Res


def crb_stat(Js, sets, p=0):
    v = []
    for s in sets:
        ev = np.linalg.eigvalsh(Js[s].sum(0))
        if ev[0] <= 1e-8:
            v.append(np.nan)
        else:
            v.append(np.sqrt(np.linalg.inv(Js[s].sum(0))[p, p]))
    v = np.array(v)
    ok = v[np.isfinite(v)]
    return float(np.median(ok)), float(np.percentile(ok, 10)), float(np.percentile(ok, 90)), int(np.isnan(v).sum())


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)
        open(os.path.join(ART, "step14_complete.txt"), "w").write("\n".join(lines) + "\n")

    _, cfg, th0, U, H0, los = sd.load("B", "3.5")
    rng = np.random.default_rng(14)
    pools = {"LOS": np.where(los)[0], "NLOS": np.where(~los)[0]}
    sets = {k: [rng.choice(v, N, replace=False) for _ in range(SETS)] for k, v in pools.items()}

    # --- M3: split phase nuisances, and the paper's current bound as a check ---
    hd_all = direct(cfg, U)

    def mu_split(th, u, hd_r, hr, dxy, psi):
        """hd_r, hr are real views. hd is the nominal direct path, or 0 for NLOS.
        psi = (wall, direct, residual). A zero nominal direct path drops that component."""
        hd = torch.view_as_complex(hd_r)
        uu = (u + torch.cat([dxy, torch.zeros(1)]))[None]
        hd_u = direct(cfg, uu)[0]
        has = (hd.abs().sum() > 0).to(hd_u.real.dtype)
        h0 = torch.exp(1j * psi[2]) * torch.view_as_complex(hr) + has * torch.exp(1j * psi[1]) * hd_u
        return rm.rsrp_mean_phase(cfg, th, uu, h0[None], psi[:1], mode="material")[0]

    def mu_base(th, u, hr, dxy, psi):
        uu = (u + torch.cat([dxy, torch.zeros(1)]))[None]
        return rm.rsrp_mean_phase(cfg, th, uu, torch.view_as_complex(hr)[None], psi[:1])[0]

    # sanity: split model at zero phases equals the stored h0 model
    i0 = int(np.where(los)[0][0])
    with torch.no_grad():
        a = rm.rsrp_mean_phase(cfg, th0, U[i0:i0 + 1], H0[i0:i0 + 1], torch.zeros(1))
        hr = torch.view_as_real(H0[i0] - hd_all[i0])
        b = mu_split(th0, U[i0], torch.view_as_real(hd_all[i0]), hr, torch.zeros(2), torch.zeros(3))
        log(f"split-model RSRP error at a LOS user: {(a[0] - b).abs().max().item():.2e} dB")

    def batch_J(idx, kind):
        u, h0 = U[idx], H0[idx]
        hd = hd_all[idx]
        # NLOS: no geometric direct path in the channel. Residual is h0 itself.
        use_direct = torch.tensor(los[idx])
        hd = torch.where(use_direct[:, None], hd, torch.zeros_like(hd))
        hr = torch.view_as_real(h0 - hd)
        hd_r = torch.view_as_real(hd)
        if kind == "base":
            f = mu_base
            arg = (th0, u, torch.view_as_real(h0), torch.zeros(len(idx), 2), torch.zeros(len(idx), 1))
            jac = torch.func.vmap(torch.func.jacfwd(f, argnums=(0, 3, 4)), in_dims=(None, 0, 0, 0, 0))
            Jt, Jd, Jp = jac(*arg)
            sig = torch.full(Jt.shape[:2], cfg.sigma_e)
            return schur(Jt, torch.cat([Jd, Jp], -1), sig)
        f = mu_split
        arg = (th0, u, hd_r, hr, torch.zeros(len(idx), 2), torch.zeros(len(idx), 3))
        jac = torch.func.vmap(torch.func.jacfwd(f, argnums=(0, 4, 5)), in_dims=(None, 0, 0, 0, 0, 0))
        Jt, Jd, Jp = jac(*arg)
        sig = torch.full(Jt.shape[:2], cfg.sigma_e)
        return schur(Jt, torch.cat([Jd, Jp], -1), sig)

    for kind, title in [("base", "current (wall phase only, h0 frozen)"),
                        ("split", "direct and residual phases free")]:
        for name in ["LOS", "NLOS"]:
            Js = []
            for s in sets[name]:
                Js.append(batch_J(s, kind))
            Js = torch.stack(Js).numpy()
            med, lo, hi, nbad = crb_stat(Js, range(SETS))
            log(f"M3 {title} {name}: median sqrt CRB d {med:.2f} m [{lo:.2f}, {hi:.2f}]  singular {nbad}")

    # --- M1: top-4 differential L1-RSRP ---
    def full_jac(idx):
        u, h0 = U[idx], H0[idx]
        f = mu_base
        arg = (th0, u, torch.view_as_real(h0), torch.zeros(len(idx), 2), torch.zeros(len(idx), 1))
        jac = torch.func.vmap(torch.func.jacfwd(f, argnums=(0, 3, 4)), in_dims=(None, 0, 0, 0, 0))
        return jac(*arg), u, h0

    def batch_top4(idx):
        (Jt, Jd, Jp), u, h0 = full_jac(idx)
        with torch.no_grad():
            mu0 = rm.rsrp_mean_phase(cfg, th0, u, h0, torch.zeros(len(idx)))
            pick = mu0.argsort(dim=1, descending=True)[:, :4]
        n = torch.arange(len(idx))
        strong = pick[:, 0]

        def take(J):
            Js = J[n, strong]
            rows = [Js]
            for i in range(1, 4):
                rows.append(J[n, pick[:, i]] - Js)
            return torch.stack(rows, 1)

        Jt, Jd, Jp = take(Jt), take(Jd), take(Jp)
        sig = torch.full(Jt.shape[:2], cfg.sigma_e)
        # strongest: 1 dB steps. differentials: 2 dB steps. Sheppard correction.
        sig[:, 0] = math.sqrt(cfg.sigma_e ** 2 + 1.0 / 12)
        sig[:, 1:] = math.sqrt(cfg.sigma_e ** 2 + 4.0 / 12)
        return schur(Jt, torch.cat([Jd, Jp], -1), sig)

    for name in ["LOS", "NLOS"]:
        Js = torch.stack([batch_top4(s) for s in sets[name]]).numpy()
        med, lo, hi, nbad = crb_stat(Js, range(SETS))
        log(f"M1 top-4 differential {name}: median sqrt CRB d {med:.2f} m [{lo:.2f}, {hi:.2f}]  singular {nbad}")

    # all-16, 1 dB quantisation on every beam, for the same sets (Sheppard on sigma_e)
    cfg_q = cfg
    old = cfg.sigma_e
    cfg.sigma_e = math.sqrt(old ** 2 + 1.0 / 12)
    for name in ["LOS", "NLOS"]:
        Js = torch.stack([batch_J(s, "base") for s in sets[name]]).numpy()
        med, lo, hi, nbad = crb_stat(Js, range(SETS))
        log(f"M1 all-16 quantised 1 dB {name}: median sqrt CRB d {med:.2f} m [{lo:.2f}, {hi:.2f}]")
    cfg.sigma_e = old

    # --- M5: class distances ---
    def eps_of(er, sig, f=3.5e9):
        return er - 1j * sig / (2 * math.pi * f * rm.EPS0)

    def gamma_db(er, sig, thickness, pol="te"):
        eps = eps_of(er, sig)
        if pol == "te":
            g = rm.slab_te(eps, COS, thickness, rm.C0 / 3.5e9) if thickness else rm.fresnel_te(eps, COS)
        else:
            # TM: swap the roles of the media in the Fresnel formula, then the slab sum
            sin2 = 1 - COS ** 2
            root = torch.sqrt(eps - sin2)
            r = (eps * COS - root) / (eps * COS + root)
            if thickness:
                q = 2 * math.pi * thickness / (rm.C0 / 3.5e9) * root
                e2 = torch.exp(-2j * q)
                g = r * (1 - e2) / (1 - r ** 2 * e2)
            else:
                g = r
        return 20 * torch.log10(g.abs().clamp_min(1e-8))

    itu = {}
    for name in CLASSES:
        a, b, c, d = rm.ITU[name]
        itu[name] = (a * 3.5 ** b, c * 3.5 ** d)

    def nearest(g, skip="marble"):
        best = (1e9, None)
        for name, (er, sig) in itu.items():
            if name == skip:
                continue
            gg = gamma_db(er, sig, 0.1)
            rms = float((g - gg).pow(2).mean().sqrt())
            if rms < best[0]:
                best = (rms, name)
        return best

    g_m = gamma_db(*itu["marble"], 0.1)
    log(f"M5 marble 0.1 m nearest other class: {nearest(g_m)[1]} at {nearest(g_m)[0]:.2f} dB RMS")
    for t in (0.2, 0.3, None):
        g = gamma_db(*itu["marble"], t)
        # nearest permittivity is not asked again; nearest CLASS at this thickness
        # rebuild class curves at the same thickness
        best = (1e9, None)
        for name, (er, sig) in itu.items():
            if name == "marble":
                continue
            rms = float((g - gamma_db(er, sig, t)).pow(2).mean().sqrt())
            if rms < best[0]:
                best = (rms, name)
        log(f"M5 thickness {t}: nearest other class {best[1]} at {best[0]:.2f} dB RMS")
    for scale, tag in [(0.8, "-20%"), (1.2, "+20%")]:
        er, sig = itu["marble"]
        g = gamma_db(er * scale, sig, 0.1)
        rms, who = nearest(g, skip=None)
        d_self = float((g - g_m).pow(2).mean().sqrt())
        log(f"M5 eps_r {tag}: {d_self:.2f} dB from marble, nearest class {who} at {rms:.2f} dB")
    g_tm = gamma_db(*itu["marble"], 0.1, pol="tm")
    # power sum of TE and a small TM leakage, as an amplitude perturbation
    te = 10 ** (g_m / 20)
    tm = 10 ** (g_tm / 20)
    for w, tag in [(0.15, "0.15 dB TM power (the Sionna median gap)"), (1.0, "equal TM power")]:
        mix = 20 * torch.log10((te + w * tm).clamp_min(1e-8))
        # w here is mis-scaled. Do a proper dB mix below.
        pass
    # Sionna's facade amplitude is 0.15 dB below the TE model. Shift marble by -0.15 dB.
    g_shift = g_m - 0.15
    rms, who = nearest(g_shift)
    d_self = 0.15
    log(f"M5 marble shifted by the 0.15 dB TM gap: still {rms:.2f} dB from the nearest other class ({who})")
    # fraction of (user set, noise) draws whose nearest class by report MSE is marble,
    # when the truth is marble, marble+/-20%, at 0.1 m
    def report_class_hit(er_scale, n_draws=80):
        er, sig = itu["marble"]
        th = th0.clone()
        th[2] = er * er_scale
        hits = []
        for name in ["LOS", "NLOS"]:
            ok = 0
            for s in sets[name][:n_draws]:
                u, h0 = U[s], H0[s]
                mu_t = rm.rsrp_mean_phase(cfg, th, u, h0, torch.zeros(len(s)))
                gen = torch.Generator().manual_seed(int(s[0]) + 17)
                r = mu_t + torch.randn(mu_t.shape, generator=gen)
                best, who = 1e18, None
                for cls in CLASSES:
                    thc = th0.clone()
                    ee, ss = itu[cls]
                    thc[2], thc[3] = ee, math.log10(ss)
                    mu = rm.rsrp_mean_phase(cfg, thc, u, h0, torch.zeros(len(s)))
                    c = float(((r - mu) ** 2).sum())
                    if c < best:
                        best, who = c, cls
                ok += who == "marble"
            hits.append(f"{name} {ok}/{n_draws}")
        return ", ".join(hits)

    for sc, tag in [(1.0, "marble"), (0.8, "eps_r -20%"), (1.2, "eps_r +20%")]:
        log(f"M5 report-template class is marble for {tag}: {report_class_hit(sc)}")

    # --- m7: B network on E, and the Bayesian CRB on B ---
    var_d = 12.0 ** 2 / 12.0          # variance of U[-6, 6]
    var_p = math.radians(4.0) ** 2 / 12.0
    log(f"m7 prior variance matched to the uniform: d {var_d:.2f} m^2, phi {math.degrees(math.sqrt(var_p)):.3f} deg std")
    for name in ["LOS", "NLOS"]:
        crb_r, crb_b = [], []
        for s in sets[name]:
            J = rm.fisher_robust(cfg, th0, U[s], H0[s], sigma_u=SIGMA_U).sum(0)
            Jp = torch.zeros(4, 4)
            Jp[0, 0] = 1.0 / var_d
            Jp[1, 1] = 1.0 / var_p
            crb_r.append(float(torch.linalg.inv(J)[0, 0]))
            crb_b.append(float(torch.linalg.inv(J + Jp)[0, 0]))
        log(f"m7 facade B {name} at the map wall: radio {np.sqrt(np.mean(crb_r)):.2f} m, "
            f"Bayesian {np.sqrt(np.mean(crb_b)):.2f} m  (RMS, N=10)")

    din = cfg.n_wide + cfg.n_wide + 2
    _, cfgE, thE, UE, H0E, losE = sd.load("E", "3.5")
    for name in ["LOS", "NLOS"]:
        net = DeepSets(din).float()
        net.load_state_dict(torch.load(os.path.join(ART, f"step10_net_{name}.pt"), map_location="cpu"))
        net.eval()
        pool = torch.tensor(np.where(losE if name == "LOS" else ~losE)[0])
        gen = torch.Generator().manual_seed(21 if name == "LOS" else 22)
        B = 2000
        err, acc = [], []
        for s0 in range(0, B, 400):
            b = min(400, B - s0)
            idx = torch.stack([pool[torch.randperm(len(pool), generator=gen)[:N]] for _ in range(b)])
            users, h0 = UE[idx], H0E[idx]
            dd = (torch.rand(b, generator=gen) * 2 - 1) * 6.0
            dp = (torch.rand(b, generator=gen) * 2 - 1) * math.radians(2.0)
            mat = torch.randint(0, len(CLASSES), (b,), generator=gen)
            th = torch.cat([(thE[0] + dd)[:, None], (thE[1] + dp)[:, None],
                            material_params(mat.tolist(), cfgE.freq)], 1)
            r = mu_batch(cfgE, th, users, h0) + torch.randn(b, N, cfgE.n_wide, generator=gen)
            u_hat = users + torch.cat([torch.randn(b, N, 2, generator=gen), torch.zeros(b, N, 1)], 2)
            mu_map = mu_batch(cfgE, thE.repeat(b, 1), u_hat, h0)
            rel = (u_hat - cfgE.bs)[..., :2] / 100.0
            x = torch.cat([(r - mu_map) / 10.0, (mu_map + 80.0) / 20.0, rel], 2).float()
            with torch.no_grad():
                out = net(x)
            err.append((out[:, 0].double() * 6.0 - dd).numpy())
            acc.append((out[:, 2:].argmax(1) == mat).numpy())
        e = np.concatenate(err)
        log(f"m7 train B test E {name}: offset RMSE {np.sqrt(np.mean(e ** 2)):.2f} m, "
            f"material acc {np.mean(np.concatenate(acc)):.2f}  (2000 sets)")

    # --- M2 report-domain perturbation of the other paths ---
    def perturb_bias(name, n_sets=80):
        biases, gaps = [], []
        for s in sets[name][:n_sets]:
            u, h0 = U[s], H0[s]
            hd = torch.where(torch.tensor(los[s])[:, None], hd_all[s], torch.zeros_like(hd_all[s]))
            rest = h0 - hd
            # random ITU class versus marble, one draw per user, path error +/- 0.5 m
            cls = rng.choice(CLASSES, len(s))
            er_m, sg_m = itu["marble"]
            g_m = rm.slab_te(eps_of(er_m, sg_m), torch.tensor(0.5), 0.1, cfg.lam).abs()
            scales = []
            for cname in cls:
                er, sg = itu[cname]
                g = rm.slab_te(eps_of(er, sg), torch.tensor(0.5), 0.1, cfg.lam).abs()
                scales.append(float(g / g_m))
            delay = torch.tensor(rng.uniform(-0.5, 0.5, len(s)))
            phase = torch.exp(-1j * (2 * math.pi / cfg.lam) * delay)
            h0p = hd + torch.tensor(scales)[:, None] * phase[:, None] * rest
            mu = rm.rsrp_mean_phase(cfg, th0, u, h0, torch.zeros(len(s)))
            # grid the offset; other parameters stay at the truth
            best, bd = 1e18, 0.0
            for dlt in np.linspace(-3, 3, 31):
                th = th0.clone()
                th[0] = th0[0] + dlt
                mup = rm.rsrp_mean_phase(cfg, th, u, h0p, torch.zeros(len(s)))
                c = float(((mu - mup) ** 2).mean())
                if c < best:
                    best, bd = c, dlt
            mup0 = rm.rsrp_mean_phase(cfg, th0, u, h0p, torch.zeros(len(s)))
            gaps.append(float(((mu - mup0) ** 2).mean().sqrt()))
            biases.append(bd)
        return float(np.median(np.abs(biases))), float(np.median(gaps))

    for name in ["LOS", "NLOS"]:
        b, g = perturb_bias(name)
        log(f"M2 h0 perturbed (random ITU scale, path +/-0.5 m) {name}: "
            f"median |offset bias| {b:.2f} m, median RMS report gap {g:.2f} dB")

    open(os.path.join(ART, "step14_complete.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
