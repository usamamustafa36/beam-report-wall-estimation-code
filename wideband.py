"""Linear average of per-subcarrier RSRP for the two-path model.

The stored non-wall channel has no per-path delay, so the average uses the geometric
direct path (LOS users only) and the specular wall path. Subcarrier spacing is 0.24 MHz,
the same grid as step12_wideband.py. The array response is the half-wavelength response
at the carrier: steering does not depend on wavelength, which is what step 12 computes.
"""
import math
import torch
import rsrp_model as rm

SPACING_MHZ = 0.24


def frequencies(cfg, band_mhz):
    k = max(1, int(round(band_mhz / SPACING_MHZ)) + 1) if band_mhz > 0 else 1
    if k == 1:
        return torch.tensor([cfg.freq])
    return cfg.freq + torch.linspace(-0.5, 0.5, k) * (band_mhz * 1e6)


def _slab(eps, cos_i, thickness, lam):
    """eps (F,), cos_i (N,), lam (F,) -> Gamma (F, N)."""
    sin2 = 1 - cos_i ** 2
    root = torch.sqrt(eps[:, None] - sin2)
    r = (cos_i - root) / (cos_i + root)
    q = 2 * math.pi * thickness / lam[:, None] * torch.sqrt(eps[:, None] - sin2)
    e2 = torch.exp(-2j * q)
    return r * (1 - e2) / (1 - r ** 2 * e2)


def rsrp(cfg, theta, users, wall_phase, band_mhz, use_direct, mode="material", material=None):
    """Noise-free wideband RSRP (N, K) in dBm. use_direct is per user, shape (N,)."""
    d, phi = theta[0], theta[1]
    L_d, L_r, k_d, k_r, cos_i, valid = rm.geometry(cfg, d, phi, users)
    a_d = rm.steering(cfg, k_d)
    a_r = rm.steering(cfg, k_r)
    freqs = frequencies(cfg, band_mhz)
    lam = rm.C0 / freqs
    kw = 2 * math.pi / lam
    g_d = lam[:, None] / (4 * math.pi * L_d) * torch.exp(-1j * kw[:, None] * L_d)
    g_r = lam[:, None] / (4 * math.pi * L_r) * torch.exp(-1j * kw[:, None] * L_r)
    if wall_phase is not None:
        g_r = g_r * torch.exp(1j * wall_phase)[None, :]
    if mode == "gamma":
        gam = (theta[2] * torch.exp(1j * theta[3])).expand(len(freqs), len(users))
    elif mode == "material":
        eps = theta[2] - 1j * (10 ** theta[3]) / (2 * math.pi * freqs * rm.EPS0)
        gam = _slab(eps, cos_i, cfg.thickness, lam) if cfg.thickness is not None else (
            (cos_i - torch.sqrt(eps[:, None] - (1 - cos_i ** 2)))
            / (cos_i + torch.sqrt(eps[:, None] - (1 - cos_i ** 2))))
    else:
        er = torch.tensor([rm.itu_eps(material, float(f)) for f in freqs])
        gam = _slab(er, cos_i, cfg.thickness, lam)
    direct = use_direct.to(g_d.real.dtype)[None, :, None] * g_d[:, :, None] * a_d[None]
    wall = (valid.to(g_r.real.dtype) * gam * g_r)[:, :, None] * a_r[None]
    p = torch.abs((direct + wall) @ cfg.C.conj().T) ** 2
    return 10 * torch.log10(cfg.ptx * p.mean(0) + cfg.noise)


def install(band_mhz, use_direct=None):
    """Point the narrowband calls used by fit() at this wideband mean.

    use_direct is a 0-d bool, or None. None reads cfg.use_direct (a bool) at call time,
    which is how a worker distinguishes LOS from NLOS. A length-N mask is rejected:
    fisher_robust maps the mean over one user at a time, so a long mask would attach
    the first user's flag to every user.
    """
    def wrapped(cfg, theta, users, h0, wall_phase, mode="material", material=None):
        mask = use_direct if use_direct is not None else getattr(cfg, "use_direct", True)
        if not torch.is_tensor(mask):
            mask = torch.tensor(bool(mask))
        if mask.ndim == 0:
            mask = mask.expand(users.shape[0])
        elif mask.shape[0] != users.shape[0]:
            raise RuntimeError("use_direct must be a scalar or one flag per user in this call")
        return rsrp(cfg, theta, users, wall_phase, band_mhz, mask, mode, material)
    rm.rsrp_mean_phase = wrapped
    return wrapped
