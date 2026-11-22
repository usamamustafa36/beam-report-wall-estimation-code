"""Closed-form wide-beam RSRP model for one specular wall (image-source method).

Geometry is 3D, the wall is vertical (an infinite plane n(phi)^T x = d with
n = (cos phi, sin phi, 0)), and the BS carries a horizontal half-wavelength ULA.
The channel at user u is a direct path plus the single specular reflection:
    h(theta) = 1[LOS] g_d a(k_d) + Gamma g_r a(k_r),   g = lam/(4 pi L) exp(-j 2 pi L / lam)
The UE reports RSRP on L wide beams c_k (16-element sub-array, uniform spatial grid,
identical to revision_common.wide_codebook in the TCCN code):
    mu_k(theta) = 10 log10( P_tx |c_k^H h|^2 + sigma^2 )  [dBm],   r_k = mu_k + e_k.
The reflection coefficient is either a free complex constant ("gamma" mode, used for
the lemmas) or the TE coefficient of an ITU-R P.2040 material ("material" mode, vertical
polarization on a vertical wall = field perpendicular to the plane of incidence). The wall is a
slab of thickness cfg.thickness (ITU-R P.2040 eq. 43, as in Sionna RT, default 0.1 m); with
thickness=None it is a half-space.
Everything is torch float64 so the Fisher information comes from autograd.
"""
import math
import torch

torch.set_default_dtype(torch.float64)
C0 = 299792458.0
EPS0 = 8.8541878128e-12

# ITU-R P.2040 parameters (a, b, c, d): eps_r = a f^b, sigma = c f^d, f in GHz.
# Copied from sionna/rt/radio_materials/itu.py (sionna-rt 1.2.2); all valid at 3.5 and 28 GHz.
ITU = {"concrete": (5.24, 0.0, 0.0462, 0.7822), "brick": (3.91, 0.0, 0.0238, 0.16),
       "plasterboard": (2.73, 0.0, 0.0085, 0.9395), "wood": (1.99, 0.0, 0.0047, 1.0718),
       "glass": (6.31, 0.0, 0.0036, 1.3394), "marble": (7.074, 0.0, 0.0055, 0.9262),
       "metal": (1.0, 0.0, 1e7, 0.0)}


class Config:
    def __init__(self, freq=3.5e9, n_ant=64, n_wide=16, n_narrow=64, bs=(0.0, 0.0, 25.0),
                 axis=(0.0, 1.0, 0.0), ptx_dbm=15.0, noise_dbm=-122.0, sigma_e_db=1.0, thickness=0.1):
        self.freq, self.lam = freq, C0 / freq
        self.n_ant, self.n_wide = n_ant, n_wide
        self.bs = torch.tensor(bs)
        self.axis = torch.tensor(axis) / torch.linalg.norm(torch.tensor(axis))   # ULA axis (horizontal)
        self.ptx = 10 ** (ptx_dbm / 10)            # mW per resource element
        self.noise = 10 ** (noise_dbm / 10)        # mW per resource element (30 kHz, NF 7 dB)
        self.sigma_e = sigma_e_db                  # RSRP measurement error std [dB]
        self.thickness = thickness                 # wall slab thickness [m] (Sionna default 0.1)
        self.C = wide_codebook(n_wide, n_ant, n_narrow)


def wide_codebook(L=16, n=64, n_narrow=64):
    """L wide beams from an (n*L/n_narrow)-element sub-array, spatial grid matched to L."""
    ang = torch.linspace(-1, 1, L + 1)[:-1]
    n_act = n * L // n_narrow
    C = torch.zeros(L, n, dtype=torch.complex128)
    C[:, :n_act] = torch.exp(1j * math.pi * torch.outer(ang, torch.arange(n_act, dtype=torch.float64))) / math.sqrt(n_act)
    return C


def steering(cfg, k):
    """ULA response to unit directions k (..., 3): a_m = exp(j pi (m - (M-1)/2) <k, axis>).

    The array is centred on cfg.bs, as in Sionna RT."""
    s = k @ cfg.axis
    m = torch.arange(cfg.n_ant, dtype=torch.float64) - (cfg.n_ant - 1) / 2
    return torch.exp(1j * math.pi * s[..., None] * m)


def fresnel_te(eps, cos_i):
    """TE (perpendicular) reflection coefficient for complex relative permittivity eps."""
    sin2 = 1 - cos_i ** 2
    root = torch.sqrt(eps - sin2)
    return (cos_i - root) / (cos_i + root)


def slab_te(eps, cos_i, thickness, lam):
    """TE reflection coefficient of a slab (ITU-R P.2040 eq. 43), identical to Sionna RT."""
    r = fresnel_te(eps, cos_i)
    q = 2 * math.pi * thickness / lam * torch.sqrt(eps - (1 - cos_i ** 2))
    e2 = torch.exp(-2j * q)
    return r * (1 - e2) / (1 - r ** 2 * e2)


def reflect_te(cfg, eps, cos_i):
    return fresnel_te(eps, cos_i) if cfg.thickness is None else slab_te(eps, cos_i, cfg.thickness, cfg.lam)


def itu_eps(name, freq):
    a, b, c, d = ITU[name]
    f = freq / 1e9
    return complex(a * f ** b, -c * f ** d / (2 * math.pi * freq * EPS0))


def wall_normal(phi):
    z = torch.zeros_like(phi)
    return torch.stack([torch.cos(phi), torch.sin(phi), z])


def geometry(cfg, d, phi, users):
    """Image-source quantities for users (N, 3) and the wall (d, phi).

    Returns direct length, reflected length, unit departure directions of both paths,
    cosine of the incidence angle, and a validity mask (BS and user on the same side,
    reflection point on the wall segment between their images).
    """
    n = wall_normal(phi)
    b = cfg.bs
    sb = b @ n - d
    su = users @ n - d
    b_img = b - 2 * sb * n
    v = users - b_img                                # image BS -> user
    L_r = torch.linalg.norm(v, dim=-1)
    t = (d - b_img @ n) / (v @ n)                    # reflection point on the image ray
    x_r = b_img + t[:, None] * v
    k_r = (x_r - b) / torch.linalg.norm(x_r - b, dim=-1, keepdim=True)
    dv = users - b
    L_d = torch.linalg.norm(dv, dim=-1)
    k_d = dv / L_d[:, None]
    cos_i = torch.abs(v @ n) / L_r
    valid = (sb * su > 0) & (t > 0) & (t < 1)
    return L_d, L_r, k_d, k_r, cos_i, valid


def channel(cfg, theta, users, los, mode="gamma", material=None, h0=None, wall_phase=None):
    """Narrowband channel (N, n_ant).

    h0 (N, n_ant), if given, replaces the direct path by every path the twin already knows
    (LOS, ground, other walls); los is then ignored. wall_phase (N,), if given, adds an unknown
    per-user phase to the wall path (nuisance for sub-wavelength user-position uncertainty).

    mode "gamma":    theta = (d, phi, |Gamma|, arg Gamma)
    mode "material": theta = (d, phi, eps_r, log10 sigma)   (Fresnel TE)
    mode "fixed":    theta = (d, phi), material given by name (ITU)
    """
    d, phi = theta[0], theta[1]
    L_d, L_r, k_d, k_r, cos_i, valid = geometry(cfg, d, phi, users)
    kw = 2 * math.pi / cfg.lam
    g_d = cfg.lam / (4 * math.pi * L_d) * torch.exp(-1j * kw * L_d)
    g_r = cfg.lam / (4 * math.pi * L_r) * torch.exp(-1j * kw * L_r)
    if mode == "gamma":
        gam = theta[2] * torch.exp(1j * theta[3])
    elif mode == "material":
        eps = theta[2] - 1j * (10 ** theta[3]) / (2 * math.pi * cfg.freq * EPS0)
        gam = reflect_te(cfg, eps, cos_i)
    else:
        gam = reflect_te(cfg, torch.tensor(itu_eps(material, cfg.freq)), cos_i)
    if wall_phase is not None:
        g_r = g_r * torch.exp(1j * wall_phase)
    base = h0 if h0 is not None else los.to(torch.float64)[:, None] * g_d[:, None] * steering(cfg, k_d)
    h = base + (valid.to(torch.float64) * gam * g_r)[:, None] * steering(cfg, k_r)
    return h


def rsrp_mean(cfg, theta, users, los, mode="gamma", material=None, h0=None):
    """Noise-free RSRP mean mu (N, L) in dBm."""
    h = channel(cfg, theta, users, los, mode, material, h0)
    p = torch.abs(h @ cfg.C.conj().T) ** 2           # |c_k^H h|^2, same convention as the TCCN code
    return 10 * torch.log10(cfg.ptx * p + cfg.noise)


def measure(cfg, mu, gen, quantise=False):
    """Noisy RSRP reports; optional 3GPP-style 1 dB quantisation and [-156, -31] dBm clipping."""
    r = mu + cfg.sigma_e * torch.randn(mu.shape, generator=gen)
    if quantise:
        r = torch.clamp(torch.round(r), -156.0, -31.0)
    return r


def jacobian(cfg, theta, users, los, mode="gamma", material=None, h0=None):
    """d mu / d theta, shape (N, L, P). Forward mode: one pass per parameter."""
    f = lambda th: rsrp_mean(cfg, th, users, los, mode, material, h0)
    return torch.func.jacfwd(f)(theta)


def fisher(cfg, theta, users, los, mode="gamma", material=None, per_user=False, h0=None):
    """Fisher information of the Gaussian-in-dB model, J = sigma_e^-2 sum_n sum_k grad grad^T."""
    Jac = jacobian(cfg, theta, users, los, mode, material, h0)
    Jn = torch.einsum("nkp,nkq->npq", Jac, Jac) / cfg.sigma_e ** 2
    return Jn if per_user else Jn.sum(0)


def rank(Jac, tol=1e-9):
    """Numerical rank of the stacked Jacobian (rows = reports), columns normalised to unit norm.

    A column below tol relative to the largest is rounding noise and is zeroed, not normalised
    (normalising it would turn noise into a spurious extra rank)."""
    A = Jac.reshape(-1, Jac.shape[-1])
    nrm = torch.linalg.norm(A, dim=0)
    keep = nrm > tol * nrm.max()
    A = torch.where(keep, A / torch.where(keep, nrm, torch.ones_like(nrm)), torch.zeros_like(A))
    s = torch.linalg.svdvals(A)
    return int((s > tol * s[0]).sum()), s


def null_space(Jac, tol=1e-9):
    """Right null vectors of the stacked Jacobian in the original parameter units."""
    A = Jac.reshape(-1, Jac.shape[-1])
    _, s, Vh = torch.linalg.svd(A)
    return Vh[(s <= tol * s[0]).nonzero().flatten()] if len(s) == A.shape[1] else Vh


def fisher_robust(cfg, theta, users, h0, sigma_u, mode="material", material=None, rsrp_fn=None):
    """Per-user Fisher information on theta with nuisances eliminated (Schur complement).

    Nuisances per user: horizontal position offset (prior std sigma_u [m]; None = no offset) and
    a phase on the wall path (no prior). rsrp_fn replaces rsrp_mean_phase (same signature), e.g.
    by a wideband RSRP. Returns (N, P, P).
    """
    f = rsrp_fn or rsrp_mean_phase

    def mu1(th, u, hr, dxy, psi):                     # hr: h0 as a real view (jacfwd rejects complex inputs)
        uu = (u + torch.cat([dxy, torch.zeros(1)]))[None]
        return f(cfg, th, uu, torch.view_as_complex(hr)[None], psi[None], mode, material)[0]

    jac = torch.func.vmap(torch.func.jacfwd(mu1, argnums=(0, 3, 4)), in_dims=(None, 0, 0, 0, 0))
    N = users.shape[0]
    Jt, Jd, Jp = jac(theta, users, torch.view_as_real(h0), torch.zeros(N, 2), torch.zeros(N))
    # Whitened Jacobians with the position prior as two extra rows. The Schur complement equals
    # At' (I - Pn) At, Pn the projector onto range(An); computing it by SVD of An (not by
    # inverting the squared block) keeps the precision when parameter scales differ widely.
    At = Jt / cfg.sigma_e
    if sigma_u is None:
        An = Jp[..., None] / cfg.sigma_e
    else:
        N = users.shape[0]
        R = torch.zeros(N, 2, 3)
        R[:, 0, 0] = R[:, 1, 1] = 1 / sigma_u
        An = torch.cat([torch.cat([Jd, Jp[..., None]], -1) / cfg.sigma_e, R], 1)
        At = torch.cat([At, torch.zeros(N, 2, At.shape[-1])], 1)
    Un, s, _ = torch.linalg.svd(An, full_matrices=False)
    # drop only nuisance directions that are exactly uninformative (rounding noise, e.g. the wall
    # phase of a user with no other known path); weakly informative ones must still be projected
    # out, else the bound depends on where a cut-off falls
    ref = torch.maximum(s[:, 0], torch.linalg.matrix_norm(At, ord=2))
    Un = Un * (s > 1e-12 * ref[:, None])[:, None, :]
    Res = At - Un @ (Un.transpose(1, 2) @ At)
    return Res.transpose(1, 2) @ Res


def rsrp_mean_phase(cfg, theta, users, h0, wall_phase, mode="material", material=None):
    h = channel(cfg, theta, users, None, mode, material, h0, wall_phase)
    p = torch.abs(h @ cfg.C.conj().T) ** 2
    return 10 * torch.log10(cfg.ptx * p + cfg.noise)
