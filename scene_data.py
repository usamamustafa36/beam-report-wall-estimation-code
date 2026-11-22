"""Loads the traced facades (step3_trace.py, step3_e2e_trace.py) for the analysis scripts."""
import math, os
import numpy as np
import torch
import rsrp_model as rm

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SCENE = {"A": "munich", "B": "munich", "E": "etoile"}


def facade_file(F, f):
    return os.path.join(ART, f"{SCENE[F]}_facade{F}_{f}GHz.npz")


def load(F="B", f="3.5", K=16):
    """(z, cfg, theta0, users, h0, los) for the users the facade reaches.

    theta0 = (d, phi, eps_r, log10 sigma) of the true wall, material as recorded in the scene."""
    z = np.load(facade_file(F, f))
    v = z["fval"]
    cfg = rm.Config(freq=float(z["freq"]), bs=tuple(z["bs"]), axis=tuple(z["axis"]), n_wide=K)
    n = z["n"]
    a, b, c, d = rm.ITU[str(z["material"])]
    fg = cfg.freq / 1e9
    th0 = torch.tensor([float(z["d"]), math.atan2(n[1], n[0]), a * fg ** b, math.log10(c * fg ** d)])
    return z, cfg, th0, torch.tensor(z["users"][v]), torch.tensor(z["h0"][v]), z["los"][v]


def e2e(F="B", f="3.5"):
    """Same-run Sionna channels for every 3rd reachable user where Sionna found the facade path.

    Returns pos (indices into the reachable users), h0_same, h_model, h_all, h_rest (complex128):
    h_rest = h_all minus Sionna's single-bounce facade path = every path a complete twin predicts
    from its map (incl. two-bounce paths via the facade, at the map position)."""
    z = np.load(facade_file(F, f))
    e = np.load(os.path.join(ART, f"e2e_{SCENE[F]}_facade{F}_{f}GHz.npz"))
    pos = np.searchsorted(np.where(z["fval"])[0], e["idx"])
    k = e["found"]
    t = lambda x: torch.tensor(x[k]).to(torch.complex128)
    h0, hm, ha = t(e["h0_same"]), t(e["h_model"]), t(e["h_all"])
    return pos[k], h0, hm, ha, ha - (hm - h0)
