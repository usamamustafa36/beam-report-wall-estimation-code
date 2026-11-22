"""Step 7: why the learned wide-beam predictor of the TCCN paper breaks, through Lemma 1.

In the TCCN model (paper2026/ris.py) the malicious RIS adds  scale * (k_u^H v) * a_bsR : a fixed
BS-side direction times ONE complex number zeta = scale * k_u^H v per user. By Lemma 1 the 16
wide-beam reports depend on the 64 RIS phases only through zeta, and since sums of M >= 2 unit
phasors fill the disk |zeta| <= scale * M, the M-dimensional phase attack is equivalent to choosing
zeta in that disk. This script, with the TCCN code unchanged and its headline setting
(M = 64, kappa = 1, i.e. equal power; seeds as stage_ci.py):
  1. re-runs the TCCN attack and checks top-1;
  2. checks that every attacked channel equals h_d + zeta a_bsR (Lemma 1 holds exactly);
  3. replaces the 64-phase attack by a grid search over zeta (8 radii x 32 phases) with the same
     soft objective, and by a phase-only search on the rim |zeta| = scale * M;
  4. relates, per user, the phase information of Lemma 2 (sum_k (d mu_k / d arg zeta)^2 at the
     rim, averaged over arg zeta) to whether the attack fools the predictor (AUC).
Output: artifacts/step7_predictor.json
"""
import json, math, os, sys
import numpy as np
import torch

TCCN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "usama", "paper2026")
sys.path.insert(0, TCCN)
import beamdata as bd              # noqa: E402
import ris                         # noqa: E402
import revision_common as rc       # noqa: E402
from model import DEVICE           # noqa: E402

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SEEDS = rc.TRAIN_SEEDS[:3] if len(sys.argv) < 2 else rc.TRAIN_SEEDS[:1]
RADII = np.linspace(0.125, 1.0, 8)
PHASES = np.arange(32) * 2 * np.pi / 32


def zeta_channels(g, zeta):
    """h_d + zeta a_bsR for zeta of shape (U,) -> (U, N)."""
    return g["H_d"] + zeta[:, None] * g["a_bsR"][None, :]


def soft(v, H, Wc):
    with torch.no_grad():
        return rc.soft_objective(v, H, Wc)[:, 0]


def zeta_attack(v, g, radii):
    """Per user, the zeta on the grid (radii x PHASES, in units of scale*M) minimising the soft objective."""
    U = g["H_d"].shape[0]
    R = (g["scale"] * g["M"]).to(torch.complex64)
    best_obj = torch.full((U,), float("inf"), device=DEVICE)
    best = torch.zeros(U, dtype=torch.complex64, device=DEVICE)
    for rad in radii:
        for ph in PHASES:
            z = R * complex(rad * math.cos(ph), rad * math.sin(ph))
            o = soft(v, zeta_channels(g, z), g["W"].conj())
            better = o < best_obj
            best_obj = torch.where(better, o, best_obj)
            best = torch.where(better, z, best)
    return zeta_channels(g, best).cpu().numpy()


def phase_information(g, C):
    """Lemma 2: mean over arg zeta of sum_k (d mu_k / d arg zeta)^2 at the rim, mu in dB (noise-free)."""
    R = (g["scale"] * g["M"]).double()
    Hd = g["H_d"].to(torch.complex128)
    a = g["a_bsR"].to(torch.complex128)
    Ct = torch.as_tensor(C, dtype=torch.complex128, device=DEVICE)
    alpha = Hd @ Ct.conj().T                                   # (U, K): c_k^H h_d
    beta = (Ct.conj() @ a)[None, :]                            # (1, K): c_k^H a_bsR
    info = torch.zeros(Hd.shape[0], dtype=torch.float64, device=DEVICE)
    for ph in PHASES:
        z = (R * complex(math.cos(ph), math.sin(ph)))[:, None]
        tot = torch.abs(alpha + z * beta) ** 2 + 1e-30
        d = -(20 / math.log(10)) * torch.imag(alpha.conj() * z * beta) / tot
        info += (d ** 2).sum(1) / len(PHASES)
    return info.cpu().numpy()


def auc(score, label):
    """Probability that a random positive outranks a random negative."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]
    out = {"seeds": SEEDS, "trials": []}
    for s in SEEDS:
        v = rc.train_victim("rsrp", d, seed=s)
        v.model.eval()
        g = ris.build_geometry(H_te, 64, 1.0, seed=s)
        Ha, theta = rc.ris_attack(v, g, seed=s, return_theta=True)
        # Lemma 1 in the TCCN model: the attacked channel is h_d + zeta a_bsR
        zeta = g["scale"] * (g["k_u"].conj() * torch.exp(1j * theta)).sum(1)
        rec = zeta_channels(g, zeta).cpu().numpy()
        lemma_err = float(np.abs(rec - Ha).max() / np.abs(Ha).max())
        rmax = (g["scale"] * g["M"]).cpu().numpy()
        frac_r = np.abs(zeta.cpu().numpy()) / rmax
        clean = rc.evaluate_noise_avg(v, H_te, W)
        tccn = rc.evaluate_noise_avg(v, Ha, W)
        zgrid = rc.evaluate_noise_avg(v, zeta_attack(v, g, RADII), W)
        zrim = rc.evaluate_noise_avg(v, zeta_attack(v, g, [1.0]), W)
        # per user: fooled by the TCCN attack (noise seed 42) and phase information
        Hn = bd.add_cn_noise(Ha, bd.PILOT_SNR_DB, np.random.default_rng(42))
        Hc = bd.add_cn_noise(H_te, bd.PILOT_SNR_DB, np.random.default_rng(42))
        y_a = np.abs(Ha.astype(np.complex128) @ W.conj().T).argmax(1)
        y_c = np.abs(H_te.astype(np.complex128) @ W.conj().T).argmax(1)
        with torch.no_grad():
            p_a = v.model(torch.as_tensor(v.feat_np(Hn), device=DEVICE)).argmax(1).cpu().numpy()
            p_c = v.model(torch.as_tensor(v.feat_np(Hc), device=DEVICE)).argmax(1).cpu().numpy()
        right_clean = p_c == y_c
        fooled = right_clean & (p_a != y_a)
        info = phase_information(g, v.C)
        t = dict(seed=s, lemma1_rel_err=lemma_err, zeta_radius_median=float(np.median(frac_r)),
                 zeta_radius_q10_q90=[float(x) for x in np.percentile(frac_r, [10, 90])],
                 clean_top1=clean["top1"], tccn_attack_top1=tccn["top1"], zeta_grid_top1=zgrid["top1"],
                 zeta_rim_top1=zrim["top1"], clean_se3=clean["se_ref3"], tccn_se3=tccn["se_ref3"],
                 zeta_grid_se3=zgrid["se_ref3"],
                 auc_phaseinfo_fooled=auc(info[right_clean], fooled[right_clean]),
                 fooled_rate_among_correct=float(fooled[right_clean].mean()),
                 info_median_fooled=float(np.median(info[fooled])), info_median_kept=float(np.median(info[right_clean & ~fooled])))
        out["trials"].append(t)
        print(json.dumps(t), flush=True)
    for k in ["clean_top1", "tccn_attack_top1", "zeta_grid_top1", "zeta_rim_top1", "auc_phaseinfo_fooled", "lemma1_rel_err"]:
        xs = [t[k] for t in out["trials"]]
        out[k + "_mean"] = float(np.mean(xs))
    json.dump(out, open(os.path.join(ART, "step7_predictor.json"), "w"), indent=1)
    print({k: v for k, v in out.items() if k.endswith("_mean")})


if __name__ == "__main__":
    main()
