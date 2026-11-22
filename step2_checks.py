"""Step 2: numerical checks of the statements proved in paper/identifiability.tex.

Same scene as step1_checks.py. Each check names the lemma, proposition or remark it supports.
Writes artifacts/step2_checks.txt.
"""
import math, os, sys
import numpy as np
import torch
import rsrp_model as rm

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
lines = []


def log(s):
    print(s, flush=True)
    lines.append(s)


def check(name, ok, detail=""):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


log("\n================ STEP 2 ================")
cfg = rm.Config()
D, PHI = -60.0, math.pi
thg = torch.tensor([D, PHI, 0.5, 0.7])
T, F = torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
u0 = torch.tensor([[30.0, 12.0, 1.5]])
kap = 2 * math.pi / cfg.lam
ok_all = True

# Remark (per-beam phase information): d mu_k / d arg(Gamma) = -(20/ln10) P Im{a_k^* c b_k} / (P|a_k + c b_k|^2 + s2)
L_d, L_r, k_d, k_r, cos_i, _ = rm.geometry(cfg, thg[0], thg[1], u0)
g_d = cfg.lam / (4 * math.pi * L_d) * torch.exp(-1j * kap * L_d)
c = thg[2] * torch.exp(1j * thg[3]) * cfg.lam / (4 * math.pi * L_r) * torch.exp(-1j * kap * L_r)
alpha = (cfg.C.conj() @ (g_d[:, None] * rm.steering(cfg, k_d)).T)[:, 0]
beta = (cfg.C.conj() @ rm.steering(cfg, k_r).T)[:, 0]
tot = cfg.ptx * torch.abs(alpha + c * beta) ** 2 + cfg.noise
formula = -(20 / math.log(10)) * cfg.ptx * torch.imag(alpha.conj() * c * beta) / tot
auto = rm.jacobian(cfg, thg, u0, T)[0, :, 3]
err = (formula - auto).abs().max().item() / auto.abs().max().item()
ok_all &= check("Remark: closed-form per-beam phase derivative = autograd", err < 1e-10, f"rel err {err:.1e}")
share = (formula ** 2 / (formula ** 2).sum()).numpy()
log(f"  share of arg(Gamma) information per wide beam: {np.array2string(share, precision=3)}")

# Lemma 2: explicit null vectors.  eta = (s_r, log|c|, arg c);  t = (ds/dphi, -ds/dd) keeps s_r fixed
def eta(th):
    _, Lr, _, kr, _, _ = rm.geometry(cfg, th[0], th[1], u0)
    return torch.stack([(kr @ cfg.axis)[0], torch.log(th[2] / Lr[0]), th[3] - kap * Lr[0]])


def Lr_of(g):
    return rm.geometry(cfg, g[0], g[1], u0)[1][0]


gs = torch.autograd.functional.jacobian(lambda g: eta(torch.cat([g, thg[2:]]))[0], thg[:2])
gL = torch.autograd.functional.jacobian(Lr_of, thg[:2])
t = torch.stack([gs[1], -gs[0]])
v_los = torch.cat([t, thg[2:3] * (gL @ t)[None] / Lr_of(thg[:2]), (kap * gL @ t)[None]])
w_nlos = torch.cat([t, thg[2:3] * (gL @ t)[None] / Lr_of(thg[:2]), torch.zeros(1)])
for name, los, vecs in [("LOS", T, [v_los]), ("NLOS", F, [w_nlos, torch.tensor([0, 0, 0, 1.0])])]:
    J = rm.jacobian(cfg, thg, u0, los).reshape(-1, 4)
    for v in vecs:
        r = (torch.linalg.norm(J @ v) / (torch.linalg.norm(J) * torch.linalg.norm(v))).item()
        ok_all &= check(f"Lemma 2 ({name}): explicit null vector annihilates J", r < 1e-10, f"|Jv|/|J||v| = {r:.1e}")

# A_n = d mu / d eta has full column rank (assumption of the proposition)
users = torch.stack([10 + 40 * torch.rand(500, generator=torch.Generator().manual_seed(1)),
                     -40 + 80 * torch.rand(500, generator=torch.Generator().manual_seed(2)),
                     torch.full((500,), 1.5)], 1)


def mu_of_eta(e, u, los):
    """RSRP as a function of eta = (s_r, |c|, arg c) for a fixed user (direct path known)."""
    L_d = torch.linalg.norm(u - cfg.bs)
    kd = (u - cfg.bs) / L_d
    h = los * cfg.lam / (4 * math.pi * L_d) * torch.exp(-1j * kap * L_d) * rm.steering(cfg, kd) \
        + e[1] * torch.exp(1j * e[2]) * torch.exp(1j * math.pi * e[0] * torch.arange(cfg.n_ant, dtype=torch.float64))
    return 10 * torch.log10(cfg.ptx * torch.abs(cfg.C.conj() @ h) ** 2 + cfg.noise)


full = {"LOS": 0, "NLOS": 0}
for n in range(500):
    _, Lr, _, kr, _, _ = rm.geometry(cfg, thg[0], thg[1], users[n:n + 1])
    e = torch.stack([(kr @ cfg.axis)[0], thg[2] * cfg.lam / (4 * math.pi * Lr[0]), thg[3] - kap * Lr[0]])
    for name, los, m in [("LOS", 1.0, 3), ("NLOS", 0.0, 2)]:
        A = torch.autograd.functional.jacobian(lambda x: mu_of_eta(x, users[n], los), e)[:, :m]
        full[name] += int(rm.rank(A[None])[0] == m)
ok_all &= check("A_n full column rank (LOS, 3 columns)", full["LOS"] == 500, f"{full['LOS']}/500 users")
ok_all &= check("A_n full column rank (NLOS, 2 columns)", full["NLOS"] == 500, f"{full['NLOS']}/500 users")

# Proposition (genericity): random user pairs
g = torch.Generator().manual_seed(7)
stats = {k: [] for k in ["LOS pair", "NLOS pair", "LOS+NLOS pair", "material NLOS pair"]}
a, b_, c_, d_ = rm.ITU["concrete"]
f = cfg.freq / 1e9
thm = torch.tensor([D, PHI, a * f ** b_, math.log10(c_ * f ** d_)])
for _ in range(300):
    U = users[torch.randperm(500, generator=g)[:2]]                    # two distinct users
    stats["LOS pair"].append(rm.rank(rm.jacobian(cfg, thg, U, T.repeat(2)))[0])
    stats["NLOS pair"].append(rm.rank(rm.jacobian(cfg, thg, U, F.repeat(2)))[0])
    stats["LOS+NLOS pair"].append(rm.rank(rm.jacobian(cfg, thg, U, torch.tensor([True, False])))[0])
    stats["material NLOS pair"].append(rm.rank(rm.jacobian(cfg, thm, U, F.repeat(2), mode="material"))[0])
for k, want in [("LOS pair", 4), ("NLOS pair", 3), ("LOS+NLOS pair", 4), ("material NLOS pair", 4)]:
    frac = np.mean(np.array(stats[k]) == want)
    ok_all &= check(f"Proposition: random {k} has rank {want}", frac == 1.0, f"{100 * frac:.0f}% of 300 pairs")

# Proposition (material): two users with the same incidence angle lose one rank
U_sym = torch.tensor([[30.0, 20.0, 1.5], [30.0, -20.0, 1.5]])       # mirror pair about the plane y = 0 through the BS
ci = rm.geometry(cfg, thm[0], thm[1], U_sym)[4]
r_sym = rm.rank(rm.jacobian(cfg, thm, U_sym, F.repeat(2), mode="material"))[0]
ok_all &= check("Proposition: equal incidence angles -> material pair rank 3", r_sym == 3,
                f"cos theta_i = {ci.numpy().round(6)}, rank {r_sym}")

# Remark (metal): sigma is practically unidentifiable for a good conductor
thmet = torch.tensor([D, PHI, 1.0, 7.0])
U2 = users[:10]
r_met, s_met = rm.rank(rm.jacobian(cfg, thmet, U2, F.repeat(10), mode="material"))
r_con, s_con = rm.rank(rm.jacobian(cfg, thm, U2, F.repeat(10), mode="material"))
Jmet = rm.jacobian(cfg, thmet, U2, F.repeat(10), mode="material").reshape(-1, 4)
sig_share = (Jmet[:, 3].norm() / Jmet.norm()).item()
log(f"  10 NLOS users: concrete rank {r_con}, cond {(s_con[0] / s_con[-1]).item():.2e}; metal rank {r_met}, "
    f"|d mu/d log sigma| share {sig_share:.1e}")
ok_all &= check("Remark: metal (sigma = 1e7 S/m) gives no information on sigma", r_met == 3 and r_con == 4,
                f"metal rank {r_met}, concrete rank {r_con}")

# Corollary (map prior): prior information on (d, phi) restores (d, phi, |Gamma|) for one NLOS user, never arg(Gamma)
Jn = rm.fisher(cfg, thg, u0, F)
Jp = torch.zeros(4, 4)
Jp[0, 0], Jp[1, 1] = 1 / 0.5 ** 2, 1 / 0.01 ** 2                  # 0.5 m and 0.01 rad map uncertainty
ev = torch.linalg.eigvalsh(Jn + Jp)
sub = torch.linalg.eigvalsh((Jn + Jp)[:3, :3])
ok_all &= check("Corollary: map prior -> (d, phi, |Gamma|) block nonsingular", sub.min().item() > 1e-9 * sub.max().item(),
                f"min eig {sub.min().item():.2e}")
ok_all &= check("Corollary: map prior never informs arg(Gamma) in NLOS", ev.min().abs().item() < 1e-9 * ev.max().item(),
                f"min eig {ev.min().item():.1e}")

log("\nSTEP 2: ALL CHECKS PASSED" if ok_all else "\nSTEP 2: SOME CHECKS FAILED")
with open(os.path.join(ART, "step2_checks.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
STEP2_OK = ok_all


# Extra facts quoted in the text (appended after the main checks; do not change the verdict above)
log("\n-- Facts quoted in paper/identifiability.tex --")
Jm = rm.jacobian(cfg, thmet, U2, F.repeat(10), mode="material").reshape(-1, 4)
cn = (Jm.norm(dim=0) / Jm.norm()).numpy()
log(f"  metal, column shares (d, phi, eps_r, log10 sigma): {np.array2string(cn, precision=2)}")
top = []
for n in range(500):
    Jl = rm.jacobian(cfg, thg, users[n:n + 1], T)[0, :, 3]
    sh = (Jl ** 2) / (Jl ** 2).sum()
    top.append(sh.max().item())
top = np.array(top)
log(f"  LOS users: share of arg(Gamma) information in the single best beam: median {np.median(top):.3f}, "
    f"10th pct {np.percentile(top, 10):.3f}, min {top.min():.3f}  (500 users)")
with open(os.path.join(ART, "step2_checks.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
sys.exit(0 if STEP2_OK else 1)
