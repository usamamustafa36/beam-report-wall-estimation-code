"""Step 1: sanity checks of the closed-form model and numerical rank checks for the lemmas.

Scene: BS at (0, 0, 25) m with a 64-element ULA along y (broadside +x), 3.5 GHz.
Wall: vertical plane x = 60 m facing the BS, i.e. n = (cos pi, sin pi, 0), d = -60.
Users at 1.5 m height between the BS and the wall.
Prints PASS/FAIL lines and writes artifacts/step1_checks.txt.
"""
import math, os, sys
import numpy as np
import torch
import rsrp_model as rm

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ART, exist_ok=True)
lines = []


def log(s):
    print(s, flush=True)
    lines.append(s)


def check(name, ok, detail=""):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


rank, null_space = rm.rank, rm.null_space


cfg = rm.Config()
gen = torch.Generator().manual_seed(42)
D, PHI = -60.0, math.pi
users = torch.stack([10 + 40 * torch.rand(200, generator=gen), -40 + 80 * torch.rand(200, generator=gen),
                     torch.full((200,), 1.5)], 1)
ok_all = True

# 1. image-source geometry: point on the wall, equal angles, path length = two segments
th = torch.tensor([D, PHI])
L_d, L_r, k_d, k_r, cos_i, valid = rm.geometry(cfg, th[0], th[1], users)
n = rm.wall_normal(th[1])
b_img = cfg.bs - 2 * (cfg.bs @ n - th[0]) * n
v = users - b_img
t = (th[0] - b_img @ n) / (v @ n)
x_r = b_img + t[:, None] * v
on_wall = torch.abs(x_r @ n - th[0]).max().item()
seg = (torch.linalg.norm(x_r - cfg.bs, dim=1) + torch.linalg.norm(users - x_r, dim=1) - L_r).abs().max().item()
inc = (cfg.bs - x_r) / torch.linalg.norm(cfg.bs - x_r, dim=1, keepdim=True)
out = (users - x_r) / torch.linalg.norm(users - x_r, dim=1, keepdim=True)
snell = ((inc @ n) - (out @ n)).abs().max().item()
ok_all &= check("reflection point lies on the wall", on_wall < 1e-9, f"max |n'x-d| = {on_wall:.1e} m")
ok_all &= check("reflected length = BS->wall + wall->user", seg < 1e-9, f"max err {seg:.1e} m")
ok_all &= check("angle of incidence = angle of reflection", snell < 1e-12, f"max err {snell:.1e}")
ok_all &= check("all users have a valid specular point", bool(valid.all()), f"{int(valid.sum())}/{len(valid)}")

# 2. wide codebook identical to the TCCN code (revision_common.wide_codebook)
ang = np.linspace(-1, 1, 16, endpoint=False)
C_ref = np.zeros((16, 64), complex)
C_ref[:, :16] = np.exp(1j * np.pi * np.outer(ang, np.arange(16))) / np.sqrt(16)
err = np.abs(cfg.C.numpy() - C_ref).max()
ok_all &= check("wide codebook matches revision_common.wide_codebook", err < 1e-12, f"max diff {err:.1e}")

# 3. LOS-only users: the strongest wide beam is the grid point nearest the direct spatial frequency
h_d = (cfg.lam / (4 * math.pi * L_d))[:, None] * rm.steering(cfg, k_d)
best = (torch.abs(h_d @ cfg.C.conj().T) ** 2).argmax(1)
s_d = k_d @ cfg.axis
nearest = torch.round((s_d + 1) * 8).long() % 16
match = (best == nearest).float().mean().item()
ok_all &= check("LOS-only: best wide beam = nearest grid point", match > 0.95, f"{100 * match:.1f}% of users")

# 4. Fresnel sanity
g_metal = rm.fresnel_te(torch.tensor(rm.itu_eps("metal", cfg.freq)), torch.tensor(0.7)).abs().item()
eps_g = rm.itu_eps("glass", cfg.freq)
g_glass0 = rm.fresnel_te(torch.tensor(eps_g), torch.tensor(1.0))
ref = (1 - np.sqrt(eps_g)) / (1 + np.sqrt(eps_g))
ok_all &= check("metal |Gamma| ~ 1", abs(g_metal - 1) < 1e-3, f"|Gamma| = {g_metal:.5f}")
ok_all &= check("glass normal incidence = (1-sqrt(eps))/(1+sqrt(eps))", abs(g_glass0.item() - ref) < 1e-12,
                f"{g_glass0.item():.4f}")
graz = rm.fresnel_te(torch.tensor(rm.itu_eps("concrete", cfg.freq)), torch.tensor(1e-4)).item()
ok_all &= check("grazing incidence -> Gamma = -1", abs(graz + 1) < 1e-3, f"{graz:.4f}")

# 5. rank checks, Gamma mode, theta = (d, phi, |Gamma|, arg Gamma)
log("\n-- Rank of the RSRP Jacobian, theta = (d, phi, |Gamma|, arg Gamma) --")
thg = torch.tensor([D, PHI, 0.5, 0.7])
T, F = torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
res = {}
for name, idx, los in [("1 user NLOS", [0], F), ("1 user LOS", [0], T),
                       ("2 users NLOS", [0, 1], F.repeat(2)), ("2 users LOS", [0, 1], T.repeat(2)),
                       ("10 users NLOS", list(range(10)), F.repeat(10)),
                       ("10 users LOS", list(range(10)), T.repeat(10))]:
    J = rm.jacobian(cfg, thg, users[idx], los)
    r, s = rank(J)
    res[name] = r
    log(f"  {name:14s} rank {r}/4   singular values {np.array2string(s.numpy(), precision=2)}")
    if "NLOS" in name:
        z = (J[..., 3].abs().max() / J.abs().max()).item()
        ok_all &= check(f"  {name}: d mu/d arg(Gamma) = 0 (Lemma 1)", z < 1e-12, f"relative max {z:.1e}")
ok_all &= check("Lemma 2: 1 user NLOS rank <= 2", res["1 user NLOS"] <= 2, f"rank {res['1 user NLOS']}")
ok_all &= check("Lemma 2: 1 user LOS rank <= 3", res["1 user LOS"] <= 3, f"rank {res['1 user LOS']}")
ok_all &= check("2 users LOS: full rank 4", res["2 users LOS"] == 4)
ok_all &= check("2 users NLOS: rank 3 (all but arg Gamma)", res["2 users NLOS"] == 3)

# 6. the null directions are the predicted ones
log("\n-- Null directions of a single user --")


def observables(theta, u):
    """(spatial frequency of reflected path, log reflected amplitude, reflected phase)."""
    _, L_r, _, k_r, _, _ = rm.geometry(cfg, theta[0], theta[1], u)
    s_r = (k_r @ cfg.axis)[0]
    amp = torch.log(theta[2] / L_r[0])
    ph = theta[3] - 2 * math.pi * L_r[0] / cfg.lam
    return torch.stack([s_r, amp, ph])


u0 = users[:1]
G = torch.autograd.functional.jacobian(lambda th: observables(th, u0), thg)    # (3, 4)
for name, los, keep, dim in [("NLOS", F, [0, 1], 2), ("LOS", T, [0, 1, 2], 1)]:
    Ns = null_space(rm.jacobian(cfg, thg, u0, los))
    ok_all &= check(f"  {name}: null space has dimension {dim}", Ns.shape[0] == dim, f"found {Ns.shape[0]}")
    for v in Ns:
        move = (G[keep] @ v).abs().max().item()
        log(f"  {name}: null vector (d, phi, |G|, argG) = {np.array2string(v.numpy(), precision=4)}")
        ok_all &= check(f"  {name}: observables {['s_r', 'log|c|', 'arg c'][:len(keep)]} constant along it",
                        move < 1e-8, f"max change {move:.1e}")
    if name == "NLOS":
        geo = Ns[:, :3]                     # the null space also contains a (d, phi, |Gamma|) direction
        _, sg, _ = torch.linalg.svd(geo)
        ok_all &= check("  NLOS: one null direction mixes geometry and |Gamma| (Lemma 2)", sg[0].item() > 0.5,
                        f"largest geometry component {sg[0].item():.3f}")
    else:
        v = Ns[0]
        ok_all &= check("  LOS: null vector moves arg(Gamma) together with d (phase confounded with path length)",
                        abs(v[3].item()) > 1e-3 and abs(v[0].item()) > 1e-4,
                        f"arg-Gamma {v[3].item():.4f}, d {v[0].item():.4f}")

# 7. material mode, theta = (d, phi, eps_r, log10 sigma): angle diversity across users
log("\n-- Material mode, theta = (d, phi, eps_r, log10 sigma), NLOS --")
for mat in ["concrete", "glass", "brick"]:
    a, b, c, dd = rm.ITU[mat]
    f = cfg.freq / 1e9
    thm = torch.tensor([D, PHI, a * f ** b, math.log10(c * f ** dd)])
    for N in [1, 2, 3, 5, 10]:
        r, s = rank(rm.jacobian(cfg, thm, users[:N], F.repeat(N), mode="material"))
        log(f"  {mat:9s} N={N:2d} NLOS rank {r}/4   cond {(s[0] / s[-1]).item():.2e}")

log("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
with open(os.path.join(ART, "step1_checks.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
sys.exit(0 if ok_all else 1)
