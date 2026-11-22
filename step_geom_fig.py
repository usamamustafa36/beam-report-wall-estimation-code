"""Top view of the facade-B image construction used in the system model.

Uses the stored base station, wall, and served users in step3_crb_B_3.5GHz.npz.
No new channel or bound is computed. Writes ../paper/fig_geom.pdf.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
OUT = os.path.join(HERE, "..", "paper", "fig_geom.pdf")
COL = {"LOS": "#2a78d6", "NLOS": "#eb6834"}
INK, MUTED = "#2b2b2b", "#6b6b6b"

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.pad_inches": 0.02,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.linewidth": 0.8, "lines.linewidth": 1.8,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "text.color": INK,
})

z = np.load(os.path.join(ART, "step3_crb_B_3.5GHz.npz"))
d, phi = float(z["theta"][0]), float(z["theta"][1])
n = np.array([np.cos(phi), np.sin(phi)])
tangent = np.array([-n[1], n[0]])
bs = z["bs"][:2].astype(float)
bbar = bs - 2.0 * (n @ bs - d) * n
tri = z["facade_tri"].astype(float)
p1, p2 = tri[0, :2], tri[1, :2]
along = p2 - p1
U = z["users"][:, :2].astype(float)
los = z["los"].astype(bool)


def specular(u):
    den = n @ (u - bbar)
    xi = (d - n @ bbar) / den
    return bbar + xi * (u - bbar), xi


def on_facade(x):
    t = np.dot(x - p1, along) / np.dot(along, along)
    return 0.12 < t < 0.88 and np.linalg.norm(x - (p1 + t * along)) < 1.0


# LOS reflection near the left of the facade, NLOS near the right, so the rays separate.
picks = []
for want_los, t_goal in [(True, 0.22), (False, 0.82)]:
    best, best_score = None, np.inf
    for i, u in enumerate(U):
        if bool(los[i]) != want_los:
            continue
        x, xi = specular(u)
        t = float(np.dot(x - p1, along) / np.dot(along, along))
        if not (0 < xi < 1 and on_facade(x)):
            continue
        depth = n @ (u - (p1 + t * along))
        if not (35 < depth < 80):
            continue
        score = abs(t - t_goal)
        if score < best_score:
            best, best_score = i, score
    if best is None:
        raise RuntimeError("no served user in the requested part of the facade")
    picks.append(best)

fig, ax = plt.subplots(figsize=(3.15, 4.15), constrained_layout=True)
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=INK, lw=5, solid_capstyle="butt", zorder=3)
# Normal at the left end, on the base-station side, clear of both reflections.
n0 = p1 + 4 * n
n1 = p1 + 18 * n
ax.annotate("", xy=n1, xytext=n0, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
ax.text(*(n1 - 2 * tangent + 4 * n), "normal", fontsize=13, color=INK, ha="right", va="bottom")
# One image stub, on the far side of the wall. The image itself is outside the window.
image_from = p1 + 0.55 * along
ax.plot([image_from[0], (image_from - 16 * n)[0]], [image_from[1], (image_from - 16 * n)[1]],
        color=INK, lw=1.2, ls=(0, (3, 2)), zorder=2)
ax.text(*(image_from - 18 * n), "image", fontsize=13, color=INK, ha="left", va="center")
ax.text(*(p2 + 3 * tangent), "facade B", fontsize=14, color=INK, ha="left", va="top")

names = ["LOS", "NLOS"]
offsets = {"LOS": (7, 8), "NLOS": (-28, -12)}
for i, name in zip(picks, names):
    u = U[i]
    x, _ = specular(u)
    ax.plot([bs[0], x[0], u[0]], [bs[1], x[1], u[1]], color=COL[name], lw=1.7, zorder=2)
    ax.scatter([x[0]], [x[1]], s=36, color=COL[name], zorder=4, edgecolors="white", linewidths=0.6)
    ax.scatter([u[0]], [u[1]], s=80, color=COL[name], zorder=5, edgecolors="white", linewidths=0.8)
    ax.annotate(name, u, xytext=offsets[name], textcoords="offset points", fontsize=14,
                color=COL[name], fontweight="bold")
    if name == "LOS":
        ax.plot([bs[0], u[0]], [bs[1], u[1]], color=COL[name], lw=1.3, ls=":", zorder=1)

ax.plot(bs[0], bs[1], marker="^", ms=13, color=INK, mec="white", mew=0.8, ls="none", zorder=6)
ax.annotate("BS", bs, xytext=(8, -2), textcoords="offset points", fontsize=14, fontweight="bold")

ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
# Window on the array, the wall, and the two users. The image is outside it.
xs = np.r_[bs[0], p1[0], p2[0], U[picks, 0], (image_from - 20 * n)[0]]
ys = np.r_[bs[1], p1[1], p2[1], U[picks, 1], (image_from - 22 * n)[1]]
pad_x, pad_y = 6, 5
ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(True, alpha=0.28, lw=0.6)
ax.set_axisbelow(True)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=160, bbox_inches="tight", pad_inches=0.02)
print("xlim", ax.get_xlim(), "ylim", ax.get_ylim())
print("saved", OUT)
print("users", picks)
print("LOS", U[picks[0]], "NLOS", U[picks[1]])
print("image", bbar)
