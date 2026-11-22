"""Step 3 figure: what wide-beam reports pin down about facade B in Munich.

(a) users served by facade B, by LOS to the BS; (b) wall-offset bound vs number of users;
(c) permittivity bound vs number of users, 3.5 vs 28 GHz; (d) wall-offset bound vs number
of wide beams. Robust bounds (user position prior sigma_u, unknown wall-path phase), sigma_e = 1 dB.
Colours: categorical slots 1-2 of the dataviz reference palette (validated, light mode).
Writes ../paper/fig_crb.pdf.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
OUT = os.path.join(HERE, "..", "paper", "fig_crb.pdf")
COL = {"LOS": "#2a78d6", "NLOS": "#eb6834"}
INK, MUTED = "#2b2b2b", "#6b6b6b"
TEXT_W = 7.16

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.pad_inches": 0.02,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 10.5, "axes.labelsize": 11.0, "xtick.labelsize": 10.0, "ytick.labelsize": 10.0,
    "legend.fontsize": 9.5, "axes.linewidth": 0.8, "lines.linewidth": 1.8, "lines.markersize": 6.0,
    "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.28, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.handlelength": 2.2, "legend.labelspacing": 0.35,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
})


def q(z, key, col):
    x = z[key][:, col]
    x = x[np.isfinite(x)]
    return np.percentile(x, [10, 50, 90])


def despine(ax, tag):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(tag, loc="left", fontsize=12, fontweight="bold", pad=6, color=INK)


z3 = np.load(os.path.join(ART, "step3_crb_B_3.5GHz.npz"))
z28 = np.load(os.path.join(ART, "step3_crb_B_28GHz.npz"))
Ns = z3["Ns"]
fig, ax = plt.subplots(2, 2, figsize=(TEXT_W, 5.15), constrained_layout=True)
ax = ax.ravel()

# (a) map
a = ax[0]
U, los = z3["users"], z3["los"]
for name, m in [("LOS", los), ("NLOS", ~los)]:
    a.scatter(U[m, 0], U[m, 1], s=7, color=COL[name], lw=0, label=name, rasterized=True)
tri = z3["facade_tri"]
t = np.array([-np.cross([0, 0, 1], tri[1] - tri[0])[1], 0])
xy = tri[:, :2]
ends = xy[[np.argmin(xy @ (tri[1] - tri[0])[:2]), np.argmax(xy @ (tri[1] - tri[0])[:2])]]
a.plot(ends[:, 0], ends[:, 1], color=INK, lw=4, solid_capstyle="butt")
a.annotate("facade B", ends.mean(0), xytext=(28, 55), textcoords="data", fontsize=10, color=INK,
           ha="center", va="top", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
bs = z3["bs"]
a.plot(bs[0], bs[1], marker="^", ms=9, color=INK, mec="white", mew=0.7, ls="none")
a.annotate("BS", bs[:2], xytext=(6, 4), textcoords="offset points", fontsize=10, color=INK)
a.set_aspect("equal")
a.set_xlabel("x [m]")
a.set_ylabel("y [m]")
a.grid(False)
a.set_ylim(top=180)                      # headroom so the legend clears the user grid
a.legend(loc="upper left", markerscale=2.2, handletextpad=0.4, borderaxespad=0.3)
despine(a, "(a)")

# (b) wall offset vs N
b = ax[1]
for name in ["LOS", "NLOS"]:
    for kind, ls in [("robust1", "-"), ("robust3", "--")]:
        Q = np.array([q(z3, f"crb_{kind}_{name}_{N}", 0) for N in Ns])
        if kind == "robust1":
            b.fill_between(Ns, Q[:, 0], Q[:, 2], color=COL[name], alpha=0.15, lw=0)
        b.plot(Ns, Q[:, 1], ls, color=COL[name], marker="o" if kind == "robust1" else None, ms=5.5)
b.set_xscale("log")
b.set_yscale("log")
b.set_xticks(Ns)
b.set_xticklabels([str(n) for n in Ns])
b.minorticks_off()
b.set_xlabel("users $N$")
b.set_ylabel(r"$\sqrt{\mathrm{CRB}}$ wall offset $d$ [m]")
b.set_ylim(0.3, 3000)                    # headroom for the two-column legend
h = [mpl.lines.Line2D([], [], color=COL["LOS"]), mpl.lines.Line2D([], [], color=COL["NLOS"]),
     mpl.lines.Line2D([], [], color=MUTED, ls="-"), mpl.lines.Line2D([], [], color=MUTED, ls="--")]
b.legend(h, ["LOS", "NLOS", r"$\sigma_u=1$ m", r"$\sigma_u=3$ m"], loc="upper right",
         ncol=1, borderaxespad=0.4)
despine(b, "(b)")

# (c) permittivity vs N, 3.5 vs 28 GHz
c = ax[2]
for name in ["LOS", "NLOS"]:
    for z, ls in [(z3, "-"), (z28, "--")]:
        Q = np.array([q(z, f"crb_robust1_{name}_{N}", 2) for N in Ns])
        c.plot(Ns, Q[:, 1] / 7.074, ls, color=COL[name], marker="o" if ls == "-" else None, ms=5.5)
c.set_xscale("log")
c.set_yscale("log")
c.set_xticks(Ns)
c.set_xticklabels([str(n) for n in Ns])
c.minorticks_off()
c.set_xlabel("users $N$")
c.set_ylabel(r"$\sqrt{\mathrm{CRB}}\,\epsilon_\mathrm{r}\ /\ \epsilon_\mathrm{r}$")
h = [mpl.lines.Line2D([], [], color=COL["LOS"], ls="-"),
     mpl.lines.Line2D([], [], color=COL["NLOS"], ls="-"),
     mpl.lines.Line2D([], [], color=MUTED, ls="-"),
     mpl.lines.Line2D([], [], color=MUTED, ls="--")]
c.legend(h, ["LOS", "NLOS", "3.5 GHz", "28 GHz"], loc="upper right", ncol=1)
despine(c, "(c)")

# (d) wall offset vs number of wide beams, N = 10
d = ax[3]
Ks = [8, 16, 32]
for j, name in enumerate(["LOS", "NLOS"]):
    Q = np.array([q(z3, f"crbK{K}_{name}_10", 0) for K in Ks])
    x = np.arange(3) + (j - 0.5) * 0.18
    d.errorbar(x, Q[:, 1], yerr=[Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 1]], color=COL[name], fmt="o", ms=7,
               elinewidth=1.4, capsize=3, capthick=1.1, label=name)
d.set_xticks(range(3))
d.set_xticklabels([str(k) for k in Ks])
d.set_xlim(-0.5, 2.5)
d.set_ylim(0, None)
d.set_xlabel("wide beams $K$")
d.set_ylabel(r"$\sqrt{\mathrm{CRB}}$ $d$ [m]")
d.legend(loc="upper right", handletextpad=0.4)
despine(d, "(d)")

fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print("saved", OUT)
