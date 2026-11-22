"""Figure: detecting a stale twin, and when the radio should overrule the map.

(a) detection probability vs wall displacement at P_FA = 1%: N = 10 from step5b_detect_more.py ci
    (2000 no-change trials, 95% intervals), N = 5, 20 from step5_detect.py, squares: reports re-traced
    in Sionna with the facade moved (step11_moved_detect.py, complete twin frozen at the map);
(b) no-change statistic on full Sionna reports, complete vs single-bounce twin, N = 10.
Full text width, two panels. The map-vs-radio crossover (step6_mismatch.py a) is reported in the text.
Colours as fig_crb (blue LOS, orange NLOS).
Writes ../paper/fig_detect.pdf.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
OUT = os.path.join(HERE, "..", "paper", "fig_detect.pdf")
COL = {"LOS": "#2a78d6", "NLOS": "#eb6834"}
INK, MUTED = "#2b2b2b", "#6b6b6b"
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.pad_inches": 0.02,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 10.5, "axes.labelsize": 11.0, "xtick.labelsize": 10.0, "ytick.labelsize": 10.0,
    "legend.fontsize": 9.5, "axes.linewidth": 0.8, "lines.linewidth": 1.8, "lines.markersize": 6.0,
    "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.28, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.handlelength": 2.4, "legend.labelspacing": 0.35,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
})


def despine(ax, tag):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(tag, loc="left", fontsize=12, fontweight="bold", pad=8, color=INK)


det = np.load(os.path.join(ART, "step5_detect.npz"))
fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.7), constrained_layout=True)

# (a) P_D vs displacement
a = ax[0]
D = det["deltas"]
ci = np.load(os.path.join(ART, "step5b_detect_ci.npz"))
mv = np.load(os.path.join(ART, "step11_moved_detect.npz"))
d3 = np.load(os.path.join(ART, "step16_delta3.npz"))
recorded = {"LOS": (170, 0.850, 0.794, 0.893), "NLOS": (189, 0.945, 0.904, 0.969)}
if not np.allclose(D, ci["deltas"]):
    raise SystemExit("step5 and step5b displacements differ; the 3 m point was not added")
for name in ["LOS", "NLOS"]:
    for N, ls in [(5, ":"), (20, "--")]:
        a.plot(D, det[f"{name}_{N}_pd"], ls, color=COL[name], lw=1.7)
    kexp, pd3, lo3, hi3 = recorded[name]
    T = d3[f"{name}_T"]
    thr3 = float(np.asarray(d3[f"{name}_thr"]).reshape(-1)[0])
    k = int(np.sum(T > thr3))
    if k != kexp or len(T) != 200:
        raise SystemExit(f"{name} 3 m count {k}/{len(T)} does not match {kexp}/200")
    x = np.sort(np.append(ci["deltas"].astype(float), 3.0))
    i = int(np.flatnonzero(np.isclose(x, 3.0))[0])
    pd = np.insert(ci[f"{name}_pfa1_pd"].astype(float), i, pd3)
    lo = np.insert(ci[f"{name}_pfa1_pdlo"].astype(float), i, lo3)
    hi = np.insert(ci[f"{name}_pfa1_pdhi"].astype(float), i, hi3)
    a.errorbar(x, pd, yerr=[pd - lo, hi - pd], fmt="-o",
               color=COL[name], lw=1.7, ms=5.5, elinewidth=1.1, capsize=3)
    thr = float(ci[f"{name}_pfa1_thr"])
    a.plot([2, 4], [np.mean(mv[f"{name}_{k}_T"][:, 0] > thr) for k in (2, 4)], "s", color=COL[name], mfc="white", mew=1.2, ms=7)
a.set_xscale("log", base=2)
ticks = np.sort(np.unique(np.append(np.asarray(D, dtype=float), 3.0)))
a.set_xticks(ticks)
a.set_xticklabels([f"{t:g}" for t in ticks])
a.minorticks_off()
a.set_ylim(-0.02, 1.05)
a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
a.set_xlabel(r"wall displacement $\Delta$ [m]")
a.set_ylabel("Detection probability\n" + r"(nominal $P_{\mathrm{FA}}=1\%$)")
h = [mpl.lines.Line2D([], [], color=COL["LOS"], lw=2.0),
     mpl.lines.Line2D([], [], color=COL["NLOS"], lw=2.0),
     mpl.lines.Line2D([], [], color=INK, ls=":", lw=1.7),
     mpl.lines.Line2D([], [], color=INK, ls="-", marker="o", ms=5.5, lw=1.7),
     mpl.lines.Line2D([], [], color=INK, ls="--", lw=1.7),
     mpl.lines.Line2D([], [], color=INK, ls="", marker="s", mfc="white", mew=1.1, ms=7)]
fig.legend(h, ["LOS", "NLOS", r"$N=5$", r"$N=10$", r"$N=20$", "re-traced"],
           loc="outside upper center", ncol=6, handlelength=2.2,
           columnspacing=1.15, borderpad=0.15, labelspacing=0.3)
despine(a, "(a)")

# (b) no-change statistic on full Sionna reports, N = 10
c = ax[1]
for name in ["LOS", "NLOS"]:
    for key, ls in [("Tsionna", "-"), ("Tsionna1b", "-.")]:
        t = np.sort(np.maximum(det[f"{name}_10_{key}"], 1e-2))
        c.plot(t, 1 - np.arange(len(t)) / len(t), ls, color=COL[name], lw=1.7)
    c.axvline(float(ci[f"{name}_pfa1_thr"]), color=COL[name], lw=0.9, alpha=0.55)
c.set_xscale("log")
c.set_xlim(1e-2, 1e4)
c.set_ylim(0, 1.02)
c.set_xlabel(r"$T$, no change")
c.set_ylabel(r"$P(T > t)$")
c.text(2.2e-2, 0.28, "complete twin", fontsize=10, color=INK, va="top")
c.text(40, 0.58, "single-bounce", fontsize=10, color=INK, va="center")
despine(c, "(b)")

fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print("saved", OUT)
