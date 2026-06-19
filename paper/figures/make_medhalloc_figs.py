"""Generate the two MedHal-Loc main figures from verified numbers.
Outputs PDF (for LaTeX) + PNG into this directory.
  fig_localization : hit@1 vs per-method random baseline (the headline panel)
  fig_coverage     : AdaTriple lift rises with extraction coverage (bottleneck)
Run:  python paper/figures/make_medhalloc_figs.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams["savefig.bbox"] = "tight"      # 裁掉画布四周白边
plt.rcParams["savefig.pad_inches"] = 0.02

# ---- Fig 1: localization-faithfulness panel (verified, n=295) ----
methods = ["AdaTriple\n(triple)", "NLI-clause\n(clause)",
           "SelfCheckGPT\n(sentence)", "FAVA\n(span)"]
hit1 = [39.3, 62.0, 81.4, 55.3]
rand = [36.0, 41.7, 68.7, 18.8]
sig = [False, True, True, True]
lifts = [3.3, 20.3, 12.6, 36.5]  # true lifts (match Table 3); avoid rounding-of-rounded

x = np.arange(len(methods))
w = 0.38
fig, ax = plt.subplots(figsize=(7.2, 4.0))
b1 = ax.bar(x - w/2, rand, w, label="Random baseline", color="#bdbdbd")
colors = ["#d62728" if not s else "#2ca02c" for s in sig]
b2 = ax.bar(x + w/2, hit1, w, label="hit@1 (method)", color=colors)
for i in range(len(methods)):
    lift = lifts[i]
    tag = f"+{lift:.1f}pp" + ("" if sig[i] else "\n(n.s.)")
    ax.annotate(tag, (x[i] + w/2, hit1[i] + 1.5), ha="center",
                fontsize=9, fontweight="bold",
                color=("#d62728" if not sig[i] else "#2ca02c"))
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("Localization hit@1 (%)")
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylim(0, 95)
ax.set_title("Localization faithfulness on MedHal-Loc (controlled, n=295)\n"
             "green = localizes above chance; red = AdaTriple at chance; "
             "response-level methods = 0% by construction", fontsize=9)
ax.legend(handles=[Patch(color="#bdbdbd", label="Random baseline"),
                   Patch(color="#2ca02c", label="Method hit@1 (above chance)"),
                   Patch(color="#d62728", label="Method hit@1 (at chance, n.s.)")],
          loc="upper left", fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_localization.pdf")
fig.savefig(OUT / "fig_localization.png", dpi=200)
plt.close(fig)

# ---- Fig 2: AdaTriple coverage bottleneck ----
types = ["relation\n44%", "entity\n54%", "invented\n60%", "mechanism\n79%"]
cov = [44, 54, 60, 79]
lift = [1.9, 2.8, 1.9, 6.8]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
sc = ax.scatter(cov, lift, s=140, c=cov, cmap="viridis",
                edgecolor="k", zorder=3)
for c, l, t in zip(cov, lift, ["relation", "entity", "invented", "mechanism"]):
    ax.annotate(t, (c, l), textcoords="offset points", xytext=(8, 5),
                fontsize=9)
# trend line
z = np.polyfit(cov, lift, 1)
xs = np.linspace(40, 82, 50)
ax.plot(xs, np.polyval(z, xs), "--", color="#888", lw=1.2, zorder=1)
ax.set_xlabel("AdaTriple triple-extraction coverage (%)")
ax.set_ylabel("Localization lift over chance (pp)")
ax.set_title("AdaTriple: localization lift tracks extraction coverage\n"
             "(all per-type lifts non-significant; coverage is the bottleneck)",
             fontsize=9)
ax.set_xlim(40, 84); ax.set_ylim(0, 8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_coverage.pdf")
fig.savefig(OUT / "fig_coverage.png", dpi=200)
plt.close(fig)

# ---- Fig 3: detection vs localization are decoupled ----
# x = mean detection F1 (v9); y = localization hit@1 lift (pp)
dm = {  # method: (det F1, loc lift pp, localizes?)
    "AdaTriple+": (0.609, 3.3, "chance"),
    "SelfCheckGPT-NLI": (0.611, 12.6, "yes"),
    "NLI-DeBERTa": (0.575, 0.0, "none"),
    "HHEM": (0.594, 0.0, "none"),
    "LLM-Judge": (0.478, 0.0, "none"),
}
fig, ax = plt.subplots(figsize=(6.8, 4.2))
cmap = {"yes": "#2ca02c", "chance": "#d62728", "none": "#7f7f7f"}
loff = {"AdaTriple+": (8, -15), "SelfCheckGPT-NLI": (10, 4),
        "NLI-DeBERTa": (-2, 10), "HHEM": (6, -16), "LLM-Judge": (10, 4)}
for m, (f1, lift, kind) in dm.items():
    ax.scatter(f1, lift, s=150, c=cmap[kind], edgecolor="k", zorder=3)
    ox, oy = loff.get(m, (8, 6))
    ax.annotate(m, (f1, lift), textcoords="offset points",
                xytext=(ox, oy), fontsize=9)
ax.axhline(0, color="#aaa", lw=0.8)
ax.annotate("competitive detection,\nchance localization",
            (0.609, 3.3), xytext=(0.55, 8.5),
            arrowprops=dict(arrowstyle="->", color="#d62728"),
            fontsize=8.5, color="#d62728")
ax.set_xlabel("Detection performance (mean F1, 5 datasets)")
ax.set_ylabel("Localization hit@1 lift over chance (pp)")
ax.set_title("Detection and localization faithfulness are decoupled\n"
             "(response-level detectors localize at 0 regardless of detection F1)",
             fontsize=9)
ax.set_ylim(-2.5, 15); ax.set_xlim(0.46, 0.64)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_decoupling.pdf"); fig.savefig(OUT / "fig_decoupling.png", dpi=200)
plt.close(fig)

# ---- Fig 4: FAVA vs AdaTriple hit@1 by error type ----
et = ["entity", "relation", "mechanism", "invented"]
fava = [64.3, 66.7, 57.3, 33.3]
ada = [35.7, 28.0, 56.0, 37.3]
x = np.arange(len(et)); w = 0.38
fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.bar(x - w/2, fava, w, label="FAVA (span)", color="#1f77b4")
ax.bar(x + w/2, ada, w, label="AdaTriple (triple)", color="#d62728")
ax.set_xticks(x); ax.set_xticklabels(et)
ax.set_ylabel("Localization hit (%)")
ax.set_ylim(0, 75)
ax.set_title("Localization by error type: FAVA vs AdaTriple\n"
             "(AdaTriple weakest where triple extraction is hardest)", fontsize=9)
ax.legend(fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_bytype.pdf"); fig.savefig(OUT / "fig_bytype.png", dpi=200)
plt.close(fig)

print("wrote fig_localization, fig_coverage, fig_decoupling, fig_bytype "
      "(.pdf+.png) to", OUT)
