# -*- coding: utf-8 -*-
"""Graphical abstract | What the fusion assumes decides the outcome.

Med-X requires a graphical abstract, at a minimum of 531 x 1328 px (h x w),
legible at 13 x 5 cm, and explicitly **not** taken from the figures inside the
article. This one therefore shares no panel with Figs. 1-5: the misalignment is
drawn as two outlines rather than as the distribution in Fig. 2b, the
correspondence ladder is a schematic that appears nowhere in the paper, and the
outcome is a signed change against the single-frame baseline rather than the
Dice boxes of Fig. 3a.

The three panels are the argument in order: the frames do not correspond, a
fusion scheme has to assume something about that correspondence, and what it
assumes decides whether the second frame helps or hurts.

Numbers are the patient-level paired median differences in outputs/patient_level.csv,
so this figure cannot drift from Table 1 and Section 3.3.

Palette discipline, as everywhere in this project
  Blue / teal are EXCLUSIVELY frame identity. Red marks the baseline and the
  loss below it. Grey is neutral structure.

Output: graphical_abstract.{pdf,tiff,png}  (PDF and TIFF are Med-X preferred)
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.8,
})

# Repository root, resolved from this file so the scripts run from a clone.
# Override with EGC_REPO if the read-outs live elsewhere.
REPO = os.environ.get("EGC_REPO") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))
C_WLI, C_NBI = "#0F4D92", "#42949E"
C_SIG, C_NEU, C_PALE = "#B64342", "#767676", "#D8D8D8"

CM = 1 / 2.54
FIGW, FIGH = 13 * CM, 5 * CM          # the size the journal asks it to be legible at

d = pd.read_csv(os.path.join(REPO, "outputs", "patient_level.csv"), encoding="utf-8-sig")


def diff(frame, arm, base):
    r = d[(d.frame == frame) & (d.metric == "dice") & (d.arm == arm) & (d.baseline == base)]
    return float(r.median_diff.iloc[0])


SCHEMES = [("Channel\nconcatenation", "every pixel", "early_fusion"),
           ("Bottleneck\nconcatenation", "a 1/32 grid", "mid_fusion"),
           ("Cross-attention\n(proposed)", "nothing", "ours")]

fig = plt.figure(figsize=(FIGW, FIGH))
gs = fig.add_gridspec(1, 3, width_ratios=[0.72, 0.98, 1.30], wspace=0.16,
                      left=0.012, right=0.988, top=0.84, bottom=0.20)

# ---- 1 | the two frames do not correspond ---------------------------------
ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("Two frames, one lesion", fontsize=7.4, fontweight="bold", pad=3,
             color="black")
# outlines deliberately offset and differently scaled: that IS the finding
a = Ellipse((0.40, 0.56), 0.44, 0.34, angle=-14, fill=False, ec=C_WLI, lw=1.6)
b = Ellipse((0.58, 0.46), 0.60, 0.46, angle=8, fill=False, ec=C_NBI, lw=1.6)
ax.add_patch(Ellipse((0.40, 0.56), 0.44, 0.34, angle=-14, fc=C_WLI, alpha=0.10, lw=0))
ax.add_patch(Ellipse((0.58, 0.46), 0.60, 0.46, angle=8, fc=C_NBI, alpha=0.10, lw=0))
ax.add_patch(a); ax.add_patch(b)
ax.text(0.20, 0.86, "white light", fontsize=6.2, color=C_WLI, fontweight="bold")
ax.text(0.62, 0.21, "narrow band", fontsize=6.2, color=C_NBI, fontweight="bold")
ax.text(0.5, 0.05, "they overlap by a third", fontsize=6.4, color=C_SIG,
        ha="center", va="bottom", fontweight="bold")

# ---- 2 | what a fusion scheme assumes -------------------------------------
ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("What the fusion assumes", fontsize=7.4,
             fontweight="bold", pad=3, color="black")
# grid density is the visual metaphor for the assumed scale: fine -> coarse -> none
for i, (n, lab) in enumerate([(8, "every pixel"), (3, "a 1/32 grid"), (0, "nothing")]):
    x0 = 0.045 + i * 0.335
    ax.add_patch(Rectangle((x0, 0.42), 0.25, 0.32, fill=False, ec=C_NEU, lw=0.9))
    for k in range(1, n):
        f = k / n
        ax.plot([x0 + 0.25 * f] * 2, [0.42, 0.74], color=C_NEU, lw=0.35, alpha=0.75)
        ax.plot([x0, x0 + 0.25], [0.42 + 0.32 * f] * 2, color=C_NEU, lw=0.35, alpha=0.75)
    if n == 0:
        ax.text(x0 + 0.125, 0.58, "?", fontsize=11, color=C_NEU, ha="center",
                va="center", fontweight="bold")
    ax.text(x0 + 0.125, 0.34, lab, fontsize=6.2, color="black", ha="center", va="top")
ax.annotate("", xy=(0.955, 0.20), xytext=(0.045, 0.20),
            arrowprops=dict(arrowstyle="->", color=C_NEU, lw=0.9))
ax.text(0.5, 0.10, "coarser assumption", fontsize=6.2, color=C_NEU, ha="center",
        va="top", style="italic")

# ---- 3 | what that costs, against a single frame --------------------------
ax = fig.add_subplot(gs[0, 2])
ax.set_title("Change vs one frame alone", fontsize=7.4, fontweight="bold",
             pad=3, color="black")
h, gap = 0.30, 0.36
ys = []
for i, (_, _, key) in enumerate(SCHEMES):
    for j, (frame, base, col) in enumerate((("WLI", "wli_only", C_WLI),
                                            ("NBI", "nbi_only", C_NBI))):
        arm = f"early_fusion_{frame.lower()}" if key == "early_fusion" else key
        v = diff(frame, arm, base)
        y = -(i * (2 * h + gap) + j * h)
        ys.append(y)
        ax.barh(y, v, height=h * 0.86, color=col if v > 0 else C_SIG,
                alpha=1.0 if v > 0 else 0.9, lw=0)
        ax.text(v + (0.0035 if v > 0 else -0.0035), y, f"{v:+.3f}",
                fontsize=5.8, va="center", ha="left" if v > 0 else "right",
                color=col if v > 0 else C_SIG)
for i, (name, _, _) in enumerate(SCHEMES):
    y = -(i * (2 * h + gap)) + h * 0.80
    ax.text(-0.101, y, name.replace(chr(10), " "), fontsize=6.0, va="bottom",
            ha="left", color="black")
ax.axvline(0, color=C_SIG, lw=1.0)
ax.text(-0.0022, min(ys) - 0.30, "one frame alone", fontsize=5.4, color=C_SIG,
        va="bottom", ha="right")
ax.set_xlim(-0.101, 0.027)
ax.set_ylim(min(ys) - 0.42, h * 1.55)
ax.set_yticks([])
ax.set_xticks([-0.08, -0.04, 0])
ax.tick_params(axis="x", labelsize=5.8, length=2, pad=1)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.set_xlabel("difference in area overlap", fontsize=5.8, labelpad=0.5)

fig.text(0.5, 0.028,
         "Assuming the frames match pixel for pixel costs more than the second frame "
         "contributes. Assuming less recovers it.",
         fontsize=6.6, ha="center", va="bottom", color="black")

base = os.path.join(OUT, "graphical_abstract")
for ext, kw in ((".pdf", {}), (".png", dict(dpi=600)), (".tiff", dict(dpi=600))):
    fig.savefig(base + ext, bbox_inches="tight", pad_inches=0.02, **kw)
plt.close(fig)

from PIL import Image
im = Image.open(base + ".png")
print(f"saved -> graphical_abstract.{{pdf,png,tiff}}")
print(f"png {im.size[0]} x {im.size[1]} px (w x h); journal minimum 1328 x 531")
print(f"target print size {13:.0f} x {5:.0f} cm")
for name, _, key in SCHEMES:
    for frame, b in (("WLI", "wli_only"), ("NBI", "nbi_only")):
        arm = f"early_fusion_{frame.lower()}" if key == "early_fusion" else key
        print(f"  {name.replace(chr(10),' '):32s} {frame}  {diff(frame, arm, b):+.4f}")
