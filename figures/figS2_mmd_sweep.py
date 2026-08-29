# -*- coding: utf-8 -*-
"""Supplementary Fig. S2 | Distribution alignment costs performance at every weight.

Core conclusion this figure must defend
---------------------------------------
Adding a term that aligns the statistical distributions of the two streams, as
proposed elsewhere for this task, does not help here at any weight tested, and
hurts more as the weight rises. The comparison is exact: `mmd_fusion` differs
from `mid_fusion` only by that term, so weight 0 is the same architecture with
the term switched off.

Why the weights are drawn as categories and not on a log axis
-------------------------------------------------------------
One of the five levels is zero, which a log axis cannot place. They are five
tested settings rather than a sampled continuum, so a categorical axis states
what was actually run and avoids implying interpolation between them.

Two of the five carry meaning beyond their value and are marked: 1e-4 is the
weight the original authors report, and 0.3 is the value calibrated on the
development fold so the term is worth roughly 1-5% of the training objective.
The gap between them is the point of the panel: a loss weight does not transfer
across architectures, and at the authors' value the term is numerically close to
switched off.

Data: EGC-DualSeg results/predictions_*.csv. Patient level, n = 77, seeds
      averaged into images and images into patients within a reference frame.
Backend: Python / matplotlib (project default).
Output: figS2_mmd_sweep.{svg,pdf,tiff,png}

Palette discipline
  Blue / teal are EXCLUSIVELY frame identity. Red is reserved for reference
  lines; grey for neutral annotation.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import friedmanchisquare

# ---- MANDATORY: editable SVG/PDF text (before any figure is created) -------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

# Repository root, resolved from this file so the scripts run from a clone.
# Override with EGC_REPO if the read-outs live elsewhere.
REPO = os.environ.get("EGC_REPO") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
OUT = os.path.dirname(os.path.abspath(__file__))

C_WLI = "#0F4D92"
C_NBI = "#42949E"
C_SIG = "#B64342"
C_NEU = "#767676"

KEY = ["case", "scale", "ref_modality"]
PKEY = ["case", "ref_modality"]
# (config, weight, tick label, note)
LEVELS = [("mid_fusion", 0.0, "0", "term off"),
          ("mmd_fusion_w1e-4", 1e-4, "10$^{-4}$", "authors' value"),
          ("mmd_fusion_w0.01", 0.01, "0.01", None),
          ("mmd_fusion", 0.3, "0.3", "calibrated here"),
          ("mmd_fusion_w1.0", 1.0, "1.0", None)]
FRAMES = {"WLI": dict(colour=C_WLI, label="WLI"),
          "NBI": dict(colour=C_NBI, label="NBI")}


def to_patient(df, col):
    per_image = df.groupby(KEY + ["config"], as_index=False)[col].mean()
    return per_image.groupby(PKEY + ["config"], as_index=False)[col].mean()


files = glob.glob(os.path.join(RESULTS, "predictions_*.csv"))
if not files:
    raise SystemExit(f"no predictions_*.csv in {RESULTS}")
pred = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files],
                 ignore_index=True)
pat = to_patient(pred, "dice")

fig, ax = plt.subplots(figsize=(3.5, 2.7))
xs = np.arange(len(LEVELS))
report = {}
for fi, (frame, spec) in enumerate(FRAMES.items()):
    w = (pat[pat.ref_modality == frame]
         .pivot_table(index="case", columns="config", values="dice"))
    cols = [c for c, *_ in LEVELS]
    missing = [c for c in cols if c not in w.columns]
    if missing:
        raise SystemExit(f"{frame}: missing configs {missing}")
    W = w[cols].dropna()
    mean = np.array([W[c].mean() for c in cols])
    # Standard error across patients, not across seeds: the patient is the unit.
    sem = np.array([W[c].std(ddof=1) / np.sqrt(len(W)) for c in cols])
    st = friedmanchisquare(*[W[c] for c in cols])
    report[frame] = (len(W), mean, st.pvalue,
                     all(np.diff(mean) < 0))
    off = (fi - 0.5) * 0.06
    ax.errorbar(xs + off, mean, yerr=sem, color=spec["colour"], lw=1.5,
                marker="o", ms=4.2, capsize=2.2, elinewidth=0.9, zorder=3,
                label=f"{spec['label']}   Friedman p = {st.pvalue:.1e}")
    # The term-off level is the reference the rest are read against.
    ax.axhline(mean[0], color=spec["colour"], ls=":", lw=0.8, alpha=0.55,
               zorder=1)

ax.set_xticks(xs)
ax.set_xticklabels([lab for _, _, lab, _ in LEVELS])
# Below the tick labels rather than inside the axes: rotated in-axes notes
# collided with the white-light series and with the legend.
for i, (_, _, _, note) in enumerate(LEVELS):
    if note:
        ax.text(i, -0.115, note, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=5.8, color=C_NEU)
ax.set_xlim(-0.5, len(LEVELS) - 0.5)
ax.set_xlabel("Weight on the distribution-alignment term", fontsize=8, labelpad=12)
ax.set_ylabel("Dice, patient level", fontsize=8)
ax.legend(fontsize=6.6, loc="center left", handlelength=1.5,
          borderaxespad=0.3, labelspacing=0.3)
ax.text(1.0, 1.015, f"n = {report['WLI'][0]} patients", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=6.6, color=C_NEU)

base = os.path.join(OUT, "figS2_mmd_sweep")
for ext, kw in ((".svg", {}), (".pdf", {}), (".png", dict(dpi=600)),
                (".tiff", dict(dpi=600))):
    fig.savefig(base + ext, bbox_inches="tight", **kw)
plt.close(fig)

print("saved ->", base + ".{svg,pdf,png,tiff}")
for frame, (n, mean, p, mono) in report.items():
    print(f"\n{frame}  n = {n} patients   Friedman p = {p:.3g}   "
          f"monotonic: {'yes' if mono else 'NO'}")
    for (c, wt, *_), m in zip(LEVELS, mean):
        print(f"    weight {wt:<8g} mean Dice {m:.4f}")
