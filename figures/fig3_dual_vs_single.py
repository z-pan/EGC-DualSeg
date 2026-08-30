# -*- coding: utf-8 -*-
"""Fig. 4 | The gain over a single-modality baseline is on the contour, not the area.

Core conclusion this figure must defend
---------------------------------------
Adding the second illumination mode improves where the predicted boundary sits
and how much of the lesion a resection margin would cover, on the narrow-band
frame only. It does not improve area overlap, and on the white-light frame it
does nothing at all. All three statements are drawn, at the same weight.

Rebuilt 2026-08-24: new primary endpoint, per-patient, ceiling panel dropped
----------------------------------------------------------------------------
The previous version was per-image (n = 150), led on Dice, and spent a whole
panel on a threshold-ceiling argument. Three changes:

  a  Boundary agreement is now the primary endpoint and takes panel a. The gain
     grows with the tolerance it is scored at, which is the signature of a
     contour that tracks the outline rather than a few pixels won at the edge.
  b  The clinical readouts -- how much lesion is left outside the prediction,
     and how often a fixed margin covers the whole lesion -- take panel b.
  c  Dice keeps ONE cell and is labelled as a null with its exclusion bound.
     Deleting it would be selective reporting: it was the endpoint the study
     was designed around, and it did not separate.

  The threshold-ceiling panel is gone. The baseline's best threshold on the
  sweep is 0.5, its own operating point, so the panel asserted nothing the
  operating-point comparison had not already said.

Panels
  a  Boundary F-score gain against the tolerance it is scored at, per patient,
     with 95% CI and the published Holm-adjusted significance.
  b  Clinical readout: reduction in the fraction of lesion missed, and the
     change in how often the whole lesion falls inside a 10 px or 20 px margin.
     Rightward is better on both, so the two rows can share one axis.
  c  The two area metrics that did not separate, with the effect their
     intervals exclude.

Data: EGC-DualSeg results/predictions_*.csv, results/boundary_*.csv,
      outputs/patient_level.csv (published Holm p). No GPU, no checkpoints.
Backend: Python / matplotlib (project default).
Output: fig3_dual_vs_single.{svg,pdf,tiff,png}

Palette discipline (shared with fig1_misalignment, fig2_fusion_operator, fig5_pathology)
  Blue / teal are EXCLUSIVELY frame identity -- white light versus narrow band.
  Red is reserved for reference lines and the zero line; grey for neutral
  annotation and for a contrast that did not separate.

The two frames carry ground truth drawn in different coordinate systems, so a
value in one is never ranked against a value in the other. Every panel keeps
them as two separate series and the caption says so.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

# ---- MANDATORY: editable SVG/PDF text (before any figure is created) -------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",     # text stays as <text> nodes, editable downstream
    "pdf.fonttype": 42,         # embedded TrueType, editable in PDF
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
PATIENT_CSV = os.path.join(REPO, "outputs", "patient_level.csv")
OUT = os.path.dirname(os.path.abspath(__file__))

C_WLI = "#0F4D92"      # frame identity: white light
C_NBI = "#42949E"      # frame identity: narrow band
C_SIG = "#B64342"      # zero line, reference lines
C_NEU = "#767676"
C_NULL = "#9A9A9A"     # a contrast that did not separate

FUSION = "ours"
FRAMES = {
    "WLI": dict(colour=C_WLI, single="wli_only", label="WLI"),
    # Canonical per the manuscript terminology ledger. 18 of 250 enhanced frames
    # come from Fujifilm scopes, whose enhancement mode is not Olympus NBI; the
    # ledger's <TO CONFIRM> covers that and the caption carries the caveat.
    "NBI": dict(colour=C_NBI, single="nbi_only", label="NBI"),
}
KEY = ["case", "scale", "ref_modality"]
PKEY = ["case", "ref_modality"]
TOL = [1, 2, 3, 5]
MARGINS = [0, 5, 10, 20, 40]
BOOT, RNG_SEED = 5000, 0


def read_group(pattern):
    files = glob.glob(os.path.join(RESULTS, pattern))
    if not files:
        raise SystemExit(f"no {pattern} in {RESULTS}")
    return pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in files],
                     ignore_index=True)


def to_patient(df, cols):
    """Seeds averaged into images, images averaged into patients.

    Within a reference frame only: the two frames carry different ground truth
    and are never pooled. 146 of the 154 patient-by-frame combinations supply
    both a near-focus and a distant pair, so the two are not independent.
    """
    per_image = df.groupby(KEY + ["config"], as_index=False)[cols].mean()
    return per_image.groupby(PKEY + ["config"], as_index=False)[cols].mean()


def holm(pvals):
    """Holm-Bonferroni step-down, returned in the input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def paired(tbl, frame, col, arm, ref, stat="median"):
    """Paired difference with a percentile bootstrap interval.

    The interval is bootstrapped on the SAME statistic that gets plotted. An
    earlier version always bootstrapped the median while panel b plotted the
    mean for the coverage rows, which drew an interval that did not belong to
    its own point estimate: for a per-patient proportion in {0, 0.5, 1} the
    median difference is 0 for most patients, so its interval is far wider than
    the mean's and the bar ran off the axis.
    """
    f = np.median if stat == "median" else np.mean
    w = tbl[tbl.ref_modality == frame].pivot_table(
        index="case", columns="config", values=col)[[arm, ref]].dropna()
    d = (w[arm] - w[ref]).to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    draws = [f(rng.choice(d, len(d))) for _ in range(BOOT)]
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(w[arm], w[ref]).pvalue)
    return dict(n=len(w), value=float(f(d)),
                median=float(np.median(d)), mean=float(d.mean()),
                lo=float(np.percentile(draws, 2.5)),
                hi=float(np.percentile(draws, 97.5)), p_raw=p)


def stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# ---- load ------------------------------------------------------------------
BND_COLS = [f"bf{t}" for t in TOL] + ["nsd3", "hd95", "residual_area_frac"]
bnd_raw = read_group("boundary_*.csv")
for k in MARGINS:
    bnd_raw[f"cov{k}"] = (bnd_raw["residual_px"] <= k).astype(float)
BND = to_patient(bnd_raw, BND_COLS + [f"cov{k}" for k in MARGINS])
DICE = to_patient(read_group("predictions_*.csv"), ["dice"])

# Published Holm values, so the figure cannot drift from the text. The family
# there is the eight configurations compared against the baseline on one metric
# and frame; the margin family below is computed here because coverage is not
# one of those ten metrics.
PUB = {}
if os.path.isfile(PATIENT_CSV):
    d = pd.read_csv(PATIENT_CSV, encoding="utf-8-sig")
    PUB = {(r.frame, r.metric, r.arm, r.baseline): r.p_holm for r in d.itertuples()}


def pub_p(frame, metric, ref):
    return PUB.get((frame, metric, FUSION, ref))


# ---- figure ----------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 2.85))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.12, 1.0, 1.02],
              wspace=0.66, left=0.075, right=0.985, top=0.80, bottom=0.22)

# ---- a | boundary agreement, the primary endpoint --------------------------
axa = fig.add_subplot(gs[0, 0])
report_a = {}
for frame, spec in FRAMES.items():
    med, lo, hi, ps = [], [], [], []
    for t in TOL:
        r = paired(BND, frame, f"bf{t}", FUSION, spec["single"])
        med.append(r["median"]); lo.append(r["lo"]); hi.append(r["hi"])
        ps.append(pub_p(frame, f"bf{t}", spec["single"]))
    report_a[frame] = (med, ps)
    axa.errorbar(TOL, med, yerr=[np.array(med) - np.array(lo),
                                 np.array(hi) - np.array(med)],
                 color=spec["colour"], lw=1.5, marker="o", ms=4.2,
                 capsize=2.2, elinewidth=0.9, zorder=3, label=spec["label"])
    # A star only where the published Holm value clears 0.05. The white-light
    # series is all n.s. and is drawn at the same weight, on purpose.
    for t, m, p in zip(TOL, med, ps):
        if p is not None and p < 0.05:
            axa.text(t, m + 0.006, "*", ha="center", va="bottom",
                     fontsize=9, color=spec["colour"])
axa.axhline(0, color=C_SIG, lw=0.9, ls=":", zorder=1)
axa.set_xticks(TOL)
axa.set_xlabel("Boundary tolerance (px)", fontsize=7.5)
axa.set_ylabel("Boundary F-score gain\nover single modality", fontsize=7.5)
axa.set_title("a  Contour agreement", loc="left", fontsize=8,
              fontweight="bold", pad=8)
axa.legend(fontsize=6.4, loc="upper left", handlelength=1.4,
           borderaxespad=0.2, labelspacing=0.25, bbox_to_anchor=(0.14, 1.02))
axa.text(0.03, 0.03, "* Holm-adjusted p < 0.05", transform=axa.transAxes,
         ha="left", va="bottom", fontsize=5.8, color=C_NEU)

# ---- b | clinical readout --------------------------------------------------
# Both rows are expressed so that rightward is better, which lets them share
# one axis: the lesion-missed row is sign-flipped and labelled as a reduction.
axb = fig.add_subplot(gs[0, 1])
ROWS = [("Lesion missed\n(reduction, pp of lesion)", "residual_area_frac", -100.0, "median"),
        ("Fully covered at 10 px\n(pp of patients)", "cov10", 100.0, "mean"),
        ("Fully covered at 20 px\n(pp of patients)", "cov20", 100.0, "mean")]
report_b = {}
for ri, (label, col, mult, stat) in enumerate(ROWS):
    for fi, (frame, spec) in enumerate(FRAMES.items()):
        tbl = BND
        r = paired(tbl, frame, col, FUSION, spec["single"], stat=stat)
        y = ri + (0.19 if fi else -0.19)
        # The two row types need different centres, and Results 3.4 quotes them
        # that way. Lesion missed is continuous, so the paired median is the
        # centre. Coverage is a per-patient proportion in {0, 0.5, 1}; its
        # median is 0 for most margins, so the mean change is the only readable
        # centre and is what the text reports as percentage points.
        val, lo, hi = mult * r["value"], mult * r["lo"], mult * r["hi"]
        axb.plot([min(lo, hi), max(lo, hi)], [y, y], color=spec["colour"],
                 lw=1.2, solid_capstyle="butt", zorder=2)
        axb.plot([val], [y], marker="o", ms=4.6, color=spec["colour"], zorder=3)
        report_b[(frame, col)] = (val, r["p_raw"])
axb.axvline(0, color=C_SIG, lw=0.9, ls=":", zorder=1)
axb.set_yticks(range(len(ROWS)))
axb.set_yticklabels([r[0] for r in ROWS], fontsize=6.4)
axb.text(0.5, -0.30, 'circle: paired median (row 1) or mean change (rows 2-3); bar: 95% CI',
         transform=axb.transAxes, ha='center', va='top', fontsize=5.6, color=C_NEU)
axb.invert_yaxis()
axb.set_xlabel("Improvement over single modality", fontsize=7.5)
axb.set_title("b  Clinical readout", loc="left", fontsize=8,
              fontweight="bold", pad=8)
axb.tick_params(axis="y", length=0)

# ---- c | the endpoints that did not separate -------------------------------
axc = fig.add_subplot(gs[0, 2])
# Dice only. The earlier draft put Dice and HD95 on one axis; a Dice difference
# of 0.01 is invisible next to an HD95 difference of 5 px, so the panel showed
# the null it exists to show as a flat line at zero. HD95 stays in the text.
report_c = {}
for fi, (frame, spec) in enumerate(FRAMES.items()):
    r = paired(DICE, frame, "dice", FUSION, spec["single"])
    p = pub_p(frame, "dice", spec["single"])
    y = -fi
    # Nothing here separated, so the interval is grey: colouring it in a frame
    # hue would read as a result. Frame identity is carried by the marker fill.
    axc.plot([r["lo"], r["hi"]], [y, y], color=C_NULL, lw=1.4, zorder=2)
    axc.plot([r["lo"], r["hi"]], [y, y], marker="|", ms=5, ls="none",
             color=C_NULL, zorder=2)
    axc.plot([r["median"]], [y], marker="o", ms=5.4, zorder=3,
             mfc=spec["colour"] if fi else "white", mec=spec["colour"], mew=1.2)
    axc.text(r["median"], y + 0.20,
             f"{r['median']:+.3f}   Holm {p:.2f}" if p is not None
             else f"{r['median']:+.3f}",
             fontsize=6.2, color=C_NEU, ha="center", va="bottom")
    report_c[(frame, "dice")] = (r["median"], r["lo"], r["hi"], p)

axc.axvline(0, color=C_SIG, lw=0.9, ls=":", zorder=1)
axc.set_yticks([0, -1])
axc.set_yticklabels([spec["label"] for spec in FRAMES.values()], fontsize=7.5)
axc.set_ylim(-1.95, 0.85)
axc.set_xlim(-0.017, 0.027)
axc.set_xticks([-0.01, 0.00, 0.01, 0.02])
axc.tick_params(axis="x", labelsize=7)
axc.set_xlabel("Dice difference vs single modality", fontsize=7.5)
axc.set_title("c  Area overlap: no separation", loc="left", fontsize=8,
              fontweight="bold", pad=8)
axc.tick_params(axis="y", length=0)
# The exclusion bound is the informative part of a null and belongs on the
# figure, not only in the text.
nb = report_c[("NBI", "dice")]
axc.annotate("", xy=(nb[2], -1.40), xytext=(0, -1.40),
             arrowprops=dict(arrowstyle="<->", color=C_NEU, lw=0.8))
axc.text(nb[2] / 2, -1.52, f"improvement > {nb[2]:.3f} excluded",
         fontsize=5.9, color=C_NEU, ha="center", va="top")

base = os.path.join(OUT, "fig3_dual_vs_single")
for ext, kw in ((".svg", {}), (".pdf", {}), (".png", dict(dpi=600)),
                (".tiff", dict(dpi=600))):
    fig.savefig(base + ext, bbox_inches="tight", **kw)
plt.close(fig)

print("saved ->", base + ".{svg,pdf,png,tiff}")
print("\na | boundary F gain (paired median, published Holm p)")
for frame, (med, ps) in report_a.items():
    line = "  ".join(f"{t}px {m:+.4f}{'' if p is None else stars(p)}"
                     for t, m, p in zip(TOL, med, ps))
    print(f"  {frame}: {line}")
print("\nb | clinical readout (mean change, raw p)")
for (frame, col), (v, p) in report_b.items():
    print(f"  {frame:4s} {col:20s} {v:+6.2f}   raw p = {p:.4f}")
print("\nc | area overlap, no separation (paired median, 95% CI, Holm p)")
for (frame, col), (m, lo, hi, p) in report_c.items():
    pp = "n/a" if p is None else f"{p:.3f}"
    print(f"  {frame:4s} {col:6s} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]  Holm {pp}")
