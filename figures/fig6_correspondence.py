# -*- coding: utf-8 -*-
"""Supplementary Fig. S3 | Does fusion gain track how well the two views correspond?

Panels
  a  Per-image fusion gain against cross-modal correspondence, both frames.
     Correspondence is MI(same lesion) - MI(permuted null). Spearman rho is
     reported raw and partialled on lesion area, because Dice is size-biased.
  b  The same gain split by working distance. The prediction under this proxy
     is distant >= near, which is the OPPOSITE of what lesion size alone would
     produce -- so it cannot be explained away by size.
  c  Directional asymmetry: gain from adding NBI to WLI against gain from
     adding WLI to NBI, paired within image pair.

Gain is always `ours` minus the single-modality baseline of the SAME frame, so
it is a within-frame quantity and no Dice is ever compared across ground truths.

Data: EGC-DualSeg results/predictions_*.csv (seed-averaged per image) and the
      packaged manifest for the correspondence measurements. No GPU.
Backend: Python / matplotlib (project default).
Output: fig6_correspondence.{svg,pdf,tiff,png}

Why MI and not lesion IoU
-------------------------
Naive lesion IoU correlates with WLI lesion area at r = +0.575: a large mask
intersects anything with higher prior probability, so a "correspondence" axis
built from it would partly plot lesion size. MI is computed on greyscale and
never touches a mask (r = -0.049 with lesion area). The two proxies correlate
only weakly with each other (r = +0.332) and are not interchangeable.

Palette discipline
  Blue / teal remain EXCLUSIVELY the evaluation frame. Working distance is
  neutral grey, as everywhere else. Red is reference lines and medians only.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

# ---- MANDATORY: editable SVG/PDF text (before any figure is created) -------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",     # text stays as <text> nodes, editable downstream
    "pdf.fonttype": 42,         # embedded TrueType, editable in PDF
    "font.size": 10,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.0,
    "legend.frameon": False,
})

# Repository root, resolved from this file so the scripts run from a clone.
# Override with EGC_REPO if the read-outs live elsewhere.
REPO = os.environ.get("EGC_REPO") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
MANIFEST = os.path.join(REPO, "data", "packaged", "manifest.csv")
OUT = os.path.dirname(os.path.abspath(__file__))

C_WLI = "#0F4D92"      # frame: white light
C_NBI = "#42949E"      # frame: image-enhanced
C_NEAR = "#4D4D4D"     # working distance: near
C_FAR = "#A8A8A8"      # working distance: distant
C_SIG = "#B64342"      # medians, reference lines
C_NEU = "#767676"

MM = 1 / 25.4
FINAL_WIDTH_MM = 180          # double-column
FINAL_HEIGHT_MM = 82
FIGW, FIGH = FINAL_WIDTH_MM * MM, FINAL_HEIGHT_MM * MM

FUSION = "ours"
FRAMES = {"WLI": dict(colour=C_WLI, single="wli_only", adds="NBI"),
          "NBI": dict(colour=C_NBI, single="nbi_only", adds="WLI")}
SCALES = [("near", C_NEAR, "Near"), ("far", C_FAR, "Distant")]
PANELS = ["a", "b", "c"]

# All three panels plot the same quantity, so they share one vertical scale.
# Giving b and c a tighter range would magnify medians of about 0.01 into
# something that looks like an effect, and a reader comparing panels would be
# misled. The range covers every point; nothing is clipped out of view.
YLIM = (-0.68, 0.80)

EXCLUSIONS = []


# ---- helpers --------------------------------------------------------------
def panel_label(ax, s, x, y):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=12,
            fontweight="bold", ha="left", va="bottom")


def jitter(n, spread):
    """Deterministic spread along a categorical axis; presentation only.

    A golden-ratio sequence rather than a random generator, so re-running this
    script reproduces the figure byte for byte.
    """
    k = (np.arange(n) * 0.6180339887498949) % 1.0
    return (k * 2.0 - 1.0) * spread


def _t_sf(t, df):
    """Upper tail of Student's t, falling back to the normal approximation."""
    try:
        from scipy.stats import t as _t
        return float(_t.sf(abs(t), df))
    except Exception:
        import math
        return 0.5 * math.erfc(abs(t) / math.sqrt(2.0))


def spearman(x, y):
    n = len(x)
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if n < 4 or abs(rho) >= 1:
        return rho, float("nan")
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    return rho, 2 * _t_sf(t, n - 2)


def partial_spearman(x, y, z):
    """Spearman rho between x and y with z held constant.

    Dice rises with lesion size (r = +0.562) and near-focus lesions fill 2.9x
    more of the field than distant ones, so an uncontrolled correlation here
    could be lesion size wearing a correspondence label. Using a within-image
    difference removes most of that; this removes the rest, and reporting both
    numbers lets a reader see how much of the effect the covariate absorbs.
    """
    n = len(x)
    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if n < 5 or denom == 0:
        return float("nan"), float("nan")
    rho = float((rxy - rxz * ryz) / denom)
    if abs(rho) >= 1:
        return rho, float("nan")
    t = rho * np.sqrt((n - 3) / (1 - rho ** 2))
    return rho, 2 * _t_sf(t, n - 3)


def wilcoxon_signed(d):
    """Two-sided Wilcoxon signed-rank against zero; returns (median, p)."""
    nz = d[d != 0]
    if len(nz) < 6:
        return float(np.median(d)), float("nan")
    try:
        from scipy.stats import wilcoxon as _w
        return float(np.median(d)), float(_w(nz).pvalue)
    except Exception:
        return float(np.median(d)), float("nan")


def p_text(p):
    if not np.isfinite(p):
        return "p n/a"
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.3f}"


# ---- load -----------------------------------------------------------------
paths = sorted(f for f in os.listdir(RESULTS) if f.startswith("predictions_"))
if not paths:
    raise SystemExit(f"no prediction CSVs in {RESULTS}")
pred = pd.concat([pd.read_csv(os.path.join(RESULTS, p), encoding="utf-8-sig")
                  for p in paths], ignore_index=True)

# Seeds averaged per image first, as everywhere else in this project.
img = pred.groupby(["case", "scale", "ref_modality", "config"], as_index=False).agg(
    dice=("dice", "mean"), gt_area_frac=("gt_area_frac", "mean"))

manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")
pairs = manifest.drop_duplicates(["case", "scale"])[["case", "scale",
                                                     "mi_same", "mi_null"]].copy()
pairs["mi_delta"] = pairs.mi_same - pairs.mi_null

rows = []
for frame, spec in FRAMES.items():
    sub = img[img.ref_modality == frame]
    ours = sub[sub.config == FUSION].set_index(["case", "scale"])
    base = sub[sub.config == spec["single"]].set_index(["case", "scale"])
    joined = ours.join(base, how="inner", rsuffix="_base")
    joined["gain"] = joined.dice - joined.dice_base
    joined = joined.reset_index()
    joined["frame"] = frame
    rows.append(joined[["case", "scale", "frame", "gain", "gt_area_frac"]])

gains = pd.concat(rows, ignore_index=True).merge(
    pairs[["case", "scale", "mi_delta"]], on=["case", "scale"], how="left")
_before = len(gains)
gains = gains.dropna(subset=["mi_delta", "gain"])
EXCLUSIONS.append(("images without a correspondence measurement",
                   _before, len(gains)))

# Patient level, as everywhere else in this manuscript. A patient's near-focus
# and distant pairs are two views of one lesion, not two observations, so
# panels a and c average them within a reference frame first. Panel b is the
# one place they stay apart, because the working distance is the variable it
# stratifies on; there each patient still contributes at most one value per
# group, so that panel was already at the patient level.
gains_pat = (gains.groupby(["case", "frame"], as_index=False)
             .agg(gain=("gain", "mean"),
                  gt_area_frac=("gt_area_frac", "mean"),
                  mi_delta=("mi_delta", "mean")))

# ---- figure ---------------------------------------------------------------
fig = plt.figure(figsize=(FIGW, FIGH))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.30, 1.05, 0.90], wspace=0.40,
              left=0.068, right=0.965, top=0.80, bottom=0.185)

# ---- a: gain vs correspondence -------------------------------------------
axa = fig.add_subplot(gs[0, 0])
axa.axhline(0, color=C_NEU, ls="--", lw=1.0, zorder=1)
notes, all_p, stats_a = [], [], []
for frame, spec in FRAMES.items():
    sub = gains_pat[gains_pat.frame == frame]
    axa.scatter(sub.mi_delta, sub.gain, s=16, alpha=0.7, lw=0,
                color=spec["colour"], zorder=2)
    coef = np.polyfit(sub.mi_delta, sub.gain, 1)
    xs = np.linspace(sub.mi_delta.min(), sub.mi_delta.max(), 50)
    axa.plot(xs, np.polyval(coef, xs), color=spec["colour"], lw=1.4,
             alpha=0.9, zorder=3)
    rho, p = spearman(sub.mi_delta.to_numpy(), sub.gain.to_numpy())
    prho, pp = partial_spearman(sub.mi_delta.to_numpy(), sub.gain.to_numpy(),
                                sub.gt_area_frac.to_numpy())
    notes.append((spec["colour"],
                  f"{frame}   ρ = {rho:+.2f},  partial = {prho:+.2f}"))
    all_p += [p, pp]
    stats_a.append((frame, rho, p, prho, pp))

# Every one of the four tests is far from significance, so one line carries
# them all and the per-frame lines stay short enough not to overrun the panel.
for i, (colour, txt) in enumerate(notes):
    axa.text(0.015, 0.975 - i * 0.072, txt, transform=axa.transAxes,
             ha="left", va="top", fontsize=7.4, color=colour)
axa.text(0.015, 0.975 - len(notes) * 0.072,
         f"all four p >= {min(all_p):.2f};  partial holds lesion area constant",
         transform=axa.transAxes, ha="left", va="top", fontsize=7.0,
         color=C_NEU, style="italic")

axa.set_xlabel("Correspondence,  MI(same) - MI(null)")
axa.set_ylabel("Fusion gain, Dice\n(ours - single modality)")
axa.set_ylim(*YLIM)
axa.set_title("Gain against correspondence", fontsize=9.5, pad=14)
axa.text(0.5, 1.015, f"n = {gains_pat.case.nunique()} patients",
         transform=axa.transAxes, ha="center", va="bottom", fontsize=7.2,
         color=C_NEU)
panel_label(axa, PANELS[0], x=-0.20, y=1.04)

# ---- b: by working distance ----------------------------------------------
axb = fig.add_subplot(gs[0, 1])
axb.axhline(0, color=C_NEU, ls="--", lw=1.0, zorder=1)
positions, ticks, labels = [], [], []
for fi, (frame, spec) in enumerate(FRAMES.items()):
    for si, (scale, grey, slabel) in enumerate(SCALES):
        x = fi * 2.5 + si
        sub = gains[(gains.frame == frame) & (gains.scale == scale)]
        values = sub.gain.to_numpy()
        axb.scatter(x + jitter(len(values), 0.17), values, s=6, lw=0,
                    alpha=0.45, color=spec["colour"], zorder=2)
        med, p = wilcoxon_signed(values)
        axb.plot([x - 0.30, x + 0.30], [med, med], color=C_SIG, lw=2.0,
                 zorder=4, solid_capstyle="butt")
        # Annotations in axes fractions, so they stay put if the shared
        # vertical range is ever changed.
        axb.text(x, 0.90, f"{med:+.3f}", transform=axb.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=7.2, color=C_NEU)
        ticks.append(x)
        labels.append(f"{slabel}\nn={len(values)}")
    axb.text(fi * 2.5 + 0.5, 0.975, f"{frame} frame",
             transform=axb.get_xaxis_transform(), ha="center", va="bottom",
             fontsize=8.5, color=spec["colour"])

axb.set_xticks(ticks)
axb.set_xticklabels(labels, fontsize=7.6)
axb.set_xlim(-0.7, 3.7)
axb.set_ylim(*YLIM)
# b and c inherit a's scale; repeating the tick labels three times crowds the
# panels and invites the reader to treat the scales as independent.
axb.set_yticklabels([])
# The panel's title states a comparison, so the comparison has to be tested.
# The earlier version tested each group against zero and left the reader to
# infer a difference from one being significant and the other not, which is the
# inference this project has already had to correct once.
for fi, (frame, spec) in enumerate(FRAMES.items()):
    w = (gains[gains.frame == frame]
         .pivot_table(index="case", columns="scale", values="gain").dropna())
    if w.empty or "near" not in w or "far" not in w:
        continue
    dmed, dp = wilcoxon_signed((w["far"] - w["near"]).to_numpy())
    axb.text(fi * 2.5 + 0.5, 0.035,
             f"distant - near {dmed:+.3f}, {p_text(dp)}",
             transform=axb.get_xaxis_transform(), ha="center", va="bottom",
             fontsize=6.8, color=C_NEU)
axb.set_title("Prediction: distant >= near", fontsize=9.5, pad=14)
panel_label(axb, PANELS[1], x=-0.16, y=1.04)

# ---- c: directional asymmetry --------------------------------------------
axc = fig.add_subplot(gs[0, 2])
axc.axhline(0, color=C_NEU, ls="--", lw=1.0, zorder=1)
wide = gains_pat.pivot_table(index="case", columns="frame",
                             values="gain").dropna()
directions = [("WLI", "NBI added\nto WLI"), ("NBI", "WLI added\nto NBI")]
for i, (frame, label) in enumerate(directions):
    values = wide[frame].to_numpy()
    colour = FRAMES[frame]["colour"]
    axc.scatter(i + jitter(len(values), 0.17), values, s=6, lw=0, alpha=0.45,
                color=colour, zorder=2)
    med, p = wilcoxon_signed(values)
    axc.plot([i - 0.30, i + 0.30], [med, med], color=C_SIG, lw=2.0, zorder=4,
             solid_capstyle="butt")
    axc.text(i, 0.90, f"{med:+.3f}", transform=axc.get_xaxis_transform(),
             ha="center", va="bottom", fontsize=7.2, color=C_NEU)

asym = wide["WLI"].to_numpy() - wide["NBI"].to_numpy()
asym_med, asym_p = wilcoxon_signed(asym)
supported = asym_med > 0
axc.text(0.5, 0.975, f"paired {asym_med:+.3f}, {p_text(asym_p)}",
         transform=axc.transAxes, ha="center", va="bottom",
         fontsize=7.4, color=C_NEU)
axc.text(0.5, 0.035, "prediction NOT supported" if not supported
         else "prediction supported", transform=axc.transAxes,
         ha="center", va="bottom", fontsize=7.6, color=C_SIG, fontweight="bold")

axc.set_xticks(range(len(directions)))
axc.set_xticklabels([label for _, label in directions], fontsize=7.6)
axc.set_xlim(-0.7, 1.7)
axc.set_ylim(*YLIM)
axc.set_yticklabels([])
axc.set_title("Prediction: NBI into WLI larger", fontsize=9.5, pad=14)
panel_label(axc, PANELS[2], x=-0.16, y=1.04)

fig.text(0.5, 0.012,
         "All three panels share one vertical scale. Gain is always measured "
         "within a frame against that frame's own baseline; no Dice is "
         "compared across ground truths.",
         ha="center", va="bottom", fontsize=7.2, color=C_NEU, style="italic")

base = os.path.join(OUT, "fig6_correspondence")
fig.savefig(base + ".svg")
fig.savefig(base + ".pdf")
fig.savefig(base + ".tiff", dpi=600)
fig.savefig(base + ".png", dpi=600)
plt.close(fig)

# ---- console record -------------------------------------------------------
print(f"images with gain and correspondence = {len(gains)}   "
      f"patients = {gains_pat.case.nunique()}")
for frame in FRAMES:
    # Patient level, matching panel a. The console used to recompute this on
    # `gains`, so it printed a per-image rho next to a per-patient figure.
    sub = gains_pat[gains_pat.frame == frame]
    rho, p = spearman(sub.mi_delta.to_numpy(), sub.gain.to_numpy())
    prho, pp = partial_spearman(sub.mi_delta.to_numpy(), sub.gain.to_numpy(),
                                sub.gt_area_frac.to_numpy())
    print(f"\n{frame} frame  n = {len(sub)} patients")
    print(f"  gain vs correspondence  rho {rho:+.3f} ({p_text(p)})   "
          f"partial on lesion area {prho:+.3f} ({p_text(pp)})")
    w = (gains[gains.frame == frame]
         .pivot_table(index="case", columns="scale", values="gain").dropna())
    if not w.empty and {"near", "far"} <= set(w.columns):
        dmed, dp = wilcoxon_signed((w["far"] - w["near"]).to_numpy())
        print(f"  distant - near (paired, n = {len(w)})  "
              f"median {dmed:+.4f} ({p_text(dp)})")
    by_scale = gains[gains.frame == frame]
    for scale, _, slabel in SCALES:
        v = by_scale[by_scale.scale == scale].gain.to_numpy()
        med, sp = wilcoxon_signed(v)
        print(f"  {slabel:8s} n {len(v):3d}  median gain {med:+.4f}  {p_text(sp)}")

print(f"\ndirectional asymmetry (NBI into WLI) - (WLI into NBI): "
      f"median {asym_med:+.4f}  {p_text(asym_p)}  -> "
      f"{'supported' if supported else 'NOT supported'}")
print("\nexclusions (before -> after)")
for what, before, after in EXCLUSIONS:
    print(f"  {what}: {before} -> {after}  (dropped {before - after})")
print("saved -> fig6_correspondence.{svg,pdf,tiff,png}")
