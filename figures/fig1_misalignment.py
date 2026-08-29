# -*- coding: utf-8 -*-
"""Fig. 2 | Paired WLI-NBI frames of the same lesion are spatially unaligned.

Core conclusion this figure must defend
---------------------------------------
The two frames of one lesion do not correspond pixel for pixel, and image
content alone is a weak cue for recovering that correspondence. Everything a
fusion scheme can assume about alignment has to survive these three numbers.

Rebuilt 2026-08-24: patient-level, modelling cohort
---------------------------------------------------
The previous version drew all 158 available pairs from 81 cases and quoted
per-pair medians (IoU 0.279, MI AUC 0.714). Two problems, both fixed here:

  cohort   158/81 predates the removal of duplicated records and oesophageal
           lesions. The modelling cohort is 150 pairs from 77 patients, and it
           is the cohort every other figure and every statistic in the
           manuscript uses. `results/registration_feasibility.csv` carries
           exactly those 150 (case, scale) rows and is used here as the filter.

  unit     146 of the patient-by-frame combinations contribute both a
           near-focus and a distant pair, so two pairs from one patient are not
           independent. Panels b-d are therefore averaged within a patient
           first; n = 77 throughout. This reproduces Results 3.1 exactly:
           overlap 0.342, area ratio 1.82x, 83.1% above the diagonal, AUC 0.780.

Panel a keeps the per-pair unit, because an exemplar IS one pair; its caption
says so.

Panels
  a  Representative lesion pairs at the 10th / 50th / 90th percentile of
     cross-modal lesion overlap (near-focus). 2 rows (WLI / NBI) x 3 columns.
  b  Per-patient cross-modal lesion overlap - the pixel correspondence that
     channel-concatenation fusion assumes.
  c  Per-patient lesion-area ratio, NBI over WLI.
  d  Per-patient mutual information, true partner frame against a partner drawn
     from a different patient, scattered on the identity line.

Data: _sam_output/fov_consistency.csv, filtered by
      EGC-DualSeg/results/registration_feasibility.csv (the 150-pair cohort).
Backend: Python / matplotlib (project default).
Output: fig1_misalignment.{svg,pdf,tiff,png}

Palette discipline
  Blue / teal are reserved EXCLUSIVELY for frame identity (WLI / NBI) and so
  appear only on panel a's borders. Panels b-d describe a property of the PAIR,
  which belongs to neither frame, and are therefore drawn in neutral greys.
  One red is reserved for reference lines and medians.
"""
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

# ---- MANDATORY: editable SVG/PDF text (before any figure is created) -------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.0,
    "legend.frameon": False,
})

# The endoscopic archive is NOT public: it holds identifiable patient
# images. Set EGC_RAW to a local copy to regenerate the panels that need it.
REPO = os.environ.get("EGC_REPO") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("EGC_RAW", "")
CSV = os.path.join(ROOT, "_sam_output", "fov_consistency.csv")
MASKS = os.path.join(ROOT, "_sam_output", "masks")
COHORT = os.path.join(REPO, "results", "registration_feasibility.csv")
OUT = os.path.dirname(os.path.abspath(__file__))

C_WLI = "#0F4D92"      # frame identity: white light
C_NBI = "#42949E"      # frame identity: narrow band
C_INK = "#4D4D4D"      # pair-level data (belongs to neither frame)
C_PALE = "#BFBFBF"
C_SIG = "#B64342"      # medians / reference lines
C_NEU = "#767676"
C_CONTOUR = (255, 215, 0)   # gold lesion contour, legible on pink and green mucosa

MM = 1 / 25.4
FINAL_WIDTH_MM = 180          # double-column
FINAL_HEIGHT_MM = 150
FIGW, FIGH = FINAL_WIDTH_MM * MM, FINAL_HEIGHT_MM * MM
BOOT, RNG_SEED = 5000, 0

# --- exemplar-selection rules (panel a only) -------------------------------
# The cohort filter already removes duplicated records and oesophageal lesions,
# so only the multifocal cases are excluded here: their specimen measurement
# cannot be attributed to the lesion in the frame, which makes them misleading
# to show as a single lesion.
EXEMPLAR_EXCLUDE = {"病例14", "病例44", "病例21", "病例27", "病例56",
                    "病例19", "病例77", "病例5"}
# Require both lesions to occupy < MAX_FIELD_FRAC of their field, so panel a
# illustrates POSITIONAL offset; pure scale mismatch is panel c.
MAX_FIELD_FRAC = 0.60


# --------------------------------------------------------------------------
def fov_bbox(path):
    """Valid endoscopic field: drop black borders and the sparse-text info panel."""
    g = np.asarray(Image.open(path).convert("L"))
    bright = g > 25

    def span(f, thr=0.5):
        idx = np.where(f > thr)[0]
        if not len(idx):
            return 0, len(f)
        segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        s = max(segs, key=len)
        return int(s[0]), int(s[-1]) + 1

    x0, x1 = span(bright.mean(0))
    y0, y1 = span(bright.mean(1))
    if (x1 - x0) < 0.3 * g.shape[1] or (y1 - y0) < 0.3 * g.shape[0]:
        return 0, 0, g.shape[1], g.shape[0]
    return x0, y0, x1, y1


def mask_for(img_path):
    """Masks are named after the JSON stem, not the image stem (batch-2 differ)."""
    case_dir = os.path.dirname(img_path)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    batch = os.path.basename(os.path.dirname(case_dir))
    case = os.path.basename(case_dir)
    mdir = os.path.join(MASKS, batch, case)
    if not os.path.isdir(mdir):
        return None
    for f in os.listdir(mdir):
        if os.path.splitext(f)[0].startswith(stem):
            return os.path.join(mdir, f)
    return None


def field_frac(img_path):
    """Fraction of the endoscopic field covered by the lesion mask."""
    bb = fov_bbox(img_path)
    mp = mask_for(img_path)
    if not mp:
        return 1.0
    M = np.array(Image.open(mp))[bb[1]:bb[3], bb[0]:bb[2]] > 127
    return float(M.mean())


def tile(img_path, size=560):
    """Square centre crop of the endoscopic field, lesion contour drawn in gold."""
    bb = fov_bbox(img_path)
    im = Image.open(img_path).convert("RGB").crop(bb)
    a = np.asarray(im).copy()
    mp = mask_for(img_path)
    frac = None
    if mp:
        M = np.array(Image.open(mp))[bb[1]:bb[3], bb[0]:bb[2]] > 127
        frac = float(M.mean())
        er = np.zeros_like(M)
        er[1:-1, 1:-1] = (M[1:-1, 1:-1] & M[:-2, 1:-1] & M[2:, 1:-1]
                          & M[1:-1, :-2] & M[1:-1, 2:])
        edge = M & ~er
        k = max(1, int(round(min(a.shape[:2]) / 380)))
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                a[np.roll(np.roll(edge, dy, 0), dx, 1)] = C_CONTOUR
    h, w = a.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    a = a[y0:y0 + s, x0:x0 + s]
    return np.asarray(Image.fromarray(a).resize((size, size), Image.LANCZOS)), frac


def panel_label(ax, s, x, y):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=12,
            fontweight="bold", ha="left", va="bottom")


def vline_note(ax, x, text, color=C_SIG):
    """Rotated annotation riding on a vertical reference line (never collides)."""
    ax.axvline(x, color=color, ls="--", lw=1.3, zorder=3)
    ax.text(x, 0.98, text, transform=ax.get_xaxis_transform(),
            rotation=90, ha="right", va="top", fontsize=9, color=color, zorder=5,
            bbox=dict(fc="white", ec="none", alpha=0.78, pad=1.2))


def boot_ci(v, stat=np.median):
    rng = np.random.default_rng(RNG_SEED)
    v = np.asarray(v, float)
    draws = [stat(rng.choice(v, len(v))) for _ in range(BOOT)]
    return stat(v), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def auc(pos, neg):
    """Mann-Whitney AUC = P(pos > neg)."""
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ---- load and restrict to the modelling cohort ----------------------------
rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
for r in rows:                                   # strip a stray BOM on the key
    if "\ufeffcase" in r:
        r["case"] = r.pop("\ufeffcase")

cohort = {(r.get("case") or r.get("\ufeffcase"), r["scale"])
          for r in csv.DictReader(open(COHORT, encoding="utf-8-sig"))}

dropped = [r for r in rows if (r["case"], r["scale"]) not in cohort]
rows = [r for r in rows if (r["case"], r["scale"]) in cohort]

f = lambda r, k: float(r[k])

# ---- aggregate to the patient --------------------------------------------
# Within a patient the near-focus and distant pair are two views of the same
# lesion, so they are averaged rather than treated as two observations.
by_case = {}
for r in rows:
    by_case.setdefault(r["case"], []).append(r)
cases = sorted(by_case)

p_iou = np.array([np.mean([f(r, "naive_iou") for r in by_case[c]]) for c in cases])
p_lr = np.array([np.mean([f(r, "log2_ratio") for r in by_case[c]]) for c in cases])
p_mis = np.array([np.mean([f(r, "mi_same") for r in by_case[c]]) for c in cases])
p_mnu = np.array([np.mean([f(r, "mi_null") for r in by_case[c]]) for c in cases])
p_ratio = 2.0 ** p_lr

IOU_M, IOU_LO, IOU_HI = boot_ci(p_iou)
RAT_M, RAT_LO, RAT_HI = boot_ci(p_ratio)
MID_M, MID_LO, MID_HI = boot_ci(p_mis - p_mnu)
AUC = auc(p_mis, p_mnu)
frac_above = (p_mis > p_mnu).mean() * 100

# ---- exemplars (per-pair, near focus, inside the cohort) ------------------
pool = [r for r in rows if r["scale"] == "near"
        and r["case"] not in EXEMPLAR_EXCLUDE
        and field_frac(r["wli"]) < MAX_FIELD_FRAC
        and field_frac(r["nbi"]) < MAX_FIELD_FRAC]
pool.sort(key=lambda r: f(r, "naive_iou"))
picks = [pool[int(round(p / 100 * (len(pool) - 1)))] for p in (10, 50, 90)]
PCT_LAB = ["10th percentile", "50th percentile", "90th percentile"]

# ---- figure ---------------------------------------------------------------
fig = plt.figure(figsize=(FIGW, FIGH))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.62, 1.0],
              hspace=0.34, wspace=0.36,
              left=0.085, right=0.985, top=0.905, bottom=0.085)

gs_a = gs[0, :].subgridspec(2, 3, hspace=0.05, wspace=0.06)
for col, r in enumerate(picks):
    for row, (key, cname, ccol) in enumerate(
            [("wli", "WLI", C_WLI), ("nbi", "NBI", C_NBI)]):
        ax = fig.add_subplot(gs_a[row, col])
        img, frac = tile(r[key])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(1.5); sp.set_color(ccol)
        ax.text(0.035, 0.035, f"{frac*100:.0f}% of field", transform=ax.transAxes,
                fontsize=8.5, color="white", ha="left", va="bottom",
                bbox=dict(fc="black", ec="none", alpha=0.45, pad=1.6))
        if col == 0:
            ax.set_ylabel(cname, color=ccol, fontweight="bold", labelpad=5)
        if row == 0:
            ax.set_title(f"{PCT_LAB[col]}\noverlap = {f(r,'naive_iou'):.2f}",
                         fontsize=9.5, pad=4, color=C_NEU)
        if col == 0 and row == 0:
            panel_label(ax, "a", x=-0.20, y=1.16)

# -- b : per-patient overlap ------------------------------------------------
axb = fig.add_subplot(gs[1, 0])
bins = np.linspace(0, 1, 21)
axb.hist(p_iou, bins=bins, histtype="stepfilled", color=C_PALE, lw=0, zorder=1)
axb.hist(p_iou, bins=bins, histtype="step", color=C_INK, lw=1.6, zorder=2)
# One red region only. The earlier version shaded "<0.3" in the same hue as the
# CI band, which read as two overlapping claims; the <0.3 share is now stated
# in words instead.
axb.axvspan(IOU_LO, IOU_HI, color=C_SIG, alpha=0.15, lw=0, zorder=3)
vline_note(axb, IOU_M, f"median {IOU_M:.3f}")
axb.set_xlabel("Cross-modal lesion overlap")
axb.set_ylabel(f"Patients (n = {len(cases)})")
axb.set_xlim(0, 1)
axb.text(0.97, 0.92, f"{(p_iou < 0.3).mean()*100:.0f}% of patients\nbelow 0.3",
         transform=axb.transAxes, ha="right", va="top", fontsize=9, color=C_NEU,
         linespacing=1.3)
panel_label(axb, "b", x=-0.24, y=1.02)

# -- c : per-patient area ratio --------------------------------------------
# Drawn on a log2 axis because the quantity is a ratio, but ticked in fold
# units so the reader never has to exponentiate a label.
axc = fig.add_subplot(gs[1, 1])
cb = np.linspace(-1.6, 3.4, 21)
axc.hist(p_lr, bins=cb, histtype="stepfilled", color=C_PALE, lw=0)
axc.hist(p_lr, bins=cb, histtype="step", color=C_INK, lw=1.5)
axc.axvline(0, color="black", lw=1.0, zorder=3)
axc.text(0, 0.98, "equal size", transform=axc.get_xaxis_transform(),
         rotation=90, ha="right", va="top", fontsize=9, color="black")
vline_note(axc, np.median(p_lr), f"median {RAT_M:.2f}$\\times$")
# The data span roughly 0.4x to 9x; the old -2.5 window left two tick labels
# colliding in an empty margin.
ticks = [-1, 0, 1, 2, 3]
axc.set_xlim(-1.6, 3.4)
axc.set_xticks(ticks)
axc.set_xticklabels([f"{2.0**t:g}$\\times$" for t in ticks])
axc.set_xlabel("NBI / WLI lesion area")
axc.set_ylabel(f"Patients (n = {len(cases)})")
panel_label(axc, "c", x=-0.24, y=1.02)

# -- d : per-patient MI, true partner vs partner from another patient -------
axd = fig.add_subplot(gs[1, 2])
lo = min(p_mis.min(), p_mnu.min()) * 0.9
hi = max(p_mis.max(), p_mnu.max()) * 1.05
axd.plot([lo, hi], [lo, hi], color=C_NEU, ls="--", lw=1.1, zorder=1)
above = p_mis > p_mnu
axd.scatter(p_mnu[above], p_mis[above], s=16, color=C_INK, alpha=0.8, lw=0, zorder=2)
axd.scatter(p_mnu[~above], p_mis[~above], s=16, facecolor="none",
            edgecolor=C_PALE, lw=1.0, zorder=2)
axd.set_xlim(lo, hi); axd.set_ylim(lo, hi); axd.set_aspect("equal")
axd.set_xlabel("MI, partner from another patient")
axd.set_ylabel("MI, true partner")
axd.text(0.97, 0.16,
         f"{frac_above:.1f}% of patients\nabove diagonal\nAUC = {AUC:.3f}",
         transform=axd.transAxes, va="bottom", ha="right", fontsize=9,
         linespacing=1.35)
panel_label(axd, "d", x=-0.30, y=1.02)

base = os.path.join(OUT, "fig1_misalignment")
fig.savefig(base + ".svg")
fig.savefig(base + ".pdf")
fig.savefig(base + ".tiff", dpi=600)
fig.savefig(base + ".png", dpi=600)
plt.close(fig)

print(f"cohort filter: kept {len(rows)} pairs / {len(cases)} patients "
      f"(dropped {len(dropped)} pairs outside the modelling cohort)")
print(f"  overlap     median {IOU_M:.3f}  95% CI [{IOU_LO:.3f}, {IOU_HI:.3f}]"
      f"   <0.3 in {100*(p_iou<0.3).mean():.1f}% of patients")
print(f"  area ratio  median {RAT_M:.2f}x  95% CI [{RAT_LO:.2f}, {RAT_HI:.2f}]")
print(f"  MI diff     median {MID_M:+.3f}  95% CI [{MID_LO:+.3f}, {MID_HI:+.3f}]")
print(f"  MI          above diagonal {frac_above:.1f}%   AUC {AUC:.3f}")
print(f"exemplar pool = {len(pool)} near-focus pairs in cohort "
      f"(after multifocal exclusion and the <{MAX_FIELD_FRAC:.0%} field rule)")
for lab, r in zip(PCT_LAB, picks):
    print(f"  {lab:18s} {r['case']:8s} overlap {f(r,'naive_iou'):.3f}")
print("saved -> fig1_misalignment.{svg,pdf,tiff,png}")
