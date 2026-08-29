# -*- coding: utf-8 -*-
"""Fig. 5 | Endpoints taken from the resection pathology report.

Core conclusion this figure must defend
---------------------------------------
Scored against a measurement that owes nothing to the clinician masks, the
predicted lesion extent tracks the specimen. The grade endpoint does not
separate, and the control declared at the design stage says why: the gradeable
sample is small, not the pipeline broken.

Rebuilt 2026-08-25: the section's positive result finally has a panel
-------------------------------------------------------------------
The earlier version had four panels, all of them about the grade endpoint,
which is the one result in Section 3.5 that is null. The extent correlation,
which is the section's positive and annotation-independent finding, had no panel
anywhere in the manuscript. Two panels that Section 3.5 never refers to are gone:

  per-patient change in predicted probability   the text makes no claim from it
  representative low-grade / high-grade lesions  showing "representative" cases
                                                 of a class the model cannot
                                                 discriminate implies a
                                                 discrimination the data deny

Panels, in the order the section argues
---------------------------------------
  a  Predicted lesion extent against the microscopically measured long axis of
     the resection specimen. Nothing on either axis comes from the segmentation
     masks the models were trained on.
  b  ROC for histological grade, with DeLong intervals in the key.
  c  ROC for the macroscopic-type positive control. Read it before b.

Data: EGC-DualSeg results/predictions_*.csv and results/grade_*.csv, plus the
      packaged manifest and .npz for the specimen measurements and the reference
      masks. The extent panel imports its two helpers from
      scripts/size_correlation.py, so the figure and the script cannot compute
      the quantity differently. No GPU.
Backend: Python / matplotlib (project default).
Output: fig5_pathology.{svg,pdf,tiff,png}

Palette discipline
  Blue / teal remain EXCLUSIVELY frame identity. Single versus dual is line
  style, never a second hue. Red is the chance diagonal and reference lines;
  grey is neutral annotation and a configuration that did not separate.
"""
import os
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

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
# Overridable so the layout can be exercised against a scratch copy of the CSVs
# before the real Stage-2 run exists.
RESULTS = os.environ.get("EGC_RESULTS") or os.path.join(REPO, "results")
NPZ = os.path.join(REPO, "data", "packaged", "egc_dualseg_384.npz")
MANIFEST = os.path.join(REPO, "data", "packaged", "manifest.csv")
OUT = os.path.dirname(os.path.abspath(__file__))

# DeLong lives in the repository next to the analysis that defines it; importing
# rather than copying keeps one implementation, already checked against a
# rank-based AUC including ties.
sys.path.insert(0, os.path.join(REPO, "scripts"))
from summarise_grade import delong_auc_ci, delong_paired_test   # noqa: E402

C_WLI = "#0F4D92"      # reference modality of the arm
C_NBI = "#42949E"
C_LOW = "#A8A8A8"      # grade class: low grade
C_HIGH = "#4D4D4D"     # grade class: high grade or above
C_SIG = "#B64342"      # chance line, medians
C_NEU = "#767676"
C_CONTOUR = (255, 215, 0)   # gold lesion contour, legible on gastric mucosa

MM = 1 / 25.4
FINAL_WIDTH_MM = 180          # double-column
FINAL_HEIGHT_MM = 150
FIGW, FIGH = FINAL_WIDTH_MM * MM, FINAL_HEIGHT_MM * MM

PANELS = ["a", "b", "c", "d"]
# (config in the CSV, display label, colour, line style)
ARMS = [
    ("wli_only_wli", "WLI only",        C_WLI, "--"),
    ("ours_wli",     "WLI + NBI",       C_WLI, "-"),
    ("nbi_only_nbi", "NBI only",        C_NBI, "--"),
    ("ours_nbi",     "NBI + WLI",       C_NBI, "-"),
]
FRAME_PAIRS = {"WLI": ("wli_only_wli", "ours_wli"),
               "NBI": ("nbi_only_nbi", "ours_nbi")}
CLASS_LABEL = {0: "Low grade", 1: "High grade or above"}

EXCLUSIONS = []


# ---- helpers --------------------------------------------------------------
def panel_label(ax, s, x, y):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=12,
            fontweight="bold", ha="left", va="bottom")


def roc_curve(y_true, y_prob):
    """False-positive and true-positive rates, ties handled by shared threshold."""
    order = np.argsort(-y_prob, kind="mergesort")
    y = y_true[order]
    p = y_prob[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    keep = np.append(np.diff(p) != 0, True)      # one point per distinct score
    tps, fps = tps[keep], fps[keep]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return (np.concatenate([[0], fps / n_neg, [1]]),
            np.concatenate([[0], tps / n_pos, [1]]))


def load_grade(results_dir: str, target: str) -> pd.DataFrame:
    """One probability per patient per arm: folds concatenated, seeds averaged."""
    names = [f for f in os.listdir(results_dir)
             if f.startswith("grade_") and f.endswith(".csv")]
    if not names:
        return pd.DataFrame()
    frames = [pd.read_csv(os.path.join(results_dir, n), encoding="utf-8-sig")
              for n in names]
    df = pd.concat(frames, ignore_index=True)
    want_macro = target == "macro"
    df = df[df.config.str.startswith("macro_") == want_macro]
    if df.empty:
        return df
    if want_macro:
        df["config"] = df.config.str.replace("^macro_", "", regex=True)
    # Each patient is held out in exactly one fold, so this averages seeds only.
    return (df.groupby(["config", "case"], as_index=False)
              .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"),
                   seeds=("seed", "nunique"), folds=("fold", "nunique")))


def arm_vectors(table: pd.DataFrame, config: str):
    sub = table[table.config == config].sort_values("case")
    return (sub.case.to_numpy(), sub.y_true.to_numpy().astype(int),
            sub.y_prob.to_numpy())


def require(table: pd.DataFrame, target: str):
    missing = [c for c, *_ in ARMS if c not in set(table.config)]
    if missing:
        raise SystemExit(
            f"Fig. 5 needs all four arms for target '{target}'. Missing: "
            + ", ".join(missing) + "\n\n"
            "Stage 2 reads the Stage-1 checkpoints, so it runs where those live "
            "(Colab notebook section 11):\n"
            "  python scripts/train_grade.py --seg-config configs/wli_only.yaml "
            "--ref WLI --target " + target + "\n"
            "  python scripts/train_grade.py --seg-config configs/ours.yaml "
            "--ref WLI --target " + target + "\n"
            "  python scripts/train_grade.py --seg-config configs/nbi_only.yaml "
            "--ref NBI --target " + target + "\n"
            "  python scripts/train_grade.py --seg-config configs/ours.yaml "
            "--ref NBI --target " + target)


def draw_roc(ax, table, letter, title, subtitle=None):
    ax.plot([0, 1], [0, 1], color=C_SIG, ls=":", lw=1.1, zorder=1)
    entries = []
    for config, label, colour, style in ARMS:
        cases, y, p = arm_vectors(table, config)
        fpr, tpr = roc_curve(y, p)
        ax.plot(fpr, tpr, color=colour, ls=style, lw=1.5, zorder=3)
        auc, lo, hi = delong_auc_ci(y, p)
        entries.append((label, colour, style, auc, lo, hi, len(cases)))

    handles = [Line2D([], [], color=c, ls=s, lw=1.5,
                      label=f"{lab}   {auc:.2f} [{lo:.2f}, {hi:.2f}]")
               for lab, c, s, auc, lo, hi, _ in entries]
    leg = ax.legend(handles=handles, fontsize=7.0, loc="lower right",
                    handlelength=2.0, labelspacing=0.30, borderpad=0.35,
                    frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor(C_NEU)
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_alpha(0.92)
    leg.set_zorder(6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title, fontsize=9.5, pad=17)
    if subtitle:
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=7.4, color=C_NEU)
    panel_label(ax, letter, x=-0.26, y=1.04)
    return entries


def field_tile(images, idx, mask=None):
    """One packaged frame, with the reference lesion outlined in gold."""
    img = images[idx].copy()
    if mask is not None:
        m = mask > 127
        edge = m ^ np.pad(m, 1, mode="edge")[2:, 1:-1]      # 1-px vertical edge
        edge |= m ^ np.pad(m, 1, mode="edge")[1:-1, 2:]     # horizontal edge
        img[edge] = C_CONTOUR
    return img


def load_extent():
    """Per-patient predicted extent against the specimen measurement.

    The two parsers are imported from scripts/size_correlation.py rather than
    copied, so this panel and the number quoted in the text cannot drift apart.
    Near-focus, white-light frame, which is the unit Results 3.5 states: the
    report gives one length per lesion and it cannot be attributed across frames.
    """
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from size_correlation import long_axis_mm, gt_long_axis

    files = sorted(glob.glob(os.path.join(RESULTS, "predictions_*.csv")))
    pred = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files],
                     ignore_index=True)
    manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    ruler = (manifest.drop_duplicates("case")
             .assign(long_mm=lambda d: d.path_mm.map(long_axis_mm))
             .dropna(subset=["long_mm"]).set_index("case").long_mm)

    key = ["case", "scale", "ref_modality", "config"]
    img = pred.groupby(key, as_index=False).agg(
        pred_long_axis_px=("pred_long_axis_px", "mean"),
        content_w=("content_w", "first"))
    img["long_frac"] = 100 * img.pred_long_axis_px / img.content_w
    img = pd.concat([img[key + ["long_frac"]], gt_long_axis(NPZ, manifest)],
                    ignore_index=True)
    img = img[(img.scale == "near") & (img.ref_modality == "WLI")].copy()
    img["path_long_mm"] = img.case.map(ruler)
    return img.dropna(subset=["path_long_mm"])


def spearman(x, y):
    """Spearman rho with its two-sided p, from ranks."""
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


# ---- load -----------------------------------------------------------------
grade = load_grade(RESULTS, "grade")
macro = load_grade(RESULTS, "macro")
if grade.empty:
    raise SystemExit(
        "no grade_*.csv in " + RESULTS + "\n\n"
        "Stage 2 has not been run yet. It needs the Stage-1 checkpoints, which "
        "live on Drive, so run Colab notebook section 11 first -- positive "
        "control ('--target macro') before the grade endpoint.")
require(grade, "grade")
require(macro, "macro")

# ---- figure ---------------------------------------------------------------
# Three panels, in the order Section 3.5 makes its argument: the positive,
# annotation-independent result first, then the endpoint that did not separate,
# then the control that explains why it did not.
fig = plt.figure(figsize=(FIGW, FIGH * 0.62))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.24, 0.88, 0.88],
              wspace=0.46, left=0.070, right=0.985, top=0.82, bottom=0.19)

# ---- a: lesion extent against the resection specimen ----------------------
# Four configurations are compared, but only the proposed one is scattered:
# 44 points four times over would hide the very correlation the panel exists to
# show. The other three are carried as rho values in the key, which is the form
# the comparison is actually made in.
axa = fig.add_subplot(gs[0, 0])
extent = load_extent()
EXTENT_ROWS = [("ours", "Proposed", C_WLI),
               ("wli_only", "Single modality", C_NEU),
               ("ground_truth", "Clinician mask", C_NEU),
               ("early_fusion_wli", "Channel concatenation", C_NEU)]
extent_stats = []
for cfg, label, colour in EXTENT_ROWS:
    sub = extent[extent.config == cfg]
    if len(sub) < 6:
        continue
    rho, p = spearman(sub.path_long_mm.to_numpy(), sub.long_frac.to_numpy())
    extent_stats.append((label, rho, p, len(sub)))
    if cfg != "ours":
        continue
    axa.scatter(sub.path_long_mm, sub.long_frac, s=22, alpha=0.85, lw=0,
                color=C_WLI, zorder=3)
    lx = np.log10(sub.path_long_mm.to_numpy())
    coef = np.polyfit(lx, sub.long_frac.to_numpy(), 1)
    xs = np.linspace(lx.min(), lx.max(), 50)
    axa.plot(10 ** xs, np.polyval(coef, xs), color=C_WLI, lw=1.3, alpha=0.8,
             zorder=2)

axa.set_xscale("log")
# matplotlib labels log minor ticks by default, which here collides with the
# major labels; the minor formatter has to be silenced explicitly.
from matplotlib.ticker import NullFormatter, NullLocator
axa.xaxis.set_minor_locator(NullLocator())
axa.xaxis.set_minor_formatter(NullFormatter())
axa.set_xticks([3, 5, 10, 20, 30])
axa.set_xticklabels(["3", "5", "10", "20", "30"])
# Headroom below the cloud so the key sits on empty axes rather than on data.
axa.set_ylim(14, 95)
axa.set_xlabel("Microscopic long axis of the specimen (mm, log scale)", fontsize=8)
axa.set_ylabel("Predicted long axis\n(% of field width)", fontsize=8)
axa.set_title("Lesion extent", fontsize=9.5, pad=17)
axa.text(0.5, 1.015, "no clinician mask enters either axis",
         transform=axa.transAxes, ha="center", va="bottom",
         fontsize=7.4, color=C_NEU)
key = "\n".join(f"{lab:<22s}{rho:+.2f}   p = {p:.3f}"
                 for lab, rho, p, _ in extent_stats)
n_ext = extent_stats[0][3] if extent_stats else 0
axa.text(0.975, 0.035,
         f"Spearman's ρ, n = {n_ext} patients\n" + key,
         transform=axa.transAxes, ha="right", va="bottom", fontsize=6.2,
         family="monospace", color=C_NEU, linespacing=1.45, zorder=6,
         bbox=dict(fc="white", ec=C_NEU, lw=0.4, alpha=0.92, pad=3.0))
panel_label(axa, PANELS[0], x=-0.24, y=1.04)

# ---- b: grade ROC ---------------------------------------------------------
axb = fig.add_subplot(gs[0, 1])
n_grade = grade[grade.config == ARMS[0][0]].shape[0]
entries_grade = draw_roc(
    axb, grade, PANELS[1], "Histological grade",
    subtitle=f"n = {n_grade} patients, one ground truth for all")

# ---- c: positive control --------------------------------------------------
axc = fig.add_subplot(gs[0, 2])
n_macro = macro[macro.config == ARMS[0][0]].shape[0]
entries_macro = draw_roc(
    axc, macro, PANELS[2], "Macroscopic type (control)",
    subtitle=f"n = {n_macro}, declared at the design stage")

# The paired within-frame contrasts are no longer drawn, but they are still
# computed and printed: the text quotes them and they must stay reproducible.
frame_stats = {}
for frame, (single, dual) in FRAME_PAIRS.items():
    cases_s, y_s, p_s = arm_vectors(grade, single)
    cases_d, y_d, p_d = arm_vectors(grade, dual)
    common = sorted(set(cases_s) & set(cases_d))
    idx_s = [list(cases_s).index(c) for c in common]
    idx_d = [list(cases_d).index(c) for c in common]
    delta, z, p = delong_paired_test(np.asarray(y_s)[idx_s],
                                     np.asarray(p_d)[idx_d],
                                     np.asarray(p_s)[idx_s])
    frame_stats[frame] = dict(delta=delta, z=z, p=p, n=len(common))

base = os.path.join(OUT, "fig5_pathology")
fig.savefig(base + ".svg", bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
fig.savefig(base + ".tiff", dpi=600, bbox_inches="tight")
fig.savefig(base + ".png", dpi=600, bbox_inches="tight")
plt.close(fig)

# ---- console record -------------------------------------------------------
print(f"grade: n = {n_grade} patients   macro control: n = {n_macro}")
print(f"seeds per patient: {sorted(grade.seeds.unique())}  "
      f"folds per patient: {sorted(grade.folds.unique())}")
for name, entries in (("grade", entries_grade), ("macro control", entries_macro)):
    print(f"\n{name}")
    for lab, _, _, auc, lo, hi, n in entries:
        print(f"  {lab:12s} n {n:3d}  AUC {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
print("\npaired DeLong, dual minus single, within frame")
for frame, s in frame_stats.items():
    print(f"  {frame} frame  delta {s['delta']:+.3f}  z {s['z']:+.2f}  "
          f"p {s['p']:.4f}")
print("\nextent panel: near-focus, WLI frame, one measurement per patient")
for lab, rho, p, n_ in extent_stats:
    print(f"  {lab:<24s} rho {rho:+.3f}  p {p:.4f}  n {n_}")
print("saved -> fig5_pathology.{svg,pdf,tiff,png}")
